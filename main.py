import time
from dataclasses import dataclass
from typing import Literal, Tuple

import torch
from jaxtyping import Float
from loguru import logger
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from typeguard import typechecked


@dataclass
class InferenceConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    block_length: int = 8
    generation_length: int = 25 * block_length
    temperature: float = 0.2
    remasking: Literal["low_confidence", "random"] = "random"
    mask_id: int = 126336
    min_avg_block_confidence: float = 0.85
    max_steps_per_block: int = 15
    tokens_to_unmask_per_step: int = 1

    def __post_init__(self):
        assert self.generation_length % self.block_length == 0, "generation_length must be divisible by block_length"
        self.num_blocks = self.generation_length // self.block_length


@typechecked
@torch.no_grad()
def add_gumbel_noise(
    logits: Float[torch.Tensor, "1 vocab_size"],
    temperature: float,
) -> Float[torch.Tensor, "1 vocab_size"]:
    """
    Adds gumbel noise to the logits.
    Note: Original code applied this to (1, block_length) but it should be (1, block_length, vocab_size) if logits are from model.
    Assuming logits here are (..., vocab_size)
    """
    if temperature <= 0:
        return logits

    original_dtype = logits.dtype
    logits_float64 = logits.to(torch.float64)

    noise = torch.rand_like(logits_float64, dtype=torch.float64)
    noise.clamp_(torch.finfo(torch.float64).eps, 1.0 - torch.finfo(torch.float64).eps)

    gumbel_noise_component = (-torch.log(-torch.log(noise))) * temperature

    return (logits_float64 + gumbel_noise_component).to(original_dtype)


@typechecked
@torch.no_grad()
def calculate_dynamic_steps(
    block_content: torch.Tensor,  # Shape (1, block_length)
    mask_id: int,
) -> int:
    """
    Calculates the number of diffusion steps for a block based on its masked token ratio.
    """
    num_masked = (block_content == mask_id).sum().item()
    block_len = block_content.shape[1]

    if block_len == 0:
        return 3  # Default minimum for an empty block, though block_length should be > 0

    masked_ratio = num_masked / block_len

    if masked_ratio > 0.75:
        return 10  # High uncertainty
    elif masked_ratio > 0.25:
        return 6  # Medium uncertainty
    else:
        return 3  # Low uncertainty or mostly filled


@torch.no_grad()
def remask(logits: torch.Tensor, x0: torch.Tensor, remasking: Literal["low_confidence", "random"]) -> torch.Tensor:
    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        return torch.squeeze(torch.gather(p, dim=-1, index=x0.unsqueeze(-1)), -1)
    elif remasking == "random":
        return torch.rand((x0.shape[0], x0.shape[1]), device=x0.device, dtype=torch.float32)
    else:
        raise ValueError(f"Invalid remasking mode: {remasking}")


@torch.no_grad()
def determine_transfers(
    confidence_in_block: Float[torch.Tensor, "1 block_length"],
    mask_index_in_block: torch.Tensor,
    num_to_unmask_this_step: int,
    block_start_in_canvas: int,
    canvas_length: int,
) -> torch.Tensor:
    transfer_index_canvas = torch.zeros((1, canvas_length), dtype=torch.bool, device=confidence_in_block.device)

    masked_confidences = confidence_in_block[mask_index_in_block]

    if masked_confidences.numel() == 0:
        return transfer_index_canvas

    k = min(num_to_unmask_this_step, masked_confidences.numel())

    if k <= 0:
        return transfer_index_canvas

    _, topk_indices_relative_to_masked = torch.topk(masked_confidences, k)

    original_masked_indices_in_block = torch.where(mask_index_in_block.squeeze(0))[0]

    selected_indices_in_block = original_masked_indices_in_block[topk_indices_relative_to_masked]

    selected_indices_canvas = selected_indices_in_block + block_start_in_canvas

    transfer_index_canvas[:, selected_indices_canvas] = True

    return transfer_index_canvas


@torch.no_grad()
def diffusion_step(
    model: torch.nn.Module,
    x: torch.Tensor,  # (1, canvas_length) tensor of `long`s -- token ids.
    block_start: int,
    block_end: int,
    conf: InferenceConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    block_slice = slice(block_start, block_end)
    mask_index_in_block_relative = x[:, block_slice] == conf.mask_id

    if not mask_index_in_block_relative.any().item():
        empty_conf_block = torch.full_like(x[:, block_slice], -float("inf"), dtype=torch.float32)
        return x, empty_conf_block

    logits_canvas = model(x).logits

    logits_with_noise_canvas = add_gumbel_noise(logits_canvas, conf.temperature)

    x0_canvas = torch.argmax(logits_with_noise_canvas, dim=-1)

    x0_p_canvas = remask(logits_canvas, x0_canvas, conf.remasking)

    x0_p_for_block = x0_p_canvas[:, block_slice]

    transfer_index_canvas = determine_transfers(
        x0_p_for_block, mask_index_in_block_relative, conf.tokens_to_unmask_per_step, block_start, x.shape[1]
    )

    updated_x_canvas = torch.where(transfer_index_canvas, x0_canvas, x)

    return updated_x_canvas, x0_p_for_block


@torch.no_grad()
def generate_response(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    conf: InferenceConfig,
) -> str:
    chat_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = torch.tensor(tokenizer(chat_input)["input_ids"]).to(conf.device).unsqueeze(0)
    logger.info(f"Input IDs: {input_ids} Shape: {input_ids.shape}")
    prompt_length = input_ids.shape[1]
    logger.info(f"Prompt length: {prompt_length} tokens")

    x = torch.full((1, prompt_length + conf.generation_length), conf.mask_id, dtype=torch.long).to(conf.device)
    x[:, :prompt_length] = input_ids.clone()
    canvas_length = x.shape[1]

    logger.info(f"Starting generation with {conf.num_blocks} blocks, dynamic steps per block based on confidence.")
    logger.info(f"Canvas shape: {x.shape}")

    for block_idx in range(conf.num_blocks):
        logger.info(f"Generating block {block_idx}")
        block_start = prompt_length + block_idx * conf.block_length
        block_end = min(prompt_length + (block_idx + 1) * conf.block_length, canvas_length)
        block_slice = slice(block_start, block_end)

        steps_this_block = 0
        while True:
            current_mask_in_block = x[:, block_slice] == conf.mask_id
            if not current_mask_in_block.any().item():
                logger.info(f"Block {block_idx}: No masks remaining. Completed.")
                break

            if steps_this_block >= conf.max_steps_per_block:
                logger.warning(
                    f"Block {block_idx}: Reached max steps ({conf.max_steps_per_block}). Moving to next block."
                )
                break

            x_after_step, x0_p_for_block = diffusion_step(model, x, block_start, block_end, conf)

            relevant_confidences = x0_p_for_block[current_mask_in_block]

            avg_confidence_for_step = 0.0
            if relevant_confidences.numel() > 0:
                avg_confidence_for_step = relevant_confidences.mean().item()
            else:
                logger.info(
                    f"Block {block_idx}, Step {steps_this_block}: No masked tokens were targeted in this step (should be caught earlier)."
                )

            logger.info(
                f"Block {block_idx}, Step {steps_this_block}: Avg confidence of predictions for initially masked tokens: {avg_confidence_for_step:.4f}"
            )

            x = x_after_step

            block_is_now_fully_unmasked = not (x[:, block_slice] == conf.mask_id).any().item()

            if block_is_now_fully_unmasked and avg_confidence_for_step >= conf.min_avg_block_confidence:
                logger.info(
                    f"Block {block_idx}: Met confidence threshold ({conf.min_avg_block_confidence:.2f}) and all masks filled. Completed."
                )
                break

            steps_this_block += 1

        if (x[:, block_slice] == conf.mask_id).any().item():
            logger.warning(f"Block {block_idx} finished processing but still contains masked tokens.")

    reponse_tokens = x[0, prompt_length:]
    return tokenizer.decode(reponse_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)


def main():
    conf = InferenceConfig()
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    logger.info("Loading model...")
    start_time = time.time()

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct",
        trust_remote_code=True,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
    )
    model.tie_weights()
    model.eval()

    logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
    logger.debug(f"Memory summary\n: {torch.cuda.memory_summary(device=conf.device, abbreviated=True)}")
    messages = [
        {"role": "user", "content": "Please write a short story about a cowboy riding a horse."},
    ]
    final_text = generate_response(model, tokenizer, messages, conf)

    print(final_text)


if __name__ == "__main__":
    main()
