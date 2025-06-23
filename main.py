import time
from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Bool, Float32, Int64
from loguru import logger
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


@dataclass
class InferenceConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mask_id: int = 126336
    generation_length: int = 50
    max_steps: int = 40
    gumbel_tau: float = 0.2


def precompute_num_transfer_tokens(
    initial_mask_index: Bool[torch.Tensor, "1 canvas_length"],
    conf: InferenceConfig,
) -> Int64[torch.Tensor, "1 max_steps"]:
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into `max_steps` intervals.
    Because LLaDa employs a linear noise schedule, the number of tokens to unmask at each step should be ~consistent
    (up to remainder). This function precompute the number of tokens to unmask at each step, for each step.

    For example, if we have 23 masked tokens and five steps, we will get [[5, 5, 5, 4, 4]] as the output.
    This is because 23 / 5 = 4 remainder 3, so we have all steps getting 4 tokens, and the first three steps get one
    extra token.
    """
    mask_num = initial_mask_index.sum(dim=1, keepdim=True)
    base = mask_num // conf.max_steps
    remainder = mask_num % conf.max_steps
    num_transfer_tokens = (
        torch.zeros(mask_num.shape[0], conf.max_steps, device=mask_num.device, dtype=torch.int64) + base
    )
    for i in range(mask_num.shape[0]):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def sample_categorical(
    logits: Float32[torch.Tensor, "1 canvas_length vocab_size"],
    gumbel_tau: float,
) -> Int64[torch.Tensor, "1 canvas_length"]:
    """
    Sample from a categorical distribution with temperature scaling.

    Alternative implementation using gumbel_softmax (maybe more stable):
    # return torch.argmax(F.gumbel_softmax(logits, tau=temperature, hard=False), dim=-1)
    """
    if gumbel_tau <= 0:
        return torch.argmax(logits, dim=-1)

    # Scale logits by temperature (in log space)
    scaled_logits = logits / gumbel_tau
    # Sample from categorical distribution using Gumbel-Max trick
    # This is equivalent to: argmax(scaled_logits + Gumbel(0, 1))
    uniform_noise = torch.rand_like(scaled_logits)
    # Clamp to avoid log(0) or log(1) issues - use more conservative bounds
    uniform_noise = torch.clamp(uniform_noise, 1e-10, 1.0 - 1e-10)
    gumbel_noise = -torch.log(-torch.log(uniform_noise))

    # Add Gumbel noise and take argmax
    return torch.argmax(scaled_logits + gumbel_noise, dim=-1)


@torch.no_grad()
def determine_transfers(
    confidence: Float32[torch.Tensor, "1 canvas_length"],
    eligible_indices: Bool[torch.Tensor, "1 canvas_length"],
    transfer_token_count: int,
) -> Bool[torch.Tensor, "1 canvas_length"]:
    """
    Confidence is a tensor of shape (1, canvas_length), where each element is the confidence of the corresponding token
    It includes confidence in the prompt tokens, but we don't actually ever want to transfer those.

    We also include a *mask* index that corresponds to the tokens we might want to transfer.
    """

    # NOTE: we are only considering the top-k highest confidence *generation* tokens, not including the prompt.
    masked_confidence = torch.where(eligible_indices, confidence, torch.full_like(confidence, -np.inf))
    transfer_indices = torch.zeros((1, confidence.shape[1]), device=confidence.device, dtype=torch.bool)
    _, topk_indices = torch.topk(masked_confidence, transfer_token_count)
    transfer_indices[:, topk_indices] = True

    return transfer_indices


@torch.no_grad()
def diffusion_step(
    canvas: Int64[torch.Tensor, "1 canvas_length"],
    eligible_indices: Bool[torch.Tensor, "1 canvas_length"],
    transfer_token_count: int,
    model: torch.nn.Module,
    conf: InferenceConfig,
) -> Int64[torch.Tensor, "1 canvas_length"]:
    """
    This function performs a single diffusion step on the canvas, including model inference and remasking.
    We return the updated canvas, which should have strictly fewer masked tokens than the previous canvas.
    """

    # Run the model over the canvas.
    raw_logits = model(canvas).logits  # (1, canvas_length, vocab_size)
    # Sample tokens from the model's output.
    sampled_tokens = torch.argmax(
        F.gumbel_softmax(raw_logits, tau=conf.gumbel_tau, hard=False), dim=-1
    )  # (1, canvas_length)
    # Determine how confident we are in each of the sampled tokens.
    p = F.softmax(raw_logits, dim=-1)  # (1, canvas_length, vocab_size)
    confidence = torch.squeeze(torch.gather(p, dim=-1, index=sampled_tokens.unsqueeze(-1)), -1)  # (1, canvas_length)

    logger.info(f"Confidence:\n{confidence}")
    logger.info(f"Confidence shape: {confidence.shape}")

    # Determine which tokens to transfer from the sampled tokens back to the canvas, and which to leave masked.
    # We should transfer the tokens with the highest confidence and remask the rest.
    transfer_indices = determine_transfers(confidence, eligible_indices, transfer_token_count)

    logger.info(f"Transfer indices:\n{transfer_indices}")
    logger.info(f"Transfer indices shape: {transfer_indices.shape}")

    # Update the canvas with the sampled tokens and leave the rest as they were already generated.
    # TODO(mrdmnd) - we should probably actually remask, rather than just leaving them as they were, right?
    updated_canvas = torch.where(transfer_indices, sampled_tokens, canvas)

    return updated_canvas


@torch.no_grad()
def generate_response(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    model: torch.nn.Module,
    conf: InferenceConfig,
) -> str:
    chat_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = torch.tensor(tokenizer(chat_input)["input_ids"]).to(conf.device).unsqueeze(0)
    logger.info(f"Input token IDs:\n{input_ids}")
    logger.info(f"Input shape: {input_ids.shape}")
    prompt_length = input_ids.shape[1]

    logger.info(f"Prompt length: {prompt_length} tokens")
    logger.info(f"Desired generation length: {conf.generation_length} tokens")

    # Initialize the canvas with mask tokens.
    canvas = torch.full((1, prompt_length + conf.generation_length), conf.mask_id, dtype=torch.long).to(conf.device)
    canvas[:, 0:prompt_length] = input_ids.clone()

    mask_index: Bool[torch.Tensor, "1 canvas_length"] = canvas == conf.mask_id
    # Precompute the number of tokens to unmask at each step.
    num_transfer_tokens = precompute_num_transfer_tokens(mask_index, conf)
    logger.info(f"Number of tokens to unmask at each step: {num_transfer_tokens}")

    logger.info(f"Starting generation with {conf.max_steps} maximum steps")
    for step in tqdm(range(conf.max_steps)):
        transfer_token_count = num_transfer_tokens[0, step].item()
        logger.info(f"Transfer token count: {transfer_token_count}")
        if not mask_index.any():
            logger.info("No more masked tokens to unmask.")
            break
        canvas = diffusion_step(canvas, mask_index, transfer_token_count, model, conf)
        logger.info(f"Generation step {step} complete: {tokenizer.decode(canvas[0].tolist())}")
        mask_index: Bool[torch.Tensor, "1 canvas_length"] = canvas == conf.mask_id

    logger.info(f"Final canvas:\n{canvas}")
    logger.info(f"Final canvas shape: {canvas.shape}")

    # Decode the canvas.
    decoded_canvas = tokenizer.decode(canvas[0].tolist())
    return decoded_canvas


def main():
    conf = InferenceConfig()
    logger.info("Loading tokenizer...")
    t0 = time.perf_counter_ns()
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    t1 = time.perf_counter_ns()
    logger.info(f"Tokenizer loaded in {(t1 - t0) / 1e9:.2f} seconds")

    logger.info("Creating quantization config...")
    t2 = time.perf_counter_ns()
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    t3 = time.perf_counter_ns()
    logger.info(f"Quantization config loaded in {(t3 - t2) / 1e9:.2f} seconds")

    logger.info("Loading model...")
    t4 = time.perf_counter_ns()
    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct",
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    t5 = time.perf_counter_ns()
    model.tie_weights()
    logger.info(f"Model loaded in {(t5 - t4) / 1e9:.2f} seconds")

    logger.info(f"Total load time: {(t5 - t0) / 1e9:.2f} seconds")
    messages = [{"role": "user", "content": "Write a very short story about a martian."}]
    final_text = generate_response(tokenizer, messages, model, conf)

    print(final_text)


if __name__ == "__main__":
    main()
