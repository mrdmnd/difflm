from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from loguru import logger
from torch import Tensor
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

MASK_ID = 126336
EOS_TOKEN_ID = 126081  # <|endoftext|> token ID for LLaDA-8B-Instruct


@dataclass
class LLADAInferenceConfig:
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    generation_length: int = 64
    max_steps: int = 64
    gumbel_tau: float = 1.0


@torch.no_grad()
def constrain_logits(
    logits: Float[Tensor, "1 canvas_length vocab_size"],
    eligible_indices: Bool[Tensor, "1 canvas_length"],
) -> Float[Tensor, "1 canvas_length vocab_size"]:
    """
    Mask out illegal logits and apply structural constraints.

    Prevent <|endoftext|> from being generated at masked positions.
    """
    logits = logits.clone()
    # Set the logit for EOS_TOKEN_ID to -inf at eligible (masked) positions
    # This prevents the model from generating <|endoftext|> during generation
    logits[0, eligible_indices[0], EOS_TOKEN_ID] = -float("inf")
    return logits


@torch.no_grad()
def sample_categorical(
    logits: Float[Tensor, "1 canvas_length vocab_size"],
    tau: float,
) -> Int[Tensor, "1 canvas_length"]:
    """
    Sample from a categorical distribution with temperature scaling.
    """
    # Apply temperature scaling
    scaled_logits = logits / tau
    probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
    # probs: (1, canvas_length, vocab_size)
    # We want to sample for each position in canvas_length
    sampled = torch.multinomial(
        probs.squeeze(0),  # shape: (canvas_length, vocab_size)
        num_samples=1,
    ).squeeze(-1)  # shape: (canvas_length,)
    return sampled.unsqueeze(0)  # shape: (1, canvas_length)


@torch.no_grad()
def compute_confidence(
    logits: Float[Tensor, "1 canvas_length vocab_size"],
    sampled_tokens: Int[Tensor, "1 canvas_length"],
) -> Float[Tensor, "1 canvas_length"]:
    """
    Compute the confidence in each sampled token.

    Values are in [0, 1].
    """
    # p = torch.nn.functional.softmax(logits, dim=-1)  # (1, canvas_length, vocab_size)
    # return torch.squeeze(torch.gather(p, dim=-1, index=sampled_tokens.unsqueeze(-1)), -1)  # (1, canvas_length)

    # do randomized remasking
    return torch.rand((sampled_tokens.shape[0], sampled_tokens.shape[1]), device=sampled_tokens.device, dtype=torch.float16)


@torch.no_grad()
def determine_transfers(
    confidence: Float[Tensor, "1 canvas_length"],
    eligible_indices: Bool[Tensor, "1 canvas_length"],
    progress: float,
) -> Bool[Tensor, "1 canvas_length"]:
    """
    Confidence is a tensor of shape (1, canvas_length), where each element is the confidence of the corresponding token
    It includes confidence in the prompt tokens, but we don't actually ever want to transfer those.

    We also include a *mask* index that corresponds to the tokens we might want to transfer.
    """
    # first, figure out how many tokens we should transfer as a function of progress
    # when progress == 1.0 that's the last step so we need to transfer all the tokens.
    # for now just set this to 1.
    transfer_token_count = 1

    # NOTE: we are only considering the top-k highest confidence *generation* tokens, not including the prompt.
    # We could instead randomize!
    masked_confidence = torch.where(eligible_indices, confidence, torch.full_like(confidence, -np.inf))
    logger.info(f"masked_confidence:\n{masked_confidence}")

    _, topk_indices = torch.topk(masked_confidence, transfer_token_count)

    # Set the transfer indices to True where we want to keep a sampled token.
    transfer_indices = torch.zeros((1, confidence.shape[1]), device=confidence.device, dtype=torch.bool)
    transfer_indices[:, topk_indices] = True

    return transfer_indices


@torch.no_grad()
def diffusion_step(
    canvas: Int[Tensor, "1 canvas_length"],
    eligible_indices: Bool[Tensor, "1 canvas_length"],
    progress: float,
    model: torch.nn.Module,
    conf: LLADAInferenceConfig,
) -> Int[Tensor, "1 canvas_length"]:
    """
    This function performs a single diffusion step on the canvas, including model inference and remasking.
    We return the updated canvas, which should have strictly fewer masked tokens than the previous canvas.
    """

    # Run the model over the canvas.
    raw_logits = model(canvas).logits  # (1, canvas_length, vocab_size)
    # Mask out illegal logits and apply structural constraints.
    constrained_logits = constrain_logits(raw_logits, eligible_indices)  # (1, canvas_length, vocab_size)
    # Sample tokens from the model's (constrained) output.
    sampled_tokens = sample_categorical(constrained_logits, conf.gumbel_tau)  # (1, canvas_length)
    # Determine how confident we are in each of the sampled tokens.
    confidence = compute_confidence(constrained_logits, sampled_tokens)
    # Determine which tokens to transfer from the sampled tokens back to the canvas, and which to leave masked.
    # We should transfer the tokens with the highest confidence and remask the rest.
    transfer_indices = determine_transfers(confidence, eligible_indices, progress)
    # For eligible positions: transfer if selected, otherwise keep original value.
    return torch.where(transfer_indices, sampled_tokens, canvas)


@torch.no_grad()
def generate_response(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    model: torch.nn.Module,
    conf: LLADAInferenceConfig,
) -> str:
    chat_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = torch.tensor(tokenizer(chat_input)["input_ids"]).to(conf.device).unsqueeze(0)
    prompt_length = input_ids.shape[1]
    # Initialize the canvas with mask tokens.
    canvas = torch.full((1, prompt_length + conf.generation_length), MASK_ID, dtype=torch.long).to(conf.device)
    canvas[:, 0:prompt_length] = input_ids.clone()

    for step in tqdm(range(1, conf.max_steps + 1)):
        progress = 1.0 * step / conf.max_steps
        mask_index: Bool[Tensor, "1 canvas_length"] = canvas == MASK_ID
        if not mask_index.any():
            logger.info("No more masked tokens to unmask.")
            break
        canvas = diffusion_step(canvas, mask_index, progress, model, conf)
        logger.info(f"Generation step {step} complete:\n{tokenizer.decode(canvas[0].tolist())}")

    # Decode the canvas.
    return tokenizer.decode(canvas[0].tolist())
