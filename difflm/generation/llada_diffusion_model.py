import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.compiler
from jaxtyping import Float, Int
from loguru import logger
from torch import Tensor
from transformers import PreTrainedTokenizerBase

MASK_ID = 126336
EOS_TOKEN_ID = 126081  # <|endoftext|> token ID for LLaDA-8B-Instruct


@dataclass
class LLADAInferenceConfig:
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    generation_length: int = 97
    steps: int = 31
    sampling_temperature: float = 1.0
    # temp = 1.0 is default temperature for the model
    # temp < 1.0 is more conserative/peaked / less creative sampling - closer to argmax
    # temp > 1.0 is more creative/spread out / more random sampling


def compute_schedule(generation_length: int, steps: int) -> list[int]:
    """
    Compute a monotonically decreasing transfer schedule for the diffusion process,
    ensuring that at least one token is transferred in each step.
    """
    if generation_length < steps:
        raise ValueError("generation_length must be at least as large as steps")

    # Generate weights for each step using a cosine decay
    weights = np.array([math.cos((step / steps) * math.pi / 2) for step in range(steps)])
    weights /= weights.sum()

    # Calculate the ideal (float) number of tokens for each step
    ideal_tokens_float = weights * generation_length

    # Use the largest remainder method to convert to integers while preserving the sum
    schedule_int = np.floor(ideal_tokens_float).astype(int)
    remainders = ideal_tokens_float - schedule_int

    deficit = generation_length - schedule_int.sum()
    if deficit > 0:
        indices_to_increment = np.argsort(remainders)[-deficit:]
        schedule_int[indices_to_increment] += 1

    # Ensure the schedule is monotonic and has no zeros
    for i in range(len(schedule_int) - 1):
        if schedule_int[i] < schedule_int[i + 1]:
            diff = schedule_int[i + 1] - schedule_int[i]
            schedule_int[i] += diff
            schedule_int[i + 1] -= diff

    while schedule_int.sum() > generation_length:
        schedule_int[-1] -= 1

    while schedule_int.sum() < generation_length:
        schedule_int[0] += 1

    # Final check for zeros, although unlikely with generation_length >= steps
    if np.any(schedule_int <= 0):
        # This part should ideally not be reached
        schedule_int[schedule_int <= 0] = 1
        # Redistribute to maintain the sum
        excess = schedule_int.sum() - generation_length
        while excess > 0:
            schedule_int[np.argmax(schedule_int)] -= 1
            excess -= 1

    return schedule_int.tolist()


@torch.no_grad()
def diffusion_step(
    raw_logits: Float[Tensor, "1 canvas_length vocab_size"],
    canvas: Int[Tensor, "1 canvas_length"],
    progress: float,
    transfer_count: int,
    temperature: float = 1.0,
) -> Int[Tensor, "1 canvas_length"]:
    """
    This function performs a single diffusion step on the canvas, including model inference and remasking.
    We return the updated canvas, which should have strictly fewer masked tokens than the previous canvas.
    """
    batch_size, canvas_length, vocab_size = raw_logits.shape
    max_entropy_inv = 1.0 / np.log(vocab_size)

    # Get the probabilities for each vocabulary token, scaled by temperature, then sample from them.
    probs = torch.nn.functional.softmax(raw_logits / temperature, dim=-1)
    sampled_tokens = torch.multinomial(probs.flatten(0, 1), num_samples=1).view_as(canvas)

    # Compute all selection signals in parallel.
    sampled_indices = sampled_tokens.unsqueeze(-1)
    token_probs = torch.gather(probs, dim=-1, index=sampled_indices).squeeze(-1)
    neg_entropy = torch.special.entr(probs).sum(dim=-1) * max_entropy_inv
    top2 = torch.topk(probs, k=2, dim=-1).values
    margin = top2[:, :, 0] - top2[:, :, 1]

    # Dynamic selection weights based on progress through the diffusion process.
    w1, w2, w3 = (2.0 * (1 - progress), 2.0 * progress, 1.0)
    w_total_inv = 1.0 / (w1 + w2 + w3)
    confidence = (w1 * (1 + neg_entropy) + w2 * margin + w3 * token_probs) * w_total_inv

    current_mask = canvas == MASK_ID
    masked_confidence = confidence.masked_fill(~current_mask, -float("inf"))

    transfer_indices = torch.topk(masked_confidence, transfer_count, dim=1).indices
    transfer_mask = torch.zeros_like(canvas, dtype=torch.bool)
    transfer_mask.scatter_(1, transfer_indices, True)

    return torch.where(transfer_mask, sampled_tokens, canvas)


@torch.no_grad()
def generate_response(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    model: torch.nn.Module,
    conf: LLADAInferenceConfig,
) -> str:
    if conf.device == "cuda" and hasattr(torch.cuda, "graphs") and torch.cuda.is_available():
        torch.cuda.synchronize()

    chat_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = torch.tensor(tokenizer(chat_input)["input_ids"], device=conf.device).unsqueeze(0)
    prompt_length = input_ids.shape[1]
    # Initialize the canvas with mask tokens.
    canvas = torch.full((1, prompt_length + conf.generation_length), MASK_ID, dtype=torch.long, device=conf.device)
    canvas[:, :prompt_length] = input_ids

    transfer_schedule = compute_schedule(conf.generation_length, conf.steps)
    logger.info(f"Transfer schedule: {transfer_schedule}")

    for step, transfer_count in enumerate(transfer_schedule):
        progress = (step + 1) / conf.steps
        logits = model(canvas).logits
        canvas = diffusion_step(logits, canvas, progress, transfer_count, conf.sampling_temperature)
        print(tokenizer.decode(canvas[0].tolist()))

    # Decode the canvas.
    return tokenizer.decode(canvas[0].tolist())
