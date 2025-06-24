from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from loguru import logger
from torch import Tensor
from transformers import PreTrainedTokenizerBase

MASK_ID = 126336
EOS_TOKEN_ID = 126081  # <|endoftext|> token ID for LLaDA-8B-Instruct


@dataclass
class LLADAInferenceConfig:
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    generation_length: int = 64
    max_steps: int = 64
    sampling_temperature: float = 1.0
    # temp = 1.0 is default temperature for the model
    # temp < 1.0 is more conserative/peaked / less creative sampling - closer to argmax
    # temp > 1.0 is more creative/spread out / more random sampling


@torch.no_grad()
def adaptive_confidence(
    probs: Float[Tensor, "1 canvas_length vocab_size"],  # 0.0 to 1.0
    sampled_tokens: Int[Tensor, "1 canvas_length"],
    progress: float,  # 0.0 to 1.0
) -> Float[Tensor, "1 canvas_length"]:
    """
    Adaptively weight different confidence metrics based on decoding progress.
    Early in decoding: prefer entropy-based (more exploratory) confidence metrics.
    Late in decoding: prefer margin-based (more decisive) confidence metrics.

    There are three signals:
    - token probability (how likely is the sampled token?) (ranges from 0 to 1)
    - logit entropy (how much entropy is there in the probability distribution across the vocab?)
    - top-two margin (how much more likely is the top token than the second-top token?) (ranges from 0 to 1)

    We weight these signals based on progress.

    We return a confidence score for each token in the canvas.

    """
    _, _, vocab_size = probs.shape

    # Signal 1: token probability
    token_probs = torch.gather(probs, dim=-1, index=sampled_tokens.unsqueeze(-1)).squeeze(-1)
    logger.info(f"token_probs:\n{token_probs}")

    # Signal 2: entropy
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    max_entropy = torch.log(torch.tensor(vocab_size, device=probs.device))
    entropy_confidence = (max_entropy - entropy) / max_entropy
    logger.info(f"entropy_confidence:\n{entropy_confidence}")

    # Signal 3: top-two margin
    top_two_probs, _ = torch.topk(probs, k=2, dim=-1)
    margin = top_two_probs[:, :, 0] - top_two_probs[:, :, 1]
    logger.info(f"top_two_margin:\n{margin}")

    entropy_weight = 2.0 * (1 - progress)
    margin_weight = 2.0 * progress
    prob_weight = 1.0
    total_weight = entropy_weight + margin_weight + prob_weight
    return (entropy_weight * entropy_confidence + margin_weight * margin + prob_weight * token_probs) / total_weight


@torch.no_grad()
def determine_transfers(
    confidence: Float[Tensor, "1 canvas_length"],
    eligible_indices: Bool[Tensor, "1 canvas_length"],
) -> Bool[Tensor, "1 canvas_length"]:
    """
    Confidence is a tensor of shape (1, canvas_length), where each element is the confidence of the corresponding token
    It includes confidence in the prompt tokens, but we don't actually ever want to transfer those.

    We also include a *mask* index that corresponds to the tokens we might want to transfer.
    """
    # first, figure out how many tokens we should transfer as a function of progress
    # when progress == 1.0 that's the last step so we need to transfer all the tokens.
    # for now just set this to 1.
    transfer_token_count = 4

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
    progress: float,  # 0.0 to 1.0
    model: torch.nn.Module,
    conf: LLADAInferenceConfig,
) -> Int[Tensor, "1 canvas_length"]:
    """
    This function performs a single diffusion step on the canvas, including model inference and remasking.
    We return the updated canvas, which should have strictly fewer masked tokens than the previous canvas.
    """

    # Run the model and predict the logits for every position in the canvas.
    raw_logits: Float[Tensor, "1 canvas_length vocab_size"] = model(canvas).logits.to(torch.float64)
    scaled_logits = raw_logits / conf.sampling_temperature

    # THIS IS A NOOP for now - eventually we'll set things to log(prob) = -inf for illegal tokens
    # constrained_logits = apply_structural_constraints(scaled_logits)
    constrained_logits = scaled_logits

    probs = torch.nn.functional.softmax(constrained_logits, dim=-1)

    # Sample tokens from the renormalized, constrained, noised raw logits
    sampled_tokens = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1).view(1, -1)
    # Determine how confident we are in each of the sampled tokens.
    token_confidence = adaptive_confidence(probs, sampled_tokens, progress)

    # Determine which tokens to transfer from the sampled tokens back to the canvas, and which to leave masked.
    # We should transfer the tokens with the highest confidence and remask the rest.
    transfer_indices = determine_transfers(token_confidence, eligible_indices)
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

    step = 0
    mask_index: Bool[Tensor, "1 canvas_length"] = canvas == MASK_ID
    while step < conf.max_steps and mask_index.any():
        step += 1
        progress = step / conf.max_steps
        canvas = diffusion_step(canvas, mask_index, progress, model, conf)
        mask_index = canvas == MASK_ID
        logger.info(f"Generation step {step} complete:\n{tokenizer.decode(canvas[0].tolist())}")

    # Decode the canvas.
    return tokenizer.decode(canvas[0].tolist())
