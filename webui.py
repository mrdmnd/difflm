from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

from difflm.generation.llada_diffusion_model import (
    MASK_ID,
    compute_schedule,
    diffusion_step,
)

# --- Globals ---
MODEL = None
TOKENIZER = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GENERATION_STATE = None


class GenerationState:
    """
    Server state is maintained here.

    If you just want to reset the generation, you can call reset() and we wipe the canvas and set the current step to 0.
    """

    def __init__(self, prompt: str, generation_length: int, max_steps: int, sampling_temperature: float):
        messages = [{"role": "user", "content": prompt}]
        chat_input = TOKENIZER.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        self.prompt = prompt
        self.input_ids = torch.tensor(TOKENIZER(chat_input)["input_ids"], device=DEVICE).unsqueeze(0)
        self.max_steps = max_steps
        self.sampling_temperature = sampling_temperature
        self.generation_length = generation_length
        self.prompt_length = self.input_ids.shape[1]
        self.canvas_length = self.prompt_length + self.generation_length
        self.schedule = compute_schedule(self.generation_length, self.max_steps)

        # Initialize / reset the canvas and the logits
        self.current_step = 0
        self.canvas = torch.full((1, self.canvas_length), MASK_ID, dtype=torch.long, device=DEVICE)

        # What's the token id at each position?
        self.canvas[:, : self.prompt_length] = self.input_ids

        # What's the (temperature weighted) probability of each token at each position in the canvas?
        self.probs = torch.zeros((1, self.canvas_length, MODEL.config.vocab_size), device=DEVICE)

        # What's the confidence we have in each tokens? Special sauce.
        self.masked_token_confidence = torch.zeros((1, self.canvas_length), device=DEVICE)

    def step(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        If we haven't reached the max steps, call the diffusion step and return the current canvas and logits.

        If we have, return the current canvas and logits.

        The entire right panel of the UI is rendered based on the results *after* taking a diffusion step.
        """
        if self.current_step >= self.max_steps:
            logger.warning("Reached max steps, returning current canvas and logits.")
            return self.canvas, self.probs, self.masked_token_confidence

        transfer_count = self.schedule[self.current_step]

        with torch.no_grad():
            progress = (self.current_step + 1) / self.max_steps
            raw_logits = MODEL(self.canvas).logits
            new_canvas, probs, masked_token_confidence = diffusion_step(
                raw_logits,
                self.canvas,
                progress,
                transfer_count,
                self.sampling_temperature,
            )
        self.canvas = new_canvas
        self.probs = probs
        self.masked_token_confidence = masked_token_confidence
        self.current_step += 1
        return self.canvas, self.probs, self.masked_token_confidence

    def reset(self) -> None:
        self.current_step = 0
        self.canvas = torch.full((1, self.canvas_length), MASK_ID, dtype=torch.long, device=DEVICE)
        self.canvas[:, : self.prompt_length] = self.input_ids
        self.probs = torch.zeros((1, self.canvas_length, MODEL.config.vocab_size), device=DEVICE)
        self.masked_token_confidence = torch.zeros((1, self.canvas_length), device=DEVICE)

    def get_token_probabilities(self, position: int, top_k: int = 5) -> tuple[list[str], list[float]] | None:
        if self.probs is None or position < 0 or position >= self.probs.shape[1]:
            logger.error(f"Invalid position: {position}")
            return None

        top_k_probs, top_k_indices = torch.topk(self.probs[0, position, :], top_k)
        top_k_tokens = [TOKENIZER.decode([idx.item()]) for idx in top_k_indices]
        return top_k_tokens, top_k_probs.tolist()

    def to_json(self) -> dict[str, Any]:
        canvas_as_list = self.canvas[0].tolist()
        decoded_tokens = [TOKENIZER.decode([token_id]) for token_id in canvas_as_list]
        return {
            "prompt": self.prompt,
            "generation_length": self.generation_length,
            "max_steps": self.max_steps,
            "sampling_temperature": self.sampling_temperature,
            "schedule": self.schedule,
            "current_step": self.current_step,
            "prompt_length": self.prompt_length,
            "canvas": canvas_as_list,
            "decoded_tokens": decoded_tokens,
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model and tokenizer at startup.
    """
    model_path = "./quantized_models/llada-8b-instruct-8bit-gptq"
    global MODEL, TOKENIZER  # noqa: PLW0603
    logger.info("Loading tokenizer...")
    TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logger.info("Loading model...")
    MODEL = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    logger.info("Compiling model...")
    MODEL = torch.compile(MODEL)
    yield

    # Clean up and release resources
    torch.cuda.empty_cache()


# --- FastAPI App ---
app = FastAPI(lifespan=lifespan)


class CreateStateRequest(BaseModel):
    prompt: str
    generation_length: int
    steps: int
    sampling_temperature: float


@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    # This will serve the main HTML file.
    # We will create this file later.
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)


# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/api/create_state")
async def create_state(request: CreateStateRequest):
    """
    Initializes a new generation state. This is the first call that a user should make.
    """
    global GENERATION_STATE  # noqa: PLW0603
    GENERATION_STATE = GenerationState(
        prompt=request.prompt,
        generation_length=request.generation_length,
        max_steps=request.steps,
        sampling_temperature=request.sampling_temperature,
    )
    return GENERATION_STATE.to_json()


@app.get("/api/state")
async def get_state():
    """
    Returns the current generation state.
    """
    if not GENERATION_STATE:
        return {"error": "State not initialized. Call /api/create_state first."}
    return GENERATION_STATE.to_json()


@app.post("/api/step")
async def step():
    """
    Advances the generation by one step.
    """
    if not GENERATION_STATE:
        return {"error": "State not initialized. Call /api/create_state first."}

    GENERATION_STATE.step()
    return GENERATION_STATE.to_json()


@app.post("/api/reset")
async def reset():
    """
    Resets the generation to its initial state.
    """
    if not GENERATION_STATE:
        return {"error": "State not initialized. Call /api/create_state first."}

    GENERATION_STATE.reset()
    return GENERATION_STATE.to_json()


@app.get("/api/token_probabilities/{position}")
async def get_token_probabilities(position: int, top_k: int = 5):
    """
    Returns the top k token probabilities for a given position.
    """
    if not GENERATION_STATE:
        return {"error": "State not initialized. Call /api/create_state first."}

    return GENERATION_STATE.get_token_probabilities(position, top_k)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
