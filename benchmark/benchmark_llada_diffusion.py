import time

import torch
import torch.profiler
from loguru import logger
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from difflm.generation.llada_diffusion_model import LLADAInferenceConfig, generate_response


def load_model(compiled: bool = True) -> tuple[PreTrainedTokenizerBase, torch.nn.Module]:
    logger.info("Loading tokenizer...")
    t0 = time.perf_counter_ns()
    tokenizer = AutoTokenizer.from_pretrained("./quantized_models/llada-8b-instruct-8bit-gptq", trust_remote_code=True)
    t1 = time.perf_counter_ns()
    logger.info(f"Tokenizer loaded in {(t1 - t0) / 1e6:.2f} milliseconds")

    logger.info("Loading model...")
    t2 = time.perf_counter_ns()
    model = AutoModel.from_pretrained(
        "./quantized_models/llada-8b-instruct-8bit-gptq",
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    t3 = time.perf_counter_ns()
    logger.info(f"Model loaded in {(t3 - t2) / 1e6:.2f} milliseconds")

    if compiled:
        if model.device.type != "cuda":
            raise ValueError("Model must be on CUDA device to compile")
        logger.info("Compiling model...")
        t3 = time.perf_counter_ns()
        model = torch.compile(model)
        t4 = time.perf_counter_ns()
        logger.info(f"Model compiled in {(t4 - t3) / 1e6:.2f} milliseconds")

    return tokenizer, model


def bench_generation(
    tokenizer: PreTrainedTokenizerBase,
    model: torch.nn.Module,
) -> None:
    conf = LLADAInferenceConfig()
    conf.generation_length = 64
    conf.steps = 15
    messages = [
        {
            "role": "user",
            "content": "If there are three apples in a basket and two oranges, how many fruits are there in total?",
        }
    ]

    print("Warming up...")
    for _ in range(3):
        generate_response(messages, tokenizer, model, conf)

    print("Running pytorch profiler...")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as torch_prof:
        output = generate_response(messages, tokenizer, model, conf)

    logger.success(output)

    print(torch_prof.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_time_total"))


if __name__ == "__main__":
    tokenizer, model = load_model(compiled=True)
    bench_generation(tokenizer, model)
