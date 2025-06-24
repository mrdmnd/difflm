import time

import torch
import torch.profiler
from loguru import logger
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizerBase

from difflm.generation.llada_diffusion_model import LLADAInferenceConfig, generate_response


def shared_tokenizer() -> PreTrainedTokenizerBase:
    print("Loading tokenizer...")
    t0 = time.perf_counter_ns()
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    t1 = time.perf_counter_ns()
    print(f"Tokenizer loaded in {(t1 - t0) / 1e6:.2f} milliseconds")
    return tokenizer


def shared_quantized_model() -> torch.nn.Module:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    print("Loading quantized model...")
    t1 = time.perf_counter_ns()
    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct",
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    t2 = time.perf_counter_ns()
    print(f"Model loaded in {(t2 - t1) / 1e6:.2f} milliseconds")
    return model


def shared_full_model() -> torch.nn.Module:
    logger.info("Loading full model...")
    t1 = time.perf_counter_ns()
    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct",
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    t2 = time.perf_counter_ns()
    logger.info(f"Model loaded in {(t2 - t1) / 1e6:.2f} milliseconds")

    # logger.info("Compiling model...")
    # compiled_model = torch.compile(model)
    # t3 = time.perf_counter_ns()
    # logger.info(f"Model compiled in {(t3 - t2) / 1e6:.2f} milliseconds")

    return model


def bench_generation(
    tokenizer: PreTrainedTokenizerBase,
    model: torch.nn.Module,
) -> None:
    conf = LLADAInferenceConfig()
    conf.generation_length = 64
    conf.steps = 30
    messages = [
        {
            "role": "user",
            "content": "If there are three apples in a basket and two oranges, how many fruits are there in total?",
        }
    ]
    output = generate_response(messages, tokenizer, model, conf)
    logger.success(output)

    # print("Warming up...")
    # for _ in range(3):
    #     generate_response(messages, tokenizer, model, conf)

    # print("Running pytorch profiler...")
    # with torch.profiler.profile(
    #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    #     record_shapes=False,
    #     profile_memory=False,
    #     with_stack=False,
    # ) as torch_prof:
    #     generate_response(messages, tokenizer, model, conf)

    # print(torch_prof.key_averages().table(sort_by="cpu_time_total"))


if __name__ == "__main__":
    tokenizer = shared_tokenizer()
    model = shared_full_model()
    bench_generation(tokenizer, model)
