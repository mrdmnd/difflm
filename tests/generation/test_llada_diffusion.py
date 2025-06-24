import time

import pytest
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizerBase

from difflm.generation.llada_diffusion_model import LLADAInferenceConfig, generate_response


@pytest.fixture(scope="session")
def shared_tokenizer() -> PreTrainedTokenizerBase:
    print("Loading tokenizer...")
    t0 = time.perf_counter_ns()
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    t1 = time.perf_counter_ns()
    print(f"Tokenizer loaded in {(t1 - t0) / 1e6:.2f} milliseconds")
    return tokenizer


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def shared_full_model() -> torch.nn.Module:
    print("Loading full model...")
    t1 = time.perf_counter_ns()
    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct",
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    t2 = time.perf_counter_ns()
    model.tie_weights()
    print(f"Model loaded in {(t2 - t1) / 1e6:.2f} milliseconds")
    return model


# def test_basic_generation(
#     shared_tokenizer: PreTrainedTokenizerBase,
#     shared_full_model: torch.nn.Module,
# ) -> None:
#     tokenizer, model = shared_tokenizer, shared_full_model
#     conf = LLADAInferenceConfig()
#     messages = [{"role": "user", "content": "Write a very short story about a martian who enjoys pancakes."}]
#     final_text = generate_response(messages, tokenizer, model, conf)
#     print(final_text)
#     assert len(final_text) > 0


def test_quantized_generation(
    shared_tokenizer: PreTrainedTokenizerBase,
    shared_quantized_model: torch.nn.Module,
) -> None:
    tokenizer, model = shared_tokenizer, shared_quantized_model
    conf = LLADAInferenceConfig()
    messages = [{"role": "user", "content": "How many one fourths are there in 7/2?"}]
    final_text = generate_response(messages, tokenizer, model, conf)
    print(final_text)
    assert len(final_text) > 0
