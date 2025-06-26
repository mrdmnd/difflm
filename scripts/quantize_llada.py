"""
This script uses Modal to quantize the LLaDA family of models.

First, install modal and log into the CLI.
uv add modal
uv run modal login

Then, add an environment and a volume to the project.
uv run modal volume create quantized-model-output

Then, run the quantization script:
uv run modal run scripts/quantize_llada.py

Finally you need to grab the model from the volume when it's done.
uv run modal volume get quantized-model-output llada-8b-instruct-Nbit-gptq .
"""

import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04", add_python="3.13")
    .apt_install("git", "curl")
    .pip_install(
        "torch>=2.7.0",
        "torchvision",
        "torchaudio",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "numpy",
        "accelerate",
        "optimum",
        "loguru",
        "transformers",
    )
    .pip_install("triton")
    .pip_install("gptqmodel")
)

# These need to be created before running the app:
# uv run modal volume create quantized-model-output
output_volume = modal.Volume.from_name("quantized-model-output")
volume_config = {"/quantized-model-output": output_volume}

app = modal.App("quantize-llada", image=image, volumes=volume_config)

TRAIN_GPU_COUNT = 4
TRAIN_GPU = f"B200:{TRAIN_GPU_COUNT}"
TRAIN_CPU_COUNT = 4
MINUTES = 40


@app.function(gpu=TRAIN_GPU, cpu=TRAIN_CPU_COUNT, timeout=MINUTES * 60)
def quantize_model() -> None:
    import types  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from loguru import logger  # noqa: PLC0415
    from optimum.gptq import GPTQQuantizer  # noqa: PLC0415
    from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

    output_volume.reload()

    # Check if CUDA is available, show device count
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"PyTorch version: {torch.__version__}")

    # Check if GPU is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)

    # We need to do some shenanigans.
    # First, load the model on the CPU so we can patch the forward pass.
    logger.info("Loading model on CPU...")
    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct",
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    logger.info("Patching model forward pass...")

    def patched_forward(self, x, *args, attention_bias=None, layer_past=None, use_cache=False, **kwargs):  # noqa
        """
        Patched forward that handles both positional and keyword arguments for attention_bias
        """
        # If attention_bias was passed as positional argument, use it
        if len(args) > 0 and attention_bias is None:
            attention_bias = args[0]
            args = args[1:]
        if len(args) > 0 and layer_past is None:
            layer_past = args[0]
            args = args[1:]
        if len(args) > 0:
            use_cache = args[0]

        # Call the original forward with cleaned arguments
        return self._original_forward(x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache)

    # Apply patch to all LLaDALlamaBlock instances
    for name, module in model.named_modules():
        if module.__class__.__name__ == "LLaDALlamaBlock":
            # Store original forward
            module._original_forward = module.forward
            # Replace with patched version
            module.forward = types.MethodType(patched_forward, module)
            logger.info(f"Patched {name}")

    # Move model to GPU
    logger.info("Moving model to GPU...")
    model = model.to("cuda")

    logger.info("Setting up GPTQ quantizer...")
    quantizer = GPTQQuantizer(
        bits=8,
        group_size=128,
        desc_act=False,
        sym=True,
        true_sequential=True,
        dataset="c4",
        tokenizer=tokenizer,
        block_name_to_quantize="model.transformer.blocks",
    )

    logger.info("Quantizing model...")
    quantizer.quantize_model(model, tokenizer)

    logger.info("Quantization done, saving model...")
    output_path = "/quantized-model-output/llada-8b-instruct-8bit-gptq"
    logger.info(f"Saving model to {output_path}")
    quantizer.save(model, output_path)
    tokenizer.save_pretrained(output_path)

    logger.success("Quantization done, model saved!")


@app.local_entrypoint()
def main() -> None:
    quantize_model.remote()
