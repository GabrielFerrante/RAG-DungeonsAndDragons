"""Carrega o Qwen3-VL em 4-bit NF4 (bitsandbytes), na GPU (RTX 3060 de 8 GB; medido em
scripts/vram_smoke_test.py) ou em CPU/RAM quando resolve_device_mode() cai pra CPU — a partir do
bitsandbytes 0.50, 4-bit tambem roda em CPU (testado neste projeto)."""
import torch

from src.utils.helpers import CONFIG, resolve_device_mode

DEFAULT_MODEL = CONFIG["models"]["vlm"]


def load_qwen3vl_4bit(name: str = DEFAULT_MODEL):
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()  # sem barras de "Loading weights" nos logs de execucoes longas

    on_cpu = resolve_device_mode() == "cpu"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32 if on_cpu else torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        # torre visual fora da quantizacao; lm_head fora porque compartilha pesos com os embeddings
        llm_int8_skip_modules=["visual", "lm_head"],
    )
    model = AutoModelForImageTextToText.from_pretrained(
        name, quantization_config=bnb, dtype=torch.float32 if on_cpu else torch.bfloat16,
        attn_implementation="sdpa", device_map="cpu" if on_cpu else {"": 0},
    )
    return model, AutoProcessor.from_pretrained(name)
