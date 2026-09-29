"""Carrega o Qwen3-VL em 4-bit NF4 (cabe na RTX 3060 de 8 GB; medido em scripts/vram_smoke_test.py)."""
import torch

from src.utils.helpers import CONFIG

DEFAULT_MODEL = CONFIG["models"]["vlm"]


def load_qwen3vl_4bit(name: str = DEFAULT_MODEL):
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()  # sem barras de "Loading weights" nos logs de execucoes longas
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        # torre visual em bf16; lm_head fora porque compartilha pesos com os embeddings
        llm_int8_skip_modules=["visual", "lm_head"],
    )
    model = AutoModelForImageTextToText.from_pretrained(
        name, quantization_config=bnb, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0}
    )
    return model, AutoProcessor.from_pretrained(name)
