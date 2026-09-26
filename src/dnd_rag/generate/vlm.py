"""Carrega o Qwen3-VL em 4-bit NF4 (cabe na RTX 3060 de 8 GB; medido em scripts/vram_smoke_test.py)."""
import math

import torch
from PIL import Image

DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
TOKEN_PIXELS = 32 * 32  # patch 16 com merge 2x2 => 1 token visual por 32x32 px


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


def fit_visual_tokens(image: Image.Image, max_tokens: int) -> Image.Image:
    """Reduz a imagem (mantendo a proporcao) para que gere no maximo ~max_tokens tokens visuais."""
    factor = math.sqrt(max_tokens * TOKEN_PIXELS / (image.width * image.height))
    if factor >= 1:
        return image
    return image.resize((max(32, int(image.width * factor)), max(32, int(image.height * factor))), Image.LANCZOS)
