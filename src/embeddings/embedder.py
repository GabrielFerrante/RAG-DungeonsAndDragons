"""Embeddings multi-vetor (ColQwen3) de paginas e perguntas, com o modelo base em 4-bit."""
import torch
from PIL import Image

from src.chunking.chunker import fit_visual_tokens
from src.utils.helpers import CONFIG

DEFAULT_BASE = CONFIG["models"]["retriever"]


def load_image(args, pages, key) -> Image.Image:
    image = Image.open(args.data / pages[key]["image"]).convert("RGB")
    return fit_visual_tokens(image, args.max_visual_tokens)


def load_retriever(args):
    from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True, max_num_visual_tokens=args.max_visual_tokens)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["visual", "embedding_proj_layer"],  # torre visual e cabeca em bf16
    )
    model = AutoModel.from_pretrained(
        args.base, trust_remote_code=True, quantization_config=bnb, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": 0},
    )
    return model, processor


def to_cuda(batch):
    return {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def embed_pages(model, processor, images):
    feats = to_cuda(processor.process_images(images=images))
    emb = model(**feats, use_cache=False).embeddings  # [B, L, D]
    return emb, feats.get("attention_mask", torch.ones(emb.shape[:2], dtype=torch.long, device=emb.device))


def embed_query(model, processor, text):
    feats = to_cuda(processor.process_texts(texts=[text]))
    emb = model(**feats, use_cache=False).embeddings[0]  # [Lq, D]
    return emb, feats["attention_mask"][0] if "attention_mask" in feats else torch.ones(emb.shape[0], dtype=torch.long, device=emb.device)
