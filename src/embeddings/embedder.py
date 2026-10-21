"""Embeddings multi-vetor (ColQwen3) de paginas e perguntas.

4-bit NF4 (bitsandbytes) tanto na GPU quanto na CPU (fallback, ver
src/utils/helpers.py::resolve_device_mode) — a partir do bitsandbytes 0.50, 4-bit tambem roda em CPU
(testado neste projeto; versoes mais antigas so suportavam CUDA). Em CPU o compute_dtype e float32
(mais previsivel entre CPUs diferentes que bfloat16) e o modelo fica em RAM, nao na GPU.
"""
import torch
from PIL import Image

from src.chunking.chunker import fit_visual_tokens
from src.utils.helpers import CONFIG, resolve_device_mode

DEFAULT_BASE = CONFIG["models"]["retriever"]
DEFAULT_REVISION = CONFIG["models"].get("retriever_revision")  # ver comentario em config.yaml
LM_HEAD_KEY = r"^vlm\.lm_head\.weight$"


def load_image(args, pages, key) -> Image.Image:
    image = Image.open(args.data / pages[key]["image"]).convert("RGB")
    return fit_visual_tokens(image, args.max_visual_tokens)


def ignore_tied_lm_head_in_report(base: str, revision: str | None):
    """Tira o falso "vlm.lm_head.weight | MISSING" do relatorio de carga.

    O checkpoint nao guarda vlm.lm_head.weight (e o mesmo tensor do embed_tokens). O ColQwen3 faz esse vinculo,
    mas o tie_weights dele nao repassa missing_keys ao carregador, que entao reporta a chave como faltando
    mesmo ja vinculada. Declara-la como ignoravel remove o aviso; check_lm_head_tied garante que o vinculo existe.
    """
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    # mesma revisao do from_pretrained logo abaixo: sem isso, poderiam resolver commits diferentes de "main" e
    # o patch cairia numa classe (cache de modulo) diferente da que o modelo de fato instancia
    colqwen3 = get_class_from_dynamic_module("modeling_colqwen3.ColQwen3", base, revision=revision)
    colqwen3._keys_to_ignore_on_load_missing = [LM_HEAD_KEY]


def check_lm_head_tied(model):
    """O lm_head tem de ser o proprio tensor do embed_tokens. Se nao for (ex.: o bnb o quantizou a parte), ele
    ficou com pesos aleatorios, que era o que o aviso MISSING denunciava de verdade."""
    if model.get_output_embeddings().weight is not model.get_input_embeddings().weight:
        raise RuntimeError("vlm.lm_head nao esta amarrado ao embed_tokens: confira llm_int8_skip_modules em load_retriever.")


def load_retriever(args):
    from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    # revision=None (args.base != DEFAULT_BASE) cai no comportamento padrao do transformers: ultimo commit de "main"
    revision = DEFAULT_REVISION if args.base == DEFAULT_BASE else None
    ignore_tied_lm_head_in_report(args.base, revision)
    processor = AutoProcessor.from_pretrained(
        args.base, revision=revision, trust_remote_code=True, max_num_visual_tokens=args.max_visual_tokens
    )

    on_cpu = resolve_device_mode() == "cpu"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32 if on_cpu else torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        # torre visual e cabeca de projecao fora da quantizacao; lm_head tambem fica fora: e so Linear,
        # nunca e lido pelo forward do ColQwen3 (que chama model.vlm.model, pulando a lm_head), mas sem
        # este skip o bnb trocava a Linear tied por uma Linear4bit solta, descasando do embed_tokens
        # e gerando o aviso "vlm.lm_head.weight | MISSING" (pesos aleatorios, so desperdicio de memoria)
        llm_int8_skip_modules=["visual", "embedding_proj_layer", "lm_head"],
    )
    model = AutoModel.from_pretrained(
        args.base, revision=revision, trust_remote_code=True, quantization_config=bnb,
        dtype=torch.float32 if on_cpu else torch.bfloat16, attn_implementation="sdpa",
        device_map="cpu" if on_cpu else {"": 0},
    )
    check_lm_head_tied(model)
    return model, processor


def to_device(batch, model):
    """Move o batch pro mesmo device do modelo (cuda ou cpu) — nunca assume qual dos dois."""
    device = next(model.parameters()).device
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def embed_pages(model, processor, images):
    feats = to_device(processor.process_images(images=images), model)
    emb = model(**feats, use_cache=False).embeddings  # [B, L, D]
    return emb, feats.get("attention_mask", torch.ones(emb.shape[:2], dtype=torch.long, device=emb.device))


def embed_query(model, processor, text):
    feats = to_device(processor.process_texts(texts=[text]), model)
    emb = model(**feats, use_cache=False).embeddings[0]  # [Lq, D]
    return emb, feats["attention_mask"][0] if "attention_mask" in feats else torch.ones(emb.shape[0], dtype=torch.long, device=emb.device)
