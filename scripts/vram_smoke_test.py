"""Spike de viabilidade de VRAM (Fase 0 do plano).

Mede pico de VRAM e tempo por passo na GPU local (RTX 3060, 8 GB) usando UM modelo base
(Qwen3-VL-4B-Instruct em 4-bit NF4) como proxy dos dois papeis:

  retriever-infer  forward de 1 pagina (como o ColQwen3 indexando)      -> s/pagina, VRAM
  retriever-train  1 passo LoRA + perda contrastiva MaxSim              -> s/passo, VRAM
  gen-infer        3 paginas + pergunta -> generate()                   -> tokens/s, VRAM
  gen-train        1 passo QLoRA (perda so nos tokens da resposta)      -> s/passo, VRAM

O Ops-ColQwen3-4B tem a mesma arquitetura do Qwen3-VL-4B-Instruct, entao o custo de VRAM
medido aqui vale para ele sem precisar baixar outro checkpoint de ~9 GB.

A resolucao e controlada pela escala de render do PDF: tokens visuais ~= 489 * escala^2
(pagina A4, patch 16 com merge 2x2 -> 32 px por token).

Uso:  conda run -n ragded python scripts/vram_smoke_test.py --stages all
"""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
QUESTION = "Qual a classe de armadura e os pontos de vida da criatura descrita nestas paginas?"
ANSWER = "A criatura tem Classe de Armadura 18 (armadura natural) e 185 pontos de vida (16d10 + 96)."
QUERIES = ["Qual a CA e os PV do Nalfeshnee?", "Quais sao os ataques multiplos de um demonio?"]


def reset_vram():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def peak_gb():
    return round(torch.cuda.max_memory_allocated() / 1e9, 2), round(torch.cuda.max_memory_reserved() / 1e9, 2)


def render_pages(pdf_path, page_indexes, scale):
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    return [pdf[i].render(scale=scale).to_pil().convert("RGB") for i in page_indexes]


def load_model(name):
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        # torre visual em bf16 (congelada); lm_head fora porque tem pesos compartilhados com os
        # embeddings (tie_word_embeddings) e o bitsandbytes falha ao quantizar um lm_head "tied"
        llm_int8_skip_modules=["visual", "lm_head"],
    )
    model = AutoModelForImageTextToText.from_pretrained(
        name, quantization_config=bnb, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0}
    )
    return model, AutoProcessor.from_pretrained(name)


def chat_inputs(processor, conversations, add_generation_prompt=True):
    return processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"padding": True},
    ).to("cuda")


def page_conversation(images, text):
    content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": text}]
    return [{"role": "user", "content": content}]


def image_tokens(inputs):
    return int((inputs["image_grid_thw"].prod(-1) // 4).sum())


def maxsim_loss(q, q_mask, p, p_mask, temperature=0.02):
    """Perda contrastiva late-interaction: a consulta i deve casar com a pagina i."""
    sim = torch.einsum("qid,pjd->qpij", q, p)
    sim = sim.masked_fill(~p_mask.bool()[None, :, None, :], -1e4)
    scores = sim.max(-1).values.mul(q_mask[:, None, :]).sum(-1) / q_mask.sum(-1)[:, None]
    return F.cross_entropy(scores / temperature, torch.arange(q.shape[0], device=q.device))


def hidden_size(model):
    cfg = model.config
    return getattr(cfg, "text_config", cfg).hidden_size


def measure(fn):
    """Executa fn() medindo tempo e pico de VRAM; registra OOM em vez de abortar.

    Faz uma passada de aquecimento antes (inicializacao do CUDA/kernels polui o 1o tempo);
    o pico de VRAM e o tempo reportados sao os da segunda passada.
    """
    reset_vram()
    try:
        fn()  # aquecimento
        reset_vram()
        start = time.perf_counter()
        extra = fn() or {}
        torch.cuda.synchronize()
        seconds, status = round(time.perf_counter() - start, 2), "ok"
    except torch.cuda.OutOfMemoryError:
        extra, seconds, status = {}, None, "OOM"
    alloc, reserved = peak_gb()
    return {"status": status, "seconds": seconds, "peak_alloc_gb": alloc, "peak_reserved_gb": reserved, **extra}


def stage_retriever_infer(args, results):
    model, processor = load_model(args.model)
    model.eval()
    inner = model.model
    head = torch.nn.Linear(hidden_size(model), 128, device="cuda")
    for scale in args.scales:
        images = render_pages(args.pdf, args.pages[:1], scale)
        inputs = chat_inputs(processor, [page_conversation(images, "Descreva a imagem.")])

        def run():
            with torch.inference_mode():
                head(inner(**inputs).last_hidden_state.float())
            return {"image_tokens": image_tokens(inputs)}

        results.append({"stage": "retriever-infer", "scale": scale, **measure(run)})
    del model, inner, head


def stage_retriever_train(args, results):
    from bitsandbytes.optim import PagedAdamW8bit
    from peft import LoraConfig, get_peft_model

    model, processor = load_model(args.model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    inner = model.model
    head = torch.nn.Linear(hidden_size(model), 128, device="cuda")
    lora = LoraConfig(
        r=args.retriever_rank,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    )
    model = get_peft_model(model, lora)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = PagedAdamW8bit(params, lr=5e-5)

    for scale in args.scales:
        images = render_pages(args.pdf, args.pages[:2], scale)  # 2 paginas = 1 positivo + 1 hard negative
        p_inputs = chat_inputs(processor, [page_conversation([im], "Descreva a imagem.") for im in images])
        q_inputs = processor.tokenizer(QUERIES, padding=True, return_tensors="pt").to("cuda")

        def run():
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = F.normalize(head(inner(**p_inputs).last_hidden_state.float()), dim=-1)
                q = F.normalize(head(inner(**q_inputs).last_hidden_state.float()), dim=-1)
            loss = maxsim_loss(q, q_inputs["attention_mask"].float(), p, p_inputs["attention_mask"])
            loss.backward()
            optimizer.step()
            return {"image_tokens": image_tokens(p_inputs), "loss": round(loss.item(), 4)}

        results.append({"stage": "retriever-train", "scale": scale, **measure(run)})
    del model, inner, head, optimizer


def stage_gen_infer(args, results):
    model, processor = load_model(args.model)
    model.eval()
    for scale in args.scales:
        images = render_pages(args.pdf, args.pages[:3], scale)
        inputs = chat_inputs(processor, [page_conversation(images, QUESTION)])

        def run():
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
            return {"image_tokens": image_tokens(inputs), "new_tokens": new_tokens}

        r = measure(run)
        if r["status"] == "ok":
            r["tokens_per_s"] = round(r["new_tokens"] / r["seconds"], 2)
        results.append({"stage": "gen-infer", "scale": scale, **r})
    del model


def stage_gen_train(args, results):
    from bitsandbytes.optim import PagedAdamW8bit
    from peft import LoraConfig, get_peft_model

    model, processor = load_model(args.model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    inner, lm_head = model.model, model.lm_head
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    )
    model = get_peft_model(model, lora)
    model.train()
    optimizer = PagedAdamW8bit([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    for scale in args.scales:
        images = render_pages(args.pdf, args.pages[:2], scale)
        prompt = page_conversation(images, QUESTION)
        full = prompt + [{"role": "assistant", "content": [{"type": "text", "text": ANSWER}]}]
        n_prompt = chat_inputs(processor, [prompt])["input_ids"].shape[1]
        f_inputs = chat_inputs(processor, [full], add_generation_prompt=False)

        def run():
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # lm_head so nas posicoes da resposta: evita logits de vocabulario x sequencia inteira
                hidden = inner(**f_inputs).last_hidden_state[:, n_prompt - 1 : -1]
                logits = lm_head(hidden).float()
            labels = f_inputs["input_ids"][:, n_prompt:]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            loss.backward()
            optimizer.step()
            return {"seq_len": f_inputs["input_ids"].shape[1], "loss": round(loss.item(), 4)}

        results.append({"stage": "gen-train", "scale": scale, **measure(run)})
    del model, inner, lm_head, optimizer


STAGES = {
    "retriever-infer": stage_retriever_infer,
    "retriever-train": stage_retriever_train,
    "gen-infer": stage_gen_infer,
    "gen-train": stage_gen_train,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--pdf", type=Path, default=ROOT / "ArquivosEFontes" / "MonstersBook.pdf")
    parser.add_argument("--pages", type=int, nargs="+", default=[60, 61, 62], help="indices 0-based")
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0, 1.4, 1.6])
    parser.add_argument("--stages", nargs="+", default=["all"], choices=["all", *STAGES])
    parser.add_argument("--retriever-rank", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA indisponivel no env atual"
    print(f"GPU: {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    out = ROOT / "outputs" / "vram_smoke_test.json"
    out.parent.mkdir(exist_ok=True)
    results = []
    for name in (STAGES if "all" in args.stages else args.stages):
        print(f"\n== {name} ==", flush=True)
        STAGES[name](args, results)
        reset_vram()
        for r in results:
            if r["stage"] == name:
                print(json.dumps(r, ensure_ascii=False), flush=True)
        # grava a cada etapa: uma falha adiante nao perde o que ja foi medido
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultados em {out}")


if __name__ == "__main__":
    main()
