"""Fine-tuning do retriever visual (ColQwen3, multi-vetor) com LoRA sobre base 4-bit (QLoRA).

Parte de um retriever JA treinado (TomoroAI/tomoro-colqwen3-embed-4b, Apache-2.0, multilingue) e o
adapta ao dominio: paginas dos livros de D&D 5E em pt-BR + perguntas sinteticas (make_synthetic.py).

Cada micro-passo = 1 pergunta + 1 pagina positiva + K hard negatives (TF-IDF do texto da pagina);
perda InfoNCE sobre MaxSim. O batch efetivo vem de --grad-accum (batch grande nao cabe em 8 GB).

Feito para rodar sem supervisao por horas:
  - avalia ANTES de treinar (baseline) e a cada --eval-every passos, num split por capitulo
    (blocos de paginas) que nunca entra no treino; guarda o melhor adapter em best/;
  - checkpoint a cada --save-every passos e retomada automatica (rode o mesmo comando de novo);
  - pula micro-passos com OOM ou perda nao finita em vez de morrer; Ctrl+C salva antes de sair;
  - impede o Windows de suspender enquanto roda.

Uso:  python main.py train [--dry-run] [--epochs 1] [--max-hours 12]   (ou: python -m src.train.train_retriever)
"""
import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.embeddings.embedder import DEFAULT_BASE, embed_pages, embed_query, load_image, load_retriever
from src.eval.metrics import rank_of, retrieval_metrics
from src.retrieval.retriever import maxsim
from src.train.data import load_pages, load_queries, mine_negatives, split_pages
from src.utils.helpers import CONFIG, ROOT, keep_awake

LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "data")
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "retriever-lora")
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--dry-run", action="store_true", help="so prepara/mostra os dados; nao carrega modelo nem treina")
    # dados
    p.add_argument("--min-chars", type=int, default=CONFIG["chunking"]["min_chars"])
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--val-block", type=int, default=20, help="tamanho (paginas) do bloco de validacao")
    p.add_argument("--val-queries", type=int, default=200)
    p.add_argument("--distractors", type=int, default=100, help="paginas de treino extras no pool de avaliacao")
    p.add_argument("--neg-per-query", type=int, default=2)
    p.add_argument("--neg-pool", type=int, default=8, help="sorteia os negativos entre os top-K por TF-IDF")
    # modelo / memoria
    p.add_argument("--max-visual-tokens", type=int, default=768, help="tokens visuais por pagina (memoria x nitidez)")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)
    # otimizacao
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.02)
    p.add_argument("--max-steps", type=int, default=0, help="limita passos de otimizacao (0 = sem limite)")
    p.add_argument("--max-hours", type=float, default=0, help="para (salvando) apos N horas (0 = sem limite)")
    # rotina
    p.add_argument("--eval-every", type=int, default=50, help="passos de otimizacao; 0 desliga")
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--keep", type=int, default=2, help="quantos checkpoints step-* manter")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ----------------------------------------------------------------------------- dados

def build_data(args):
    for needed, step in [("manifest.jsonl", "ingest"), ("synth/queries.jsonl", "synthetic")]:
        if not (args.data / needed).exists():
            sys.exit(f"Falta {args.data / needed}: rode `python main.py {step}` antes.")
    pages = load_pages(args.data)
    queries = load_queries(args.data)
    keys = sorted(k for k, r in pages.items() if r["n_chars"] >= args.min_chars)
    train_pages, val_pages = split_pages(keys, args.val_fraction, args.val_block, seed=args.seed)
    texts = [(args.data / pages[k]["text"]).read_text(encoding="utf-8") for k in train_pages]
    negatives = mine_negatives(train_pages, texts, top_k=args.neg_pool)

    train = [(k, q) for k in train_pages for q in queries.get(k, [])]
    val = [(k, q) for k in val_pages for q in queries.get(k, [])]
    rng = random.Random(args.seed)
    rng.shuffle(val)
    val = val[: args.val_queries]

    val_keys = sorted({k for k, _ in val})
    val_set = set(val_keys)
    extra = rng.sample(train_pages, min(args.distractors, len(train_pages)))
    pool = val_keys + [k for k in extra if k not in val_set]
    return pages, train, val, negatives, pool


# ----------------------------------------------------------------------------- modelo

def add_lora(model, args):
    from peft import LoraConfig, get_peft_model

    # O wrapper ColQwen3 nao declara suporte a checkpointing, mas chama vlm.model(...) diretamente:
    # ligar no VLM interno basta.
    model.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    config = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
        target_modules=LORA_TARGETS, modules_to_save=["embedding_proj_layer"],
    )
    peft_model = get_peft_model(model, config)
    for p in peft_model.parameters():  # cabeca treinavel em fp32: com lr pequeno, bf16 perderia os updates
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    peft_model.print_trainable_parameters()
    return peft_model


@torch.inference_mode()
def evaluate(model, processor, args, pages, pool, val):
    was_training = model.training
    model.eval()
    index = {k: i for i, k in enumerate(pool)}
    pool_emb = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for key in pool:
            emb, _ = embed_pages(model, processor, [load_image(args, pages, key)])
            pool_emb.append(emb[0].to(torch.bfloat16))
        ranks = []
        for key, query in val:
            q, q_mask = embed_query(model, processor, query)
            scores = torch.stack([maxsim(q, q_mask, e[None], torch.ones(1, e.shape[0], dtype=torch.bool, device="cuda"))[0] for e in pool_emb])
            ranks.append(rank_of(scores.float().cpu().numpy(), index[key]))
    model.train(was_training)
    return retrieval_metrics(ranks)


# ----------------------------------------------------------------------------- checkpoints

def save_checkpoint(model, optimizer, out: Path, name: str, state: dict, optimizer_state: bool = True):
    path = out / name
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    if optimizer_state:
        torch.save({"optimizer": optimizer.state_dict(), **state}, path / "trainer_state.pt")


def rotate_checkpoints(out: Path, keep: int):
    for old in sorted(out.glob("step-*"))[:-keep]:
        shutil.rmtree(old, ignore_errors=True)


def resume_from(model, optimizer, path: Path) -> dict:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    set_peft_model_state_dict(model, load_file(path / "adapter_model.safetensors"))
    state = torch.load(path / "trainer_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state.pop("optimizer"))
    return state


def lr_at(step: int, total: int, base: float, warmup_frac: float = 0.025) -> float:
    warmup = max(1, int(total * warmup_frac))
    if step < warmup:
        return base * (step + 1) / warmup
    return base * max(0.0, (total - step) / max(1, total - warmup))


def log(out: Path, row: dict):
    print(json.dumps(row, ensure_ascii=False), flush=True)
    with (out / "log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------- main

def main():
    args = parse_args()
    pages, train, val, negatives, pool = build_data(args)
    steps_per_epoch = max(1, len(train) // args.grad_accum)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    print(
        f"paginas: {len(pages)} | treino: {len(train)} perguntas | validacao: {len(val)} perguntas em {len(pool)} paginas de pool\n"
        f"passos de otimizacao: {total_steps} ({steps_per_epoch}/epoca x {args.epochs}) | batch efetivo: {args.grad_accum}",
        flush=True,
    )
    if not train or not val:
        sys.exit("Sem perguntas de treino/validacao: rode (ou termine) `python main.py synthetic` antes.")
    if args.dry_run:
        key, query = train[0]
        print(f"exemplo: {key} -> {query!r}\n  negativos: {negatives[key][: args.neg_per_query]}")
        return

    from bitsandbytes.optim import PagedAdamW8bit

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2), encoding="utf-8")
    torch.manual_seed(args.seed)
    keep_awake(True)

    model, processor = load_retriever(args)
    model = add_lora(model, args)
    optimizer = PagedAdamW8bit([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    state = {"micro": 0, "best": None, "baseline": None}
    checkpoints = sorted(args.out.glob("step-*"))
    if checkpoints and not args.no_resume:
        state.update(resume_from(model, optimizer, checkpoints[-1]))
        print(f"retomando de {checkpoints[-1].name} (micro-passo {state['micro']})", flush=True)

    model.train()
    if state["baseline"] is None:
        state["baseline"] = evaluate(model, processor, args, pages, pool, val)
        log(args.out, {"event": "baseline", **state["baseline"]})

    def snapshot(name, optimizer_state=True):
        save_checkpoint(model, optimizer, args.out, name, dict(state), optimizer_state)

    n = len(train)
    total_micro = total_steps * args.grad_accum
    orders = {}  # epoca -> permutacao (deterministica pela seed: a retomada reproduz a mesma ordem)
    started, window_loss, window_n, oom_streak, skipped = time.time(), 0.0, 0, 0, 0
    try:
        while state["micro"] < total_micro:
            micro = state["micro"]
            epoch, pos = divmod(micro, n)
            if epoch not in orders:
                orders[epoch] = random.Random(args.seed + epoch).sample(range(n), n)
            key, query = train[orders[epoch][pos]]
            negs = random.Random(args.seed * 1_000_003 + micro).sample(negatives[key], min(args.neg_per_query, len(negatives[key])))
            state["micro"] += 1

            try:
                images = [load_image(args, pages, k) for k in [key, *negs]]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p_emb, p_mask = embed_pages(model, processor, images)
                    q_emb, q_mask = embed_query(model, processor, query)
                scores = maxsim(q_emb, q_mask, p_emb, p_mask)
                loss = F.cross_entropy(scores[None] / args.temperature, torch.zeros(1, dtype=torch.long, device="cuda"))
                if torch.isfinite(loss):
                    (loss / args.grad_accum).backward()
                    window_loss, window_n, oom_streak = window_loss + loss.item(), window_n + 1, 0
                else:
                    skipped += 1
            except torch.cuda.OutOfMemoryError:
                skipped, oom_streak = skipped + 1, oom_streak + 1
                p_emb = q_emb = scores = loss = None
                torch.cuda.empty_cache()
                if oom_streak >= 10:
                    raise RuntimeError("10 OOM seguidos: reduza --max-visual-tokens ou --neg-per-query")

            # (o pulo por OOM/NaN nao pode furar a fronteira da janela, senao o passo de otimizacao se perde)
            if state["micro"] % args.grad_accum:
                continue

            # ---- fim de um passo de otimizacao
            step = state["micro"] // args.grad_accum
            if window_n == 0:  # janela inteira pulada: nada a otimizar
                optimizer.zero_grad(set_to_none=True)
                continue
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step - 1, total_steps, args.lr)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            elapsed = time.time() - started
            log(args.out, {
                "event": "step", "step": step, "of": total_steps, "epoch": round(state["micro"] / n, 3),
                "loss": round(window_loss / max(window_n, 1), 4), "lr": round(optimizer.param_groups[0]["lr"], 8),
                "skipped": skipped, "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "elapsed_h": round(elapsed / 3600, 2),
            })
            window_loss, window_n = 0.0, 0

            if args.eval_every and step % args.eval_every == 0:
                metrics = evaluate(model, processor, args, pages, pool, val)
                log(args.out, {"event": "eval", "step": step, **metrics})
                best = state["best"]
                if best is None or (metrics["recall@5"], metrics["mrr"]) > (best["recall@5"], best["mrr"]):
                    state["best"] = {"step": step, **metrics}
                    save_checkpoint(model, optimizer, args.out, "best", state, optimizer_state=False)
            if args.save_every and step % args.save_every == 0:
                snapshot(f"step-{step:06d}")
                rotate_checkpoints(args.out, args.keep)
            if args.max_hours and elapsed > args.max_hours * 3600:
                print(f"--max-hours atingido ({args.max_hours} h): salvando e saindo.", flush=True)
                break
        # o contador salvo tem de cair numa fronteira de passo: gradientes parciais da janela em curso se perdem
        state["micro"] -= state["micro"] % args.grad_accum
        snapshot(f"step-{state['micro'] // args.grad_accum:06d}")
        rotate_checkpoints(args.out, args.keep)
        final = evaluate(model, processor, args, pages, pool, val)
        log(args.out, {"event": "final", **final})
        save_checkpoint(model, optimizer, args.out, "final", state, optimizer_state=False)
        summary = {"baseline": state["baseline"], "best": state["best"], "final": final}
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    except KeyboardInterrupt:
        print("Interrompido: salvando checkpoint antes de sair.", flush=True)
        state["micro"] -= state["micro"] % args.grad_accum
        snapshot(f"step-{state['micro'] // args.grad_accum:06d}")
    except BaseException:
        # mesmo nome dos periodicos (step-*): a retomada automatica o encontra. Se o proprio snapshot falhar
        # (ex.: disco cheio), o erro original ainda e o que sobe.
        state["micro"] -= state["micro"] % args.grad_accum
        try:
            snapshot(f"step-{state['micro'] // args.grad_accum:06d}")
        except Exception as save_error:
            print(f"Falha ao salvar o checkpoint de emergencia: {save_error!r}", flush=True)
        raise
    finally:
        keep_awake(False)


if __name__ == "__main__":
    main()
