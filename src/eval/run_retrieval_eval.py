"""Avaliacao de retrieval nas duas direcoes, top-K (Recall@K, MRR, mediana/media do rank).

  T2I (texto -> imagem): dada uma pergunta, em que posicao aparece a pagina-ouro entre as paginas do pool?
  I2T (imagem -> texto): dada uma pagina, em que posicao aparece a MELHOR das suas perguntas entre TODAS
                         as perguntas do conjunto? (protocolo COCO/CLIP: acerto se qualquer pergunta da
                         pagina estiver no top-K)

O score e o MaxSim normalizado pelo tamanho da pergunta (o mesmo da perda de treino). Como o MaxSim e
assimetrico, o ranking I2T compara perguntas de tamanhos diferentes: a normalizacao e o que as torna comparaveis.

Por padrao avalia o split de VALIDACAO (paginas que nunca entram no treino). Com --adapter, o split e as
opcoes de modelo sao lidos do config.json do treino (ao lado do adapter), para nao avaliar por engano em
paginas que o adapter viu.

Uso:
  python main.py eval                                   # modelo original
  python main.py eval --adapter outputs\\retriever-lora\\best --compare
  python main.py eval --dry-run                         # so mostra o conjunto de avaliacao
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

from src.embeddings.embedder import DEFAULT_BASE, embed_pages, embed_query, load_image, load_retriever
from src.eval.metrics import rank_of, retrieval_metrics
from src.retrieval.retriever import score_matrix
from src.train.data import load_pages, load_queries, split_pages
from src.utils.helpers import CONFIG, ROOT

# opcao -> (conversor, padrao). Quando --adapter e dado, o config.json do treino tem precedencia sobre o padrao.
TRAIN_OPTIONS = {
    "val_fraction": (float, 0.1),
    "val_block": (int, 20),
    "min_chars": (int, CONFIG["chunking"]["min_chars"]),
    "seed": (int, 0),
    "max_visual_tokens": (int, 768),
    "base": (str, DEFAULT_BASE),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "data")
    p.add_argument("--adapter", type=Path, help="pasta do adapter LoRA (ex.: outputs/retriever-lora/best)")
    p.add_argument("--compare", action="store_true", help="com --adapter: avalia tambem o modelo original e mostra a diferenca")
    p.add_argument("--split", choices=["val", "train", "all"], default="val")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--max-queries", type=int, default=0, help="limita as perguntas avaliadas (0 = todas)")
    p.add_argument("--distractors", type=int, default=0, help="paginas do outro split adicionadas ao pool de T2I")
    p.add_argument("--miss-k", type=int, default=5, help="T2I: perguntas com pagina-ouro fora do top-K vao para t2i_misses.jsonl")
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "eval")
    p.add_argument("--dry-run", action="store_true", help="so monta e mostra o conjunto de avaliacao; nao carrega modelo")
    for name, (cast, _) in TRAIN_OPTIONS.items():  # padrao None: resolvido por resolve_train_options
        p.add_argument("--" + name.replace("_", "-"), type=cast, default=None)
    args = p.parse_args()
    if args.compare and not args.adapter:
        p.error("--compare precisa de --adapter")
    return args


def resolve_train_options(args) -> dict[str, str]:
    """Preenche as opcoes nao informadas: config.json do treino > padrao. Retorna a origem de cada uma."""
    config = {}
    if args.adapter:
        config_path = args.adapter.parent / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8-sig"))  # tolera BOM (Windows/editores)
        else:
            print(f"AVISO: {config_path} nao existe; se o treino usou outro split, a avaliacao em 'val' pode vazar.", flush=True)
    origin = {}
    for name, (cast, default) in TRAIN_OPTIONS.items():
        if getattr(args, name) is not None:
            origin[name] = "linha de comando"
        elif name in config:
            setattr(args, name, cast(config[name]))
            origin[name] = "config do treino"
        else:
            setattr(args, name, default)
            origin[name] = "padrao"
    return origin


def select_eval_set(pages, queries, args):
    """Perguntas avaliadas e pool de paginas. Mesma logica (e mesma ordem de sorteio) da avaliacao do treino,
    entao com --split val --max-queries N --distractors M os numeros sao comparaveis com o log do treino."""
    keys = sorted(k for k, r in pages.items() if r["n_chars"] >= args.min_chars)
    train_pages, val_pages = split_pages(keys, args.val_fraction, args.val_block, seed=args.seed)
    chosen, others = {"val": (val_pages, train_pages), "train": (train_pages, val_pages), "all": (keys, [])}[args.split]

    items = [(k, q) for k in chosen for q in queries.get(k, [])]
    rng = random.Random(args.seed)
    rng.shuffle(items)
    if args.max_queries:
        items = items[: args.max_queries]

    gold = sorted({k for k, _ in items})
    gold_set = set(gold)
    extra = rng.sample(others, min(args.distractors, len(others)))
    pool = gold + [k for k in extra if k not in gold_set]
    return items, pool


def t2i_ranks(scores: np.ndarray, query_page: np.ndarray) -> list[int]:
    """Uma posicao por pergunta: onde fica a pagina-ouro entre todas as paginas."""
    return [rank_of(scores[i], int(g)) for i, g in enumerate(query_page)]


def i2t_ranks(scores: np.ndarray, query_page: np.ndarray) -> list[int]:
    """Uma posicao por pagina que tem perguntas: a melhor posicao entre as perguntas dela, ranqueando todas as perguntas."""
    ranks = []
    for page in np.unique(query_page):
        column = scores[:, page]
        ranks.append(min(rank_of(column, int(i)) for i in np.flatnonzero(query_page == page)))
    return ranks


def summarize(ranks, ks) -> dict:
    metrics = retrieval_metrics(ranks, ks)
    metrics["median_rank"] = float(np.median(ranks))
    metrics["mean_rank"] = float(np.mean(ranks))
    return metrics


@torch.inference_mode()
def embed_all(model, processor, args, pages, items, pool):
    p_embs, q_embs = [], []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i, key in enumerate(pool, 1):
            emb, _ = embed_pages(model, processor, [load_image(args, pages, key)])
            p_embs.append(emb[0].to(torch.bfloat16))
            if i % 25 == 0 or i == len(pool):
                print(f"  paginas: {i}/{len(pool)}", flush=True)
        for _, query in items:
            q_embs.append(embed_query(model, processor, query)[0].to(torch.bfloat16))
    return p_embs, q_embs


def evaluate_both(model, processor, args, pages, items, pool) -> dict:
    p_embs, q_embs = embed_all(model, processor, args, pages, items, pool)
    scores = score_matrix(q_embs, p_embs)
    index = {k: i for i, k in enumerate(pool)}
    query_page = np.array([index[k] for k, _ in items])
    t2i = t2i_ranks(scores, query_page)
    return {
        "T2I": summarize(t2i, args.ks),
        "I2T": summarize(i2t_ranks(scores, query_page), args.ks),
        "_scores": scores,
        "_t2i_ranks": t2i,
    }


def load_adapter(model, path: Path):
    from peft import PeftModel, set_peft_model_state_dict
    from safetensors.torch import load_file

    peft_model = PeftModel.from_pretrained(model, str(path))
    for name, param in peft_model.named_parameters():
        if "modules_to_save" in name and param.dtype != torch.float32:
            param.data = param.data.float()
    # o treino usa a cabeca em fp32; recarrega os pesos exatos (o 1o load os arredondou para o bf16 do original)
    set_peft_model_state_dict(peft_model, load_file(path / "adapter_model.safetensors"))
    return peft_model.eval()


def rsum(result: dict, ks) -> float:
    return 100 * sum(result[d][f"recall@{k}"] for d in ("T2I", "I2T") for k in ks)


def format_table(title: str, result: dict, ks) -> str:
    head = f"{'':5s}" + "".join(f"{'R@' + str(k):>8s}" for k in ks) + f"{'MRR':>8s}{'mediana':>9s}{'media':>8s}{'n':>7s}"
    rows = [
        f"{d:5s}" + "".join(f"{100 * result[d][f'recall@{k}']:8.1f}" for k in ks)
        + f"{result[d]['mrr']:8.3f}{result[d]['median_rank']:9.1f}{result[d]['mean_rank']:8.1f}{result[d]['n']:7d}"
        for d in ("T2I", "I2T")
    ]
    return "\n".join([f"== {title} ==", head, *rows, f"rsum (soma dos R@K, as 2 direcoes): {rsum(result, ks):.1f}"])


def format_delta(base: dict, tuned: dict, ks) -> str:
    lines = ["== ajustado - original (pontos percentuais em R@K; MRR em valor) =="]
    for d in ("T2I", "I2T"):
        parts = [f"R@{k} {100 * (tuned[d][f'recall@{k}'] - base[d][f'recall@{k}']):+.1f}" for k in ks]
        parts.append(f"MRR {tuned[d]['mrr'] - base[d]['mrr']:+.3f}")
        lines.append(f"{d}: " + " | ".join(parts))
    lines.append(f"rsum: {rsum(tuned, ks) - rsum(base, ks):+.1f}")
    return "\n".join(lines)


def write_misses(path: Path, result: dict, items, pool, miss_k: int, top: int = 3):
    scores, ranks = result["_scores"], result["_t2i_ranks"]
    with path.open("w", encoding="utf-8") as f:
        for i, (key, query) in enumerate(items):
            if ranks[i] > miss_k:
                best = np.argsort(-scores[i])[:top]
                f.write(json.dumps({
                    "query": query, "gold": list(key), "rank": ranks[i],
                    "top": [{"page": list(pool[j]), "score": round(float(scores[i, j]), 3)} for j in best],
                }, ensure_ascii=False) + "\n")


def public(result: dict) -> dict:
    return {k: v for k, v in result.items() if not k.startswith("_")}


def main():
    args = parse_args()
    origin = resolve_train_options(args)
    print("opcoes:", ", ".join(f"{n}={getattr(args, n)} ({origin[n]})" for n in TRAIN_OPTIONS), flush=True)

    pages = load_pages(args.data)
    items, pool = select_eval_set(pages, load_queries(args.data), args)
    n_pages = len({k for k, _ in items})
    print(f"split '{args.split}': {len(items)} perguntas em {n_pages} paginas | pool T2I: {len(pool)} paginas", flush=True)
    if not items:
        sys.exit("Nenhuma pergunta no split escolhido: gere/termine as perguntas sinteticas (python main.py synthetic).")
    if args.dry_run:
        print(f"exemplo: {items[0][0]} -> {items[0][1]!r}")
        return

    model, processor = load_retriever(args)
    model.eval()
    if args.adapter:
        model = load_adapter(model, args.adapter)

    results = {}
    label = args.adapter.name if args.adapter else "original"
    print(f"\navaliando: {label}", flush=True)
    results[label] = evaluate_both(model, processor, args, pages, items, pool)
    print(format_table(f"{label} | split={args.split}", results[label], args.ks), flush=True)
    if args.compare:
        print("\navaliando: original (adapter desligado)", flush=True)
        with model.disable_adapter():
            results["original"] = evaluate_both(model, processor, args, pages, items, pool)
        print(format_table(f"original | split={args.split}", results["original"], args.ks), flush=True)
        print(format_delta(results["original"], results[label], args.ks), flush=True)

    out = args.out / (args.adapter.parent.name + "-" + label if args.adapter else "original")
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "split": args.split, "n_queries": len(items), "n_pages": n_pages, "pool": len(pool),
        "options": {n: getattr(args, n) for n in TRAIN_OPTIONS} | {"adapter": str(args.adapter) if args.adapter else None},
        "results": {name: public(r) for name, r in results.items()},
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_misses(out / "t2i_misses.jsonl", results[label], items, pool, args.miss_k)
    print(f"\nResultados em {out}", flush=True)


if __name__ == "__main__":
    main()
