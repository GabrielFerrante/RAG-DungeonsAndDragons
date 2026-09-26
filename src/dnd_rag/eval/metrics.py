"""Metricas de retrieval (NumPy). A relevancia e binaria: so a pagina-ouro conta."""
import numpy as np


def rank_of(scores: np.ndarray, gold: int) -> int:
    """Posicao (1-based) da pagina-ouro no ranking; empates contam contra (visao pessimista)."""
    others = np.delete(scores, gold)
    return int((others >= scores[gold]).sum()) + 1


def retrieval_metrics(ranks, ks=(1, 5, 10)) -> dict:
    r = np.asarray(ranks, dtype=np.float64)
    out = {f"recall@{k}": float((r <= k).mean()) for k in ks}
    out["mrr"] = float((1.0 / r).mean())
    out["ndcg@5"] = float(np.where(r <= 5, 1.0 / np.log2(r + 1), 0.0).mean())
    out["n"] = int(r.size)
    return out
