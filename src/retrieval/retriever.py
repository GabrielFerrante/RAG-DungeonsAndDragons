"""Busca por similaridade multi-vetor: MaxSim (ColBERT) entre os tokens da pergunta e os da pagina."""
import numpy as np
import torch


def maxsim(q, q_mask, pages, p_mask):
    """MaxSim (ColBERT): para cada token da query, o melhor token da pagina; soma e normaliza pelo tamanho da query."""
    sim = torch.einsum("qd,pld->pql", q.float(), pages.float())  # [P, Lq, Lp]
    sim = sim.masked_fill(~p_mask.bool()[:, None, :], -1e4)
    return (sim.max(-1).values * q_mask.float()[None]).sum(-1) / q_mask.sum()  # [P]


def score_matrix(q_embs, p_embs, device="cuda", chunk: int = 1024) -> np.ndarray:
    """MaxSim normalizado pelo tamanho da pergunta. q_embs: lista de [Lq, D]; p_embs: lista de [Lp, D]. Retorna [Nq, Np]."""
    n_q, dim = len(q_embs), q_embs[0].shape[-1]
    scores = np.empty((n_q, len(p_embs)), dtype=np.float32)
    for start in range(0, n_q, chunk):
        block = q_embs[start : start + chunk]
        length = max(e.shape[0] for e in block)
        q = torch.zeros(len(block), length, dim, device=device)
        mask = torch.zeros(len(block), length, device=device)
        for i, e in enumerate(block):
            q[i, : e.shape[0]] = e.to(device).float()
            mask[i, : e.shape[0]] = 1
        for j, p in enumerate(p_embs):
            sim = torch.einsum("qld,md->qlm", q, p.to(device).float())  # [Nq, Lq, Lp]
            scores[start : start + len(block), j] = ((sim.max(-1).values * mask).sum(-1) / mask.sum(-1)).cpu().numpy()
    return scores
