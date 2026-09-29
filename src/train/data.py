"""Dados do fine-tuning do retriever: leitura, split sem vazamento e hard negatives."""
import math
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

from src.utils.helpers import read_jsonl

Key = tuple[str, int]  # (livro, pagina 1-based)


def load_pages(data_dir: Path) -> dict[Key, dict]:
    return {(r["book"], r["page"]): r for r in read_jsonl(data_dir / "manifest.jsonl")}


def load_queries(data_dir: Path) -> dict[Key, list[str]]:
    path = data_dir / "synth" / "queries.jsonl"
    return {(r["book"], r["page"]): r["queries"] for r in read_jsonl(path) if r["queries"]}


def split_pages(keys: list[Key], val_fraction: float = 0.1, block: int = 20, margin: int = 2, seed: int = 0):
    """Split treino/validacao por BLOCOS de paginas consecutivas (proxy de capitulo), nao por pergunta.

    Perguntas da mesma pagina/secao no treino e na validacao inflariam a metrica. As `margin` paginas
    ao redor de cada bloco de validacao ficam fora de ambos os lados (um stat block pode continuar
    na pagina seguinte).
    """
    blocks = sorted({(book, page // block) for book, page in keys})
    random.Random(seed).shuffle(blocks)
    val_blocks = set(blocks[: max(1, round(len(blocks) * val_fraction))])
    val = [k for k in keys if (k[0], k[1] // block) in val_blocks]
    val_set = set(val)
    near = {(book, page + d) for book, page in val for d in range(-margin, margin + 1)}
    train = [k for k in keys if k not in val_set and k not in near]
    return train, val


def mine_negatives(keys: list[Key], texts: list[str], min_gap: int = 2, top_k: int = 8) -> dict[Key, list[Key]]:
    """Hard negatives por similaridade TF-IDF do texto: paginas do mesmo assunto que NAO sao a pagina-ouro.

    Paginas a menos de `min_gap` da pagina-ouro no mesmo livro ficam de fora: costumam ser a continuacao do
    mesmo stat block/regra e seriam falsos negativos.
    """
    n = len(keys)
    tokens = [re.findall(r"\w{3,}", t.lower()) for t in texts]
    doc_freq = Counter()
    for toks in tokens:
        doc_freq.update(set(toks))
    vocab = {w: i for i, w in enumerate(w for w, c in doc_freq.items() if 2 <= c <= 0.5 * n)}

    matrix = np.zeros((n, len(vocab)), dtype=np.float32)
    for i, toks in enumerate(tokens):
        for word, count in Counter(toks).items():
            j = vocab.get(word)
            if j is not None:
                matrix[i, j] = (1 + math.log(count)) * math.log(n / doc_freq[word])
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
    sim = matrix @ matrix.T

    books = np.array([k[0] for k in keys])
    pages = np.array([k[1] for k in keys])
    negatives = {}
    for i, (book, page) in enumerate(keys):
        row = np.where((books == book) & (np.abs(pages - page) < min_gap), -np.inf, sim[i])
        order = np.argsort(-row)[:top_k]
        negatives[(book, page)] = [keys[j] for j in order if np.isfinite(row[j])]
    return negatives
