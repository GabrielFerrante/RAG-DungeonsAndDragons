"""Testes (CPU) da avaliacao T2I/I2T: scores, ranks, opcoes do treino e coerencia do split com o treino."""
import json
from argparse import Namespace

import numpy as np
import torch

from dnd_rag.eval.run_retrieval_eval import (
    TRAIN_OPTIONS,
    i2t_ranks,
    resolve_train_options,
    score_matrix,
    select_eval_set,
    summarize,
    t2i_ranks,
)
from dnd_rag.train.data import load_pages, load_queries
from dnd_rag.train.train_retriever import build_data


def test_score_matrix_matches_naive_maxsim_with_chunking():
    torch.manual_seed(0)
    q = [torch.nn.functional.normalize(torch.randn(n, 8), dim=-1) for n in (3, 5, 4)]  # tamanhos diferentes
    p = [torch.nn.functional.normalize(torch.randn(n, 8), dim=-1) for n in (6, 9)]
    scores = score_matrix(q, p, device="cpu", chunk=2)  # chunk < Nq: exercita o preenchimento por blocos
    for i, qi in enumerate(q):
        for j, pj in enumerate(p):
            expected = (qi @ pj.T).max(-1).values.sum() / qi.shape[0]  # soma dos maximos / tamanho da pergunta
            assert abs(scores[i, j] - float(expected)) < 1e-5


def test_t2i_and_i2t_ranks_on_a_hand_made_matrix():
    #          pag0  pag1  pag2
    scores = np.array([
        [0.9, 0.1, 0.1],  # q0 -> pag0: 1o
        [0.2, 0.3, 0.8],  # q1 -> pag0: pior (3o)
        [0.1, 0.7, 0.2],  # q2 -> pag1: 1o
        [0.1, 0.1, 0.6],  # q3 -> pag2: 1o
    ])
    query_page = np.array([0, 0, 1, 2])
    assert t2i_ranks(scores, query_page) == [1, 3, 1, 1]
    # pag0 tem q0 e q1: vale a melhor (q0, 1o); pag2 so tem q3 (0.6), superada por q1 (0.8): 2o
    assert i2t_ranks(scores, query_page) == [1, 1, 2]
    m = summarize(t2i_ranks(scores, query_page), (1, 5))
    assert m["recall@1"] == 0.75 and m["recall@5"] == 1.0 and m["median_rank"] == 1.0


def test_resolve_train_options_prefers_cli_then_train_config_then_default(tmp_path):
    run = tmp_path / "retriever-lora"
    (run / "best").mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"val_fraction": "0.25", "val_block": "5", "seed": "7"}), encoding="utf-8")
    args = Namespace(adapter=run / "best", **{n: None for n in TRAIN_OPTIONS})
    args.val_block = 9  # veio da linha de comando
    origin = resolve_train_options(args)
    assert (args.val_fraction, args.val_block, args.seed) == (0.25, 9, 7)
    assert args.min_chars == 300  # nao estava no config: padrao
    assert origin["val_fraction"] == "config do treino" and origin["val_block"] == "linha de comando" and origin["min_chars"] == "padrao"


def test_eval_split_is_identical_to_the_training_validation_set(tmp_path):
    """Se este teste quebrar, a avaliacao pode estar medindo em paginas que o treino viu (ou nao bate com o log do treino)."""
    data = tmp_path
    (data / "text").mkdir()
    (data / "synth").mkdir()
    manifest, queries = [], []
    for book in ("livroa", "livrob"):
        for page in range(1, 61):
            text_path = data / "text" / f"{book}-{page}.txt"
            text_path.write_text(f"assunto{page} tema{page % 7} palavra{page % 5} conteudo " * 20, encoding="utf-8")
            manifest.append({"book": book, "book_name": book, "page": page, "image": f"pages/{book}/{page}.png",
                             "text": f"text/{book}-{page}.txt", "n_chars": 600})
            queries.append({"book": book, "page": page, "queries": [f"pergunta {i} sobre {book} {page}?" for i in range(3)]})
    (data / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in manifest), encoding="utf-8")
    (data / "synth" / "queries.jsonl").write_text("\n".join(json.dumps(r) for r in queries), encoding="utf-8")

    common = dict(data=data, min_chars=300, val_fraction=0.2, val_block=10, seed=3)
    train_args = Namespace(**common, neg_pool=8, val_queries=40, distractors=15)
    _, _, val, _, pool = build_data(train_args)

    eval_args = Namespace(**common, split="val", max_queries=40, distractors=15)
    items, eval_pool = select_eval_set(load_pages(data), load_queries(data), eval_args)
    assert items == val and eval_pool == pool

    train_split = select_eval_set(load_pages(data), load_queries(data), Namespace(**common, split="train", max_queries=0, distractors=0))[0]
    assert not {k for k, _ in train_split} & {k for k, _ in items}  # treino e validacao nunca se misturam
