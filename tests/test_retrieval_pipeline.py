"""Testes das partes puras do pipeline de fine-tuning (sem GPU, sem modelo)."""
import numpy as np

from dnd_rag.eval.metrics import rank_of, retrieval_metrics
from dnd_rag.train.data import mine_negatives, split_pages
from dnd_rag.train.make_synthetic import parse_questions
from dnd_rag.train.train_retriever import lr_at


def test_rank_of_counts_ties_against_gold():
    assert rank_of(np.array([0.1, 0.9, 0.5]), gold=1) == 1
    assert rank_of(np.array([0.1, 0.9, 0.5]), gold=2) == 2
    assert rank_of(np.array([0.5, 0.5, 0.5]), gold=0) == 3  # empate: pessimista


def test_retrieval_metrics():
    m = retrieval_metrics([1, 2, 6, 11])
    assert m["recall@1"] == 0.25 and m["recall@5"] == 0.5 and m["recall@10"] == 0.75
    assert abs(m["mrr"] - (1 + 1 / 2 + 1 / 6 + 1 / 11) / 4) < 1e-9
    assert abs(m["ndcg@5"] - (1 + 1 / np.log2(3)) / 4) < 1e-9


def test_split_has_no_leakage_between_train_and_val():
    keys = [(book, p) for book in ("a", "b") for p in range(1, 101)]
    train, val = split_pages(keys, val_fraction=0.2, block=10, margin=2, seed=1)
    assert val and train and not set(train) & set(val)
    for book, page in val:  # nenhuma pagina de treino colada num bloco de validacao
        for d in (-2, -1, 1, 2):
            assert (book, page + d) not in set(train)
    assert train == split_pages(keys, 0.2, 10, 2, seed=1)[0]  # deterministico


def test_negatives_skip_adjacent_pages_and_prefer_similar_text():
    keys = [("a", 1), ("a", 2), ("a", 3), ("a", 4)] + [("b", i) for i in range(1, 9)]
    texts = [
        "dragao vermelho sopro fogo escamas tesouro",
        "dragao vermelho sopro fogo escamas tesouro",  # vizinha identica: nao pode virar negativo
        "elfo arqueiro floresta arco flecha",
        "dragao vermelho sopro fogo escamas cavernas",  # mesmo assunto, longe da pagina 1
    ] + [f"assunto{i} tema{i} variado{i}" for i in range(1, 9)]  # enchimento: aumenta o corpus (filtro df <= 50%)
    negs = mine_negatives(keys, texts, min_gap=2, top_k=3)
    assert ("a", 1) not in negs[("a", 1)] and ("a", 2) not in negs[("a", 1)]
    assert negs[("a", 1)][0] == ("a", 4)  # o mais parecido fora da janela adjacente


def test_parse_questions_filters_bad_items():
    raw = '{"perguntas": ["Qual a CA do Nalfeshnee?", "curta?", "O que diz esta pagina sobre o dragao?", "Qual a CA do Nalfeshnee?", "Sem interrogacao"]}'
    assert parse_questions(raw) == ["Qual a CA do Nalfeshnee?"]
    assert parse_questions('texto solto\n1. Quantos PV tem um dragao vermelho adulto?\n') == ["Quantos PV tem um dragao vermelho adulto?"]


def test_synthetic_prompt_formats_and_keeps_json_example():
    from dnd_rag.train.make_synthetic import PROMPT

    text = PROMPT.format(n=5, text="NALFESHNEE Corruptor Grande")
    assert "5 perguntas" in text and "NALFESHNEE" in text
    assert '{"perguntas": ["...", "..."]}' in text  # chaves do exemplo JSON sobreviveram ao .format


def test_fit_visual_tokens_respects_budget_and_never_upscales():
    from PIL import Image

    from dnd_rag.generate.vlm import TOKEN_PIXELS, fit_visual_tokens

    big = Image.new("RGB", (1191, 1684))
    small = fit_visual_tokens(big, 768)
    assert small.width * small.height <= 768 * TOKEN_PIXELS * 1.01
    assert abs(small.width / small.height - big.width / big.height) < 0.02  # mantem a proporcao
    tiny = Image.new("RGB", (200, 300))
    assert fit_visual_tokens(tiny, 768) is tiny


def test_lr_schedule_warms_up_then_decays_to_zero():
    total, base = 100, 1e-4
    assert lr_at(0, total, base) < lr_at(1, total, base) <= base
    assert lr_at(50, total, base) < base
    assert lr_at(total, total, base) == 0.0
