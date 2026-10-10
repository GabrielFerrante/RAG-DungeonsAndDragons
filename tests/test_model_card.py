"""Testes (CPU) do model card gerado ao fim do treino."""
import json

from src.train.model_card import data_stats, training_summary, write_model_cards

CONFIG = {
    "base": "org/base-model", "eval_every": "50", "epochs": "1", "max_visual_tokens": "768", "min_chars": "300",
    "val_block": "20", "neg_per_query": "2", "neg_pool": "8", "temperature": "0.02", "rank": "32", "alpha": "32",
    "dropout": "0.1", "lr": "5e-05", "grad_accum": "16", "seed": "0",
}
BASELINE = {"recall@1": 0.5, "recall@5": 0.8, "recall@10": 0.9, "mrr": 0.6, "ndcg@5": 0.7, "n": 10}
TUNED = {"recall@1": 0.7, "recall@5": 0.9, "recall@10": 1.0, "mrr": 0.8, "ndcg@5": 0.85, "n": 10}
ENV = {"gpu": "GPU de teste", "versions": {"peft": "1.0"}}


def make_run(tmp_path, best=True):
    run = tmp_path / "run"
    for name in ("best", "final"):
        (run / name).mkdir(parents=True)
        (run / name / "README.md").write_text("# Model Card for Model ID\n[More Information Needed]", encoding="utf-8")
    summary = {"baseline": BASELINE, "best": {"step": 2, **TUNED} if best else None, "final": TUNED}
    log = [{"event": "step", "step": s, "of": 3, "elapsed_h": h, "vram_gb": 5.0 + s / 10} for s, h in [(1, 1.0), (2, 2.0), (3, 3.0)]]
    (run / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run / "log.jsonl").write_text("\n".join(json.dumps(r) for r in log), encoding="utf-8")
    return run


def test_training_summary_sums_sessions_after_a_resume():
    rows = [{"event": "baseline"}] + [
        {"event": "step", "step": s, "of": 4, "elapsed_h": h, "vram_gb": v}
        for s, h, v in [(1, 1.0, 4.0), (2, 2.0, 4.5), (3, 0.5, 4.2), (4, 1.0, 4.1)]  # retomou depois do passo 2
    ]
    assert training_summary(rows) == {"steps": 4, "of": 4, "hours": 3.0, "vram_gb": 4.5}


def test_model_cards_replace_the_generic_peft_card_with_the_run_numbers(tmp_path):
    run = make_run(tmp_path)
    stats = data_stats({1: 0}, [(("a", 1), "q1"), (("a", 1), "q2")], [(("a", 2), "q3")], [("a", 2), ("a", 5)])
    assert stats["train_questions"] == 2 and stats["train_pages"] == 1 and stats["pool_pages"] == 2

    written = write_model_cards(run, stats, ENV)
    assert written == [run / "best" / "README.md", run / "final" / "README.md"]
    best = written[0].read_text(encoding="utf-8")
    assert best.startswith("---\nbase_model: org/base-model\nlibrary_name: peft\n")  # front matter que o Hub/PEFT leem
    assert "[More Information Needed]" not in best
    assert "passo 2 de 3" in best and "| **este adapter (best)** | 70.0% | 90.0% | 100.0% | 0.800 | 0.850 |" in best
    assert "+20.0 pp" in best and "GPU de teste" in best and "3.0 h" in best
    assert "passo 3 de 3" in written[1].read_text(encoding="utf-8")


def test_no_best_card_when_there_was_no_periodic_evaluation(tmp_path):
    run = make_run(tmp_path, best=False)
    stats = data_stats({}, [(("a", 1), "q")], [(("a", 2), "q")], [("a", 2)])
    assert write_model_cards(run, stats, ENV) == [run / "final" / "README.md"]
    assert "[More Information Needed]" in (run / "best" / "README.md").read_text(encoding="utf-8")  # intocado
