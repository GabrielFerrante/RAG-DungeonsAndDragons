"""Grafico da perda de treino (log.jsonl -> loss.png), reaproveitavel para qualquer run do retriever
(T2T, src/train/train_retriever.py, ou I2T, src/train/train_i2t.py — o formato do log e o mesmo).

Plota a perda por passo de otimizacao: a serie bruta (mais ruidosa, já que cada passo e so alguns
exemplos por causa do acumulo de gradiente) por tras, mais clara, e uma media movel por cima, mais
legivel. Marca o passo do melhor checkpoint (evento "best" em summary.json), se houver.

Uso:
  python main.py plot --run outputs\\retriever-lora
  python main.py plot --run outputs\\retriever-lora --run outputs\\retriever-i2t-lora  # os dois de uma vez
  python main.py plot --run outputs\\retriever-i2t-lora --label "I2T (recorte de pagina)" --smooth 15
"""
import argparse
import json
from pathlib import Path

from src.utils.helpers import ROOT, read_jsonl

# src/eval/plot_training.py:52: referencia o primeiro/segundo slot categorico de references/palette.md
# (skill dataviz) — ordem fixa, nao ciclada; cada run usa o slot da sua posicao em --run.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # azul, laranja, agua, amarelo
SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e2dc"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, type=Path, help="pasta do run (ex.: outputs/retriever-lora); repita pra varios")
    p.add_argument("--label", action="append", default=[], help="legenda/titulo por --run, na mesma ordem (padrao: nome da pasta)")
    p.add_argument("--smooth", type=int, default=10, help="janela da media movel, em passos")
    p.add_argument("--out-name", default="loss.png", help="nome do arquivo, salvo dentro de cada --run")
    return p.parse_args()


def rolling_mean(values: list[float], window: int) -> list[float]:
    out, acc = [], 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        out.append(acc / min(i + 1, window))
    return out


def plot_run(run_dir: Path, label: str, color: str, smooth: int, out_name: str) -> Path:
    log_path = run_dir / "log.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(f"{log_path} nao existe (o treino desse run ainda nao gerou log)")

    rows = [r for r in read_jsonl(log_path) if r["event"] == "step"]
    if not rows:
        raise ValueError(f"{log_path} nao tem nenhum evento 'step' ainda")
    steps = [r["step"] for r in rows]
    loss = [r["loss"] for r in rows]
    smoothed = rolling_mean(loss, smooth)

    best_step = None
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        best = json.loads(summary_path.read_text(encoding="utf-8-sig")).get("best")
        best_step = best["step"] if best else None

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(steps, loss, color=color, alpha=0.25, linewidth=1, zorder=2)  # serie bruta, ao fundo
    ax.plot(steps, smoothed, color=color, linewidth=2, zorder=3, label=f"media movel ({smooth} passos)")
    if best_step is not None:  # linha + legenda (nao uma anotacao solta: evita colidir com a legenda)
        ax.axvline(best_step, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 3)), zorder=1, label=f"melhor checkpoint (passo {best_step})")

    ax.set_title(f"Perda de treino — {label}", fontsize=13, color=TEXT_PRIMARY, loc="left", pad=12)
    ax.set_xlabel("Passo de otimização", fontsize=10, color=TEXT_SECONDARY)
    ax.set_ylabel("Perda (InfoNCE sobre MaxSim)", fontsize=10, color=TEXT_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="best")
    fig.tight_layout()

    out_path = run_dir / out_name
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    labels = args.label + [run.name for run in args.run[len(args.label):]]
    for i, (run_dir, label) in enumerate(zip(args.run, labels)):
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        out_path = plot_run(run_dir, label, color, args.smooth, args.out_name)
        print(f"{run_dir} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
