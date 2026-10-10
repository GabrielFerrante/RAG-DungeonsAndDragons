"""Model card (README.md) dos adapters que o treino gera: best/ e final/.

O PEFT grava um README generico ("[More Information Needed]") a cada save_pretrained. Ao fim do treino este
modulo o substitui por um card com os dados, hiperparametros e metricas reais do run (config.json,
log.jsonl e summary.json). Tambem roda sozinho sobre um run ja concluido:

Uso:  python main.py model-card [--run outputs\\retriever-lora] [--data data]
"""
import argparse
import json
from argparse import Namespace
from importlib.metadata import version
from pathlib import Path

from src.utils.helpers import CONFIG, ROOT, read_jsonl

VARIANTS = ("best", "final")
RECALLS = ("recall@1", "recall@5", "recall@10")


# ----------------------------------------------------------------------------- dados do run

def data_stats(pages, train, val, pool) -> dict:
    """Contagens do conjunto de dados, a partir do retorno de train_retriever.build_data."""
    return {
        "manifest_pages": len(pages),
        "train_questions": len(train), "train_pages": len({k for k, _ in train}),
        "val_questions": len(val), "val_pages": len({k for k, _ in val}),
        "pool_pages": len(pool),
    }


def runtime_env() -> dict:
    import torch

    return {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "versions": {name: version(name) for name in ("peft", "transformers", "torch", "bitsandbytes")},
    }


def training_summary(log: list[dict]) -> dict:
    """Passos e tempo do log. O elapsed_h recomeca a cada retomada: soma o fim de cada sessao."""
    steps = [r for r in log if r["event"] == "step"]
    hours, previous = 0.0, 0.0
    for row in steps:
        if row["elapsed_h"] < previous:
            hours += previous
        previous = row["elapsed_h"]
    return {"steps": steps[-1]["step"], "of": steps[-1]["of"], "hours": hours + previous,
            "vram_gb": max(r["vram_gb"] for r in steps)}


# ----------------------------------------------------------------------------- texto do card

def metric_row(label: str, m: dict, bold: bool = False) -> str:
    cells = [f"{100 * m[k]:.1f}%" for k in RECALLS] + [f"{m['mrr']:.3f}", f"{m['ndcg@5']:.3f}"]
    return "| " + " | ".join([f"**{label}**" if bold else label, *cells]) + " |"


def delta_row(base: dict, tuned: dict) -> str:
    cells = [f"{100 * (tuned[k] - base[k]):+.1f} pp" for k in RECALLS]
    cells += [f"{tuned['mrr'] - base['mrr']:+.3f}", f"{tuned['ndcg@5'] - base['ndcg@5']:+.3f}"]
    return "| diferença | " + " | ".join(cells) + " |"


def render_card(variant: str, run_dir: Path, config: dict, summary: dict, log: list[dict], stats: dict, env: dict) -> str:
    train = training_summary(log)
    steps = {"best": (summary["best"] or {}).get("step"), "final": train["steps"]}
    baseline, tuned = summary["baseline"], summary[variant]
    path = (run_dir / variant).resolve()
    shown_path = (path.relative_to(ROOT) if ROOT in path.parents else path).as_posix()
    base_model = config["base"]
    distractors = stats["pool_pages"] - stats["val_pages"]
    why = {
        "best": f"checkpoint do passo {steps['best']} de {train['of']}, o de maior recall@5 (desempate por MRR) "
                f"entre as avaliações feitas a cada {config['eval_every']} passos",
        "final": f"checkpoint do passo {steps['final']} de {train['of']}, ao fim do treino ({config['epochs']} época)",
    }[variant]
    other = "final" if variant == "best" else "best"
    has_other = bool(summary.get(other))  # sem best/ quando eval_every=0
    other_row = f"\n{metric_row(f'outra variante ({other})', summary[other])}" if has_other else ""
    side_by_side = " Os dois adapters do treino ficam lado a lado (`best/` e `final/`)." if has_other else ""
    gpu = env["gpu"] or "não registrada"
    versions = ", ".join(f"{name} {v}" for name, v in env["versions"].items())

    return f"""---
base_model: {base_model}
library_name: peft
language:
- pt
tags:
- base_model:adapter:{base_model}
- lora
- colqwen3
- retrieval
- transformers
---

# Retriever visual de D&D 5E (PT-BR) — adapter LoRA `{variant}`

Adapter LoRA (QLoRA: base em 4-bit NF4) sobre o [{base_model}](https://huggingface.co/{base_model}), um retriever
visual multi-vetor no estilo ColBERT/ColPali. Dada uma pergunta em português, ele ranqueia **imagens de páginas**
dos livros de D&D 5E (Livro do Jogador, Guia do Mestre e Livro dos Monstros). Faz parte do RAG local deste repositório.

Esta variante é o {why}.{side_by_side}

## Resultados

Validação: {stats['val_questions']} perguntas sobre {stats['val_pages']} páginas que nunca entram no treino, buscando
a página correta (texto → imagem) entre {stats['pool_pages']} páginas ({stats['val_pages']} de validação +
{distractors} de treino como distratoras). Relevância binária: só a página de origem da pergunta conta, e empates
contam contra o modelo.

| | R@1 | R@5 | R@10 | MRR | nDCG@5 |
|---|---|---|---|---|---|
{metric_row("modelo original (sem adapter)", baseline)}
{metric_row("este adapter (" + variant + ")", tuned, bold=True)}
{delta_row(baseline, tuned)}{other_row}

Ressalvas:

- As perguntas são **sintéticas** (geradas por um VLM a partir de cada página), não escritas por jogadores. Os números
  dizem o quanto o adapter se ajustou a esse tipo de pergunta; não foram medidos com perguntas humanas.
- O `best` foi escolhido por esta mesma validação, então o número dele é otimista. O `final` não passou por seleção.
- O pool é pequeno ({stats['pool_pages']} páginas, de um total de {stats['manifest_pages']} no manifesto). Os valores
  absolutos não se comparam a uma busca sobre os livros inteiros; a comparação com o modelo original, sim.
- Só a direção texto → imagem foi medida neste treino. A avaliação completa (inclusive imagem → texto) roda com
  `python main.py eval --adapter {shown_path} --compare`.

## Como usar

O adapter precisa do modelo base e do carregamento do repositório: a cabeça de projeção (`embedding_proj_layer`)
foi treinada em fp32, e `load_adapter` recarrega esses pesos sem arredondá-los para bf16.

```python
from argparse import Namespace
from pathlib import Path

from src.embeddings.embedder import embed_pages, embed_query, load_retriever
from src.eval.run_retrieval_eval import load_adapter

model, processor = load_retriever(Namespace(base="{base_model}", max_visual_tokens={config['max_visual_tokens']}))
model = load_adapter(model, Path("{shown_path}"))
```

## Dados de treino

- {stats['train_questions']} perguntas em {stats['train_pages']} páginas (páginas com pelo menos {config['min_chars']} caracteres
  de texto extraído; capas e páginas em branco ficam de fora).
- Perguntas geradas pelo VLM local do projeto (`{CONFIG['models']['vlm']}`, por padrão) a partir da imagem e do texto
  da página, em PT-BR, sem referências do tipo "nesta página".
- Divisão treino/validação por **blocos de {config['val_block']} páginas** consecutivas (proxy de capítulo), com margem de
  2 páginas ao redor de cada bloco de validação, para não vazar conteúdo entre os dois lados.
- Negativos difíceis por similaridade TF-IDF do texto: {config['neg_per_query']} por pergunta, sorteados entre os
  {config['neg_pool']} mais parecidos.
- Os livros são conteúdo licenciado: nem as páginas nem as perguntas são distribuídas com o adapter.

## Treinamento

- Perda InfoNCE sobre MaxSim (normalizado pelo tamanho da pergunta): 1 página positiva e {config['neg_per_query']}
  negativas por pergunta, temperatura {config['temperature']}.
- LoRA r={config['rank']}, alpha={config['alpha']}, dropout {config['dropout']}, nas projeções q/k/v/o e gate/up/down do
  modelo de linguagem; `embedding_proj_layer` treinada por inteiro (fp32).
- Base em NF4 com dupla quantização e cálculo em bf16; torre visual e cabeça de projeção em bf16; gradient checkpointing.
  Imagens limitadas a {config['max_visual_tokens']} tokens visuais por página.
- PagedAdamW8bit, lr {config['lr']} com aquecimento linear de 2.5% e queda linear até zero. Batch efetivo de
  {config['grad_accum']} perguntas (acúmulo de gradiente), {config['epochs']} época, {train['of']} passos, seed {config['seed']}.
- Duração registrada no log: ≈ {train['hours']:.1f} h; pico de VRAM {train['vram_gb']:.2f} GB (GPU: {gpu}).

## Limitações

- Só cobre os três livros de D&D 5E em PT-BR usados no treino. Outros livros, edições ou idiomas não foram avaliados.
- É só a etapa de busca: o adapter não gera respostas.
- A licença deste adapter não foi definida. A base ({base_model}) é Apache-2.0.

## Versões

PEFT, transformers, torch e bitsandbytes usados: {versions}.
"""


# ----------------------------------------------------------------------------- escrita

def write_model_cards(run_dir: Path, stats: dict, env: dict) -> list[Path]:
    """Reescreve o README.md de best/ e final/ do run. Le config.json, log.jsonl e summary.json do proprio run."""
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8-sig"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8-sig"))
    log = read_jsonl(run_dir / "log.jsonl")
    written = []
    for variant in VARIANTS:
        if not (run_dir / variant).is_dir() or not summary.get(variant):
            continue  # ex.: sem avaliacao periodica (--eval-every 0) nao existe best/
        card = run_dir / variant / "README.md"
        card.write_text(render_card(variant, run_dir, config, summary, log, stats, env), encoding="utf-8")
        written.append(card)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs" / "retriever-lora")
    parser.add_argument("--data", type=Path, help="pasta de dados do treino (padrao: a do config.json do run)")
    args = parser.parse_args()

    from src.train.train_retriever import build_data  # import aqui: train_retriever importa este modulo

    config = json.loads((args.run / "config.json").read_text(encoding="utf-8-sig"))
    train_args = Namespace(
        data=args.data or Path(config["data"]), min_chars=int(config["min_chars"]), val_fraction=float(config["val_fraction"]),
        val_block=int(config["val_block"]), seed=int(config["seed"]), neg_pool=int(config["neg_pool"]),
        val_queries=int(config["val_queries"]), distractors=int(config["distractors"]),
    )
    pages, train, val, _, pool = build_data(train_args)
    for card in write_model_cards(args.run, data_stats(pages, train, val, pool), runtime_env()):
        print(card)


if __name__ == "__main__":
    main()
