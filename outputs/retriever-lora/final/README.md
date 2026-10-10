---
base_model: TomoroAI/tomoro-colqwen3-embed-4b
library_name: peft
language:
- pt
tags:
- base_model:adapter:TomoroAI/tomoro-colqwen3-embed-4b
- lora
- colqwen3
- retrieval
- transformers
---

# Retriever visual de D&D 5E (PT-BR) — adapter LoRA `final`

Adapter LoRA (QLoRA: base em 4-bit NF4) sobre o [TomoroAI/tomoro-colqwen3-embed-4b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-4b), um retriever
visual multi-vetor no estilo ColBERT/ColPali. Dada uma pergunta em português, ele ranqueia **imagens de páginas**
dos livros de D&D 5E (Livro do Jogador, Guia do Mestre e Livro dos Monstros). Faz parte do RAG local deste repositório.

Esta variante é o checkpoint do passo 244 de 244, ao fim do treino (1 época). Os dois adapters do treino ficam lado a lado (`best/` e `final/`).

## Resultados

Validação: 200 perguntas sobre 88 páginas que nunca entram no treino, buscando
a página correta (texto → imagem) entre 188 páginas (88 de validação +
100 de treino como distratoras). Relevância binária: só a página de origem da pergunta conta, e empates
contam contra o modelo.

| | R@1 | R@5 | R@10 | MRR | nDCG@5 |
|---|---|---|---|---|---|
| modelo original (sem adapter) | 74.0% | 93.5% | 96.0% | 0.821 | 0.845 |
| **este adapter (final)** | 83.0% | 96.5% | 99.5% | 0.895 | 0.910 |
| diferença | +9.0 pp | +3.0 pp | +3.5 pp | +0.074 | +0.065 |
| outra variante (best) | 83.5% | 96.5% | 99.0% | 0.897 | 0.911 |

Ressalvas:

- As perguntas são **sintéticas** (geradas por um VLM a partir de cada página), não escritas por jogadores. Os números
  dizem o quanto o adapter se ajustou a esse tipo de pergunta; não foram medidos com perguntas humanas.
- O `best` foi escolhido por esta mesma validação, então o número dele é otimista. O `final` não passou por seleção.
- O pool é pequeno (188 páginas, de um total de 992 no manifesto). Os valores
  absolutos não se comparam a uma busca sobre os livros inteiros; a comparação com o modelo original, sim.
- Só a direção texto → imagem foi medida neste treino. A avaliação completa (inclusive imagem → texto) roda com
  `python main.py eval --adapter outputs/retriever-lora/final --compare`.

## Como usar

O adapter precisa do modelo base e do carregamento do repositório: a cabeça de projeção (`embedding_proj_layer`)
foi treinada em fp32, e `load_adapter` recarrega esses pesos sem arredondá-los para bf16.

```python
from argparse import Namespace
from pathlib import Path

from src.embeddings.embedder import embed_pages, embed_query, load_retriever
from src.eval.run_retrieval_eval import load_adapter

model, processor = load_retriever(Namespace(base="TomoroAI/tomoro-colqwen3-embed-4b", max_visual_tokens=768))
model = load_adapter(model, Path("outputs/retriever-lora/final"))
```

## Dados de treino

- 3914 perguntas em 819 páginas (páginas com pelo menos 300 caracteres
  de texto extraído; capas e páginas em branco ficam de fora).
- Perguntas geradas pelo VLM local do projeto (`Qwen/Qwen3-VL-4B-Instruct`, por padrão) a partir da imagem e do texto
  da página, em PT-BR, sem referências do tipo "nesta página".
- Divisão treino/validação por **blocos de 20 páginas** consecutivas (proxy de capítulo), com margem de
  2 páginas ao redor de cada bloco de validação, para não vazar conteúdo entre os dois lados.
- Negativos difíceis por similaridade TF-IDF do texto: 2 por pergunta, sorteados entre os
  8 mais parecidos.
- Os livros são conteúdo licenciado: nem as páginas nem as perguntas são distribuídas com o adapter.

## Treinamento

- Perda InfoNCE sobre MaxSim (normalizado pelo tamanho da pergunta): 1 página positiva e 2
  negativas por pergunta, temperatura 0.02.
- LoRA r=32, alpha=32, dropout 0.1, nas projeções q/k/v/o e gate/up/down do
  modelo de linguagem; `embedding_proj_layer` treinada por inteiro (fp32).
- Base em NF4 com dupla quantização e cálculo em bf16; torre visual e cabeça de projeção em bf16; gradient checkpointing.
  Imagens limitadas a 768 tokens visuais por página.
- PagedAdamW8bit, lr 5e-05 com aquecimento linear de 2.5% e queda linear até zero. Batch efetivo de
  16 perguntas (acúmulo de gradiente), 1 época, 244 passos, seed 0.
- Duração registrada no log: ≈ 8.6 h; pico de VRAM 5.29 GB (GPU: NVIDIA GeForce RTX 3060).

## Limitações

- Só cobre os três livros de D&D 5E em PT-BR usados no treino. Outros livros, edições ou idiomas não foram avaliados.
- É só a etapa de busca: o adapter não gera respostas.
- A licença deste adapter não foi definida. A base (TomoroAI/tomoro-colqwen3-embed-4b) é Apache-2.0.

## Versões

PEFT, transformers, torch e bitsandbytes usados: peft 0.20.0, transformers 5.17.0, torch 2.13.0+cu130, bitsandbytes 0.50.2.
