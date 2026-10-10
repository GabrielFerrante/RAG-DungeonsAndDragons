# Projeto RAG para Dungeons And Dragons 5E

Criação de um chat para perguntas e respostas sobre D&D 5E com base nos guias oficiais em PT-BR.

RAG multimodal e 100% local. O retriever (ColQwen3, multi-vetor) lê a **imagem** de cada página dos livros, e um VLM (Qwen3-VL) gera as perguntas sintéticas usadas no fine-tuning LoRA do retriever.

## Estrutura

```
RAG-DungeonsAndDragons/
├── README.md
├── requirements.txt       # dependências + ordem de instalação
├── environment.yml        # env conda `ragded`
├── config.yaml            # modelos, tamanho mínimo de chunk, nomes dos livros
├── .env                   # segredos (não versionado; modelo em .env.example)
├── .gitignore
├── main.py                # ponto de entrada: python main.py <ingest|synthetic|train|eval|model-card|ask>
├── src/
│   ├── ingestion/         # loader.py: PDFs -> PNG + texto por página + manifesto
│   ├── chunking/          # chunker.py: o chunk é a página; ajusta a imagem ao orçamento de tokens visuais
│   ├── embeddings/        # embedder.py: embeddings multi-vetor (ColQwen3, 4-bit) de páginas e perguntas
│   ├── vectordb/          # vector_store.py: índice vetorial (Qdrant) — avaliado e descartado por ora (ver "Como é feita a inferência")
│   ├── retrieval/         # retriever.py: busca por similaridade (MaxSim); ask.py: CLI de pergunta -> resposta
│   ├── front/             # interface (a implementar)
│   ├── prompts/           # prompt_templates.py: templates de prompt
│   ├── llm/               # llm_client.py: Qwen3-VL em 4-bit
│   ├── api/               # routes.py: endpoints FastAPI — a implementar
│   ├── utils/             # helpers.py: raiz do projeto, config.yaml, JSONL, controle de energia
│   ├── train/             # perguntas sintéticas, dados, fine-tuning LoRA do retriever e model card dos adapters
│   └── eval/              # métricas e avaliação de retrieval (T2I / I2T)
├── tests/                 # testes unitários (CPU, sem modelo)
├── scripts/               # run_finetune.ps1 (pipeline completo), vram_smoke_test.py
├── logs/                  # logs de execução
├── ArquivosEFontes/       # PDFs dos livros (não versionados)
├── data/                  # páginas, texto e perguntas geradas (não versionado)
└── outputs/               # adapters, métricas e resultados (não versionado)
```

## Como rodar

Sempre da raiz do repositório, no env `ragded` (siga a ordem de instalação descrita em `requirements.txt`).

```powershell
python main.py ingest               # 1) PDFs -> data/
python main.py synthetic            # 2) perguntas sintéticas (horas, GPU; retoma de onde parou)
python main.py train --dry-run      # 3) confere os dados; sem --dry-run treina
python main.py eval --adapter outputs\retriever-lora\best --compare
python main.py model-card           # refaz o README.md de best/ e final/ de um treino já concluído
python main.py ask                  # 4) pergunta em pt-BR -> resposta com citação (retriever + gerador)
python -m pytest                    # testes
```

Ao fim do treino, o `README.md` de `best/` e `final/` (o model card de cada adapter) é preenchido sozinho com os dados, hiperparâmetros e métricas do run; o `model-card` só é preciso para refazê-lo sobre um treino antigo.

O pipeline completo, retomável em cada etapa, roda com `.\scripts\run_finetune.ps1` (`-Smoke` para um teste rápido ponta a ponta).

## Como foi feito o treino

Só o **retriever** (o modelo que rankeia páginas) passou por fine-tuning. O gerador (seção seguinte) é usado do jeito que foi publicado, sem ajuste.

**Base:** [TomoroAI/tomoro-colqwen3-embed-4b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-4b), um retriever visual multi-vetor (ColBERT/ColPali) sobre Qwen3-VL-4B, já treinado e multilíngue — o fine-tuning *adapta* esse modelo aos três livros, não treina do zero.

**Dados de treino** (`python main.py synthetic`): o próprio Qwen3-VL-4B local lê cada página (imagem + texto extraído) e escreve de 3 a 8 perguntas em pt-BR que só essa página responde, sem referências do tipo "nesta página". Do total de páginas com conteúdo aproveitável (≥300 caracteres de texto), isso deu **3.914 perguntas em 819 páginas**. As perguntas de validação (200, sobre 88 páginas) nunca entram no treino — a divisão é por **blocos de 20 páginas consecutivas** (um proxy de capítulo), com 2 páginas de margem de cada lado do bloco de validação, pra um stat block que continua na página seguinte não vazar de um lado pro outro.

**Negativos difíceis:** para cada pergunta, 2 páginas "parecidas mas erradas" por similaridade TF-IDF do texto (excluindo páginas vizinhas, que tendem a ser continuação do mesmo assunto) — sem isso, a perda contrastiva fica fácil demais e o modelo não aprende a discriminar páginas do mesmo tema.

**Fine-tuning:** LoRA (QLoRA: base em 4-bit NF4) nas projeções q/k/v/o e gate/up/down do modelo de linguagem (r=32, alpha=32, dropout 0.1), com a camada de projeção final (`embedding_proj_layer`) treinada por inteiro, em fp32. Perda InfoNCE sobre MaxSim (1 página positiva + 2 negativas por pergunta, temperatura 0,02). PagedAdamW8bit, lr 5e-5 com aquecimento linear e queda linear até zero, batch efetivo de 16 perguntas (acúmulo de gradiente), 1 época (244 passos). Checkpoint a cada 50 passos com retomada automática, avaliação a cada 50 passos no split de validação, e o melhor passo (`best/`, por recall@5) salvo à parte do último (`final/`).

**Hardware:** uma RTX 3060 de 8 GB — pico de VRAM de 5,29 GB. Duração: ≈ 8,55 h.

**Resultado** (retrieval texto → imagem, 200 perguntas, pool de 188 páginas):

| | R@1 | R@5 | R@10 | MRR | nDCG@5 |
|---|---|---|---|---|---|
| modelo original (sem adapter) | 74,0% | 93,5% | 96,0% | 0,821 | 0,845 |
| **com o adapter (`best`)** | 83,5% | 96,5% | 99,0% | 0,897 | 0,911 |

Detalhes completos (ressalvas, dados exatos, versões de biblioteca) ficam no model card gerado junto com cada adapter: `outputs/retriever-lora/best/README.md`.

## Como é feita a inferência

`python main.py ask` faz a pergunta em pt-BR virar resposta em duas etapas, com os **dois modelos carregados ao mesmo tempo**:

```
pergunta
  │
  ▼
1) RETRIEVER (ColQwen3 + adapter LoRA, 4-bit)
   embeda a pergunta → compara via MaxSim contra o índice das páginas (em cache, ver abaixo)
   → top-k páginas mais parecidas
  │
  ▼
2) GERADOR (Qwen3-VL-4B-Instruct, 4-bit, sem fine-tuning)
   lê a(s) página(s) do topo (imagem + texto extraído) + a pergunta
   → resposta em pt-BR, citando livro e página (número IMPRESSO no livro, não o índice interno —
     ver book_page_offsets em config.yaml)
```

Não existe um banco vetorial rodando: o índice é um arquivo (`page_index.pt`) com os embeddings de todas as páginas do manifesto, calculado uma vez e cacheado fora do repositório (`CACHE_DIR` em `config.yaml`, por padrão `G:\rag_cache` nesta máquina — cai em `outputs/retrieval_index/` se esse disco não existir). A comparação da pergunta contra as ~992 páginas é feita em memória (`torch.einsum` em lote na GPU). O índice é específico de cada adapter (`best`, `final` ou o modelo original têm espaços de embedding diferentes), então cada um tem seu próprio arquivo de cache.

`--no-generate` desliga a etapa 2 (só retrieval); `--gen-pages N` controla quantas páginas do topo o gerador lê (padrão 1); cada pergunta mostra o tempo gasto em cada etapa. Ver `python main.py ask --help` para as demais opções.
