# Projeto RAG para Dungeons And Dragons 5E

Criação de um chat para perguntas e respostas sobre D&D 5E com base nos guias oficiais em PT-BR.

RAG multimodal e 100% local (RTX 3060, 8 GB): o retriever (ColQwen3, multi-vetor) lê a **imagem** de cada página dos livros, e um VLM (Qwen3-VL) gera as perguntas sintéticas usadas no fine-tuning LoRA do retriever.

## Estrutura

```
RAG-DungeonsAndDragons/
├── README.md
├── requirements.txt       # dependências + ordem de instalação
├── environment.yml        # env conda `ragded`
├── config.yaml            # modelos, tamanho mínimo de chunk, nomes dos livros
├── .env                   # segredos (não versionado; modelo em .env.example)
├── .gitignore
├── main.py                # ponto de entrada: python main.py <ingest|synthetic|train|eval>
├── src/
│   ├── ingestion/         # loader.py: PDFs -> PNG + texto por página + manifesto
│   ├── chunking/          # chunker.py: o chunk é a página; ajusta a imagem ao orçamento de tokens visuais
│   ├── embeddings/        # embedder.py: embeddings multi-vetor (ColQwen3, 4-bit) de páginas e perguntas
│   ├── vectordb/          # vector_store.py: índice vetorial (Qdrant) — a implementar
│   ├── retrieval/         # retriever.py: busca por similaridade (MaxSim)
│   ├── prompts/           # prompt_templates.py: templates de prompt
│   ├── llm/               # llm_client.py: Qwen3-VL em 4-bit
│   ├── api/               # routes.py: endpoints FastAPI — a implementar
│   ├── utils/             # helpers.py: raiz do projeto, config.yaml, JSONL, controle de energia
│   ├── train/             # perguntas sintéticas, dados e fine-tuning LoRA do retriever
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
python -m pytest                    # testes
```

O pipeline completo, retomável em cada etapa, roda com `.\scripts\run_finetune.ps1` (`-Smoke` para um teste rápido ponta a ponta).
