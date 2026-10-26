"""Ponto de entrada: despacha cada etapa do pipeline para o seu modulo.

  python main.py ingest      PDFs -> PNG + texto por pagina           (src/ingestion/loader.py)
  python main.py synthetic   perguntas sinteticas com o Qwen3-VL      (src/train/make_synthetic.py)
  python main.py train       fine-tuning LoRA do retriever (T2T)      (src/train/train_retriever.py)
  python main.py train-i2t   fine-tuning LoRA do retriever (I2T)      (src/train/train_i2t.py)
  python main.py eval        avaliacao de retrieval T2I / I2T         (src/eval/run_retrieval_eval.py)
  python main.py model-card  README.md (model card) dos adapters      (src/train/model_card.py)
  python main.py plot        grafico da perda de treino (log.jsonl)   (src/eval/plot_training.py)
  python main.py ask         pergunta em pt-BR -> paginas relevantes  (src/retrieval/ask.py)
  python main.py setup       cria o .env e decide GPU (4-bit) ou CPU  (src/utils/setup.py)

As opcoes seguem o comando e sao as do modulo, ex.:  python main.py train --dry-run
"""
import importlib
import sys

COMMANDS = {
    "ingest": "src.ingestion.loader",
    "synthetic": "src.train.make_synthetic",
    "train": "src.train.train_retriever",
    "train-i2t": "src.train.train_i2t",
    "eval": "src.eval.run_retrieval_eval",
    "model-card": "src.train.model_card",
    "plot": "src.eval.plot_training",
    "ask": "src.retrieval.ask",
    "setup": "src.utils.setup",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__)
    command = sys.argv.pop(1)
    sys.argv[0] = f"main.py {command}"  # o argparse do modulo mostra o comando completo no --help
    importlib.import_module(COMMANDS[command]).main()


if __name__ == "__main__":
    main()
