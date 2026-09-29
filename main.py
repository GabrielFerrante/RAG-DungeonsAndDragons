"""Ponto de entrada: despacha cada etapa do pipeline para o seu modulo.

  python main.py ingest      PDFs -> PNG + texto por pagina           (src/ingestion/loader.py)
  python main.py synthetic   perguntas sinteticas com o Qwen3-VL      (src/train/make_synthetic.py)
  python main.py train       fine-tuning LoRA do retriever            (src/train/train_retriever.py)
  python main.py eval        avaliacao de retrieval T2I / I2T         (src/eval/run_retrieval_eval.py)

As opcoes seguem o comando e sao as do modulo, ex.:  python main.py train --dry-run
"""
import importlib
import sys

COMMANDS = {
    "ingest": "src.ingestion.loader",
    "synthetic": "src.train.make_synthetic",
    "train": "src.train.train_retriever",
    "eval": "src.eval.run_retrieval_eval",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__)
    command = sys.argv.pop(1)
    sys.argv[0] = f"main.py {command}"  # o argparse do modulo mostra o comando completo no --help
    importlib.import_module(COMMANDS[command]).main()


if __name__ == "__main__":
    main()
