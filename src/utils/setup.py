"""Prepara o .env e decide se a inferencia roda na GPU (4-bit) ou na CPU (modelos em RAM, sem quantizar).

So afeta `python main.py ask` (retriever + gerador). Treino, avaliacao e geracao de perguntas sinteticas
continuam exigindo GPU (sao caros demais pra rodar em CPU de forma pratica).

Seguro de rodar de novo: nao sobrescreve um .env que ja existe, so avisa. Nao e obrigatorio rodar este
comando antes de usar o projeto — `ask` detecta a mesma coisa sozinho se o .env nao tiver DEVICE_MODE;
rodar `setup` so deixa a decisao visivel e documentada (e permite travar manualmente com DEVICE_MODE=).

Uso:  python main.py setup [--force] [--device cuda|cpu]
"""
import argparse
import shutil

from src.utils.helpers import CONFIG, ENV_PATH, ROOT, detect_vram_gb, resolve_device_mode, write_env_value


def ensure_env_file(force: bool = False) -> bool:
    """Cria .env a partir de .env.example se ainda nao existir. Retorna True se criou agora."""
    example = ROOT / ".env.example"
    if ENV_PATH.exists() and not force:
        print(f".env ja existe em {ENV_PATH} (nao sobrescrito; use --force pra recriar)", flush=True)
        return False
    if not example.exists():
        print(f"AVISO: {example} nao existe; criando .env vazio.", flush=True)
        ENV_PATH.write_text("", encoding="utf-8")
    else:
        shutil.copyfile(example, ENV_PATH)
    print(f".env criado em {ENV_PATH}", flush=True)
    return True


def report_device(vram: float | None, mode: str):
    if vram is None:
        print("GPU com CUDA: nao detectada.", flush=True)
    else:
        print(f"GPU com CUDA: {vram:.1f} GB de VRAM (minimo configurado: {CONFIG['hardware']['min_vram_gb']} GB).", flush=True)

    if mode == "cuda":
        print("Modo escolhido: GPU (4-bit NF4), igual ao resto do projeto.", flush=True)
    else:
        print(
            "Modo escolhido: CPU (4-bit NF4 tambem, em RAM — medido: ~4,3 GB por modelo, ~8,6 GB os dois).\n"
            "  Isso funciona, mas e bem mais lento (minutos por resposta, nao segundos). Pra forcar GPU\n"
            "  mesmo assim, edite DEVICE_MODE=cuda no .env.",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="recria o .env mesmo se ja existir (perde edicoes locais)")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="grava DEVICE_MODE no .env sem autodetectar (override manual)")
    args = parser.parse_args()

    ensure_env_file(force=args.force)

    vram = detect_vram_gb()
    mode = args.device or resolve_device_mode()
    write_env_value("DEVICE_MODE", mode)
    report_device(vram, mode)
    print(f"\nDEVICE_MODE={mode} gravado em {ENV_PATH}.", flush=True)


if __name__ == "__main__":
    main()
