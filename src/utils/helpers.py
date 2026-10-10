"""Utilidades comuns: raiz do projeto, config.yaml, leitura de JSONL e controle de energia do Windows."""
import ctypes
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


CONFIG = load_config()

# Caches grandes e regeneraveis (indices de embeddings); ver comentario em config.yaml. Cai em outputs/
# quando o disco configurado nao existe nesta maquina (ex.: outro ambiente sem G:).
_configured_cache = Path(p) if (p := CONFIG.get("paths", {}).get("cache_dir")) else None
_drive_missing = _configured_cache and _configured_cache.drive and not Path(_configured_cache.drive + "\\").exists()
CACHE_DIR = ROOT / "outputs" / "retrieval_index" if _configured_cache is None or _drive_missing else _configured_cache


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def keep_awake(on: bool):
    """Windows: evita suspender o PC durante o treino."""
    if sys.platform == "win32":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
