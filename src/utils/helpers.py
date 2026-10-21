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


# ----------------------------------------------------------------------------- .env (sem dependencia nova)

ENV_PATH = ROOT / ".env"


def read_env(path: Path = ENV_PATH) -> dict[str, str]:
    """Parser minimo de .env: linhas CHAVE=valor, ignora vazias e comentarios (#). Sem aspas/expansao."""
    if not path.exists():
        return {}
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()
    return env


def write_env_value(key: str, value: str, path: Path = ENV_PATH):
    """Atualiza (ou acrescenta) uma chave em .env, preservando as demais linhas e comentarios."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    pattern = key + "="
    for i, line in enumerate(lines):
        if line.strip().startswith(pattern):
            lines[i] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- deteccao de hardware (inferencia)

def detect_vram_gb() -> float | None:
    """VRAM total (GB) da GPU 0, ou None se nao houver CUDA disponivel."""
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(0).total_memory / 1e9


def resolve_device_mode(env: dict[str, str] | None = None) -> str:
    """"cuda" ou "cpu", pra inferencia (src/embeddings/embedder.py, src/llm/llm_client.py).

    DEVICE_MODE no .env, se setado explicitamente para "cuda" ou "cpu", manda (override manual).
    Caso contrario, detecta sozinho: sem GPU, ou com menos VRAM que hardware.min_vram_gb (config.yaml),
    cai para CPU. `python main.py setup` roda essa mesma logica e grava o resultado no .env, so pra
    deixar visivel/documentado — nao e preciso roda-lo antes: o fallback abaixo sempre se protege.

    `env` e so pra teste (injeta um .env falso sem tocar no arquivo real); producao sempre chama sem
    argumento, o que le o .env de verdade.
    """
    override = (env if env is not None else read_env()).get("DEVICE_MODE", "").strip().lower()
    if override in ("cuda", "cpu"):
        return override
    vram = detect_vram_gb()
    return "cuda" if vram is not None and vram >= CONFIG["hardware"]["min_vram_gb"] else "cpu"


def keep_awake(on: bool):
    """Windows: evita suspender o PC durante o treino."""
    if sys.platform == "win32":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
