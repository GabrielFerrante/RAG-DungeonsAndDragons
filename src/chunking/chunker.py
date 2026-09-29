"""Chunking do RAG visual: o chunk e a PAGINA inteira (imagem + texto extraido), nao um trecho de texto.

O retriever multi-vetor le a imagem da pagina, entao aqui so se ajusta o tamanho da imagem ao orcamento
de tokens visuais (memoria x nitidez). A extracao do texto por pagina fica em src/ingestion/loader.py.
"""
import math

from PIL import Image

TOKEN_PIXELS = 32 * 32  # patch 16 com merge 2x2 => 1 token visual por 32x32 px


def fit_visual_tokens(image: Image.Image, max_tokens: int) -> Image.Image:
    """Reduz a imagem (mantendo a proporcao) para que gere no maximo ~max_tokens tokens visuais."""
    factor = math.sqrt(max_tokens * TOKEN_PIXELS / (image.width * image.height))
    if factor >= 1:
        return image
    return image.resize((max(32, int(image.width * factor)), max(32, int(image.height * factor))), Image.LANCZOS)
