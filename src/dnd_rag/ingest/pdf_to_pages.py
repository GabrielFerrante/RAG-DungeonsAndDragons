"""Fase 1: PDFs -> PNG por pagina + texto por pagina + manifesto JSONL.

Idempotente: paginas ja renderizadas/extraidas sao puladas, entao pode ser interrompido e retomado.

Uso:  python -m dnd_rag.ingest.pdf_to_pages [--scale 2.0]
"""
import argparse
import json
from pathlib import Path

import pdfplumber
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[3]

# slug (nome do arquivo sem extensao, minusculo) -> nome para citacao
BOOKS = {
    "playerbook": "Livro do Jogador",
    "masterguidebook": "Guia do Mestre",
    "monstersbook": "Livro dos Monstros",
}


def extract_text(page) -> str:
    """Texto da pagina; em layout de 2 colunas le a coluna esquerda e depois a direita.

    O extract_text() padrao intercala as colunas linha a linha. Consideramos a pagina como
    2 colunas quando (quase) nenhuma palavra cruza a calha central e ha texto dos dois lados.
    """
    words = page.extract_words()
    if not words:
        return ""
    mid = page.width / 2
    crossing = sum(1 for w in words if w["x0"] < mid - 3 and w["x1"] > mid + 3)
    left = sum(1 for w in words if w["x1"] <= mid + 3)
    right = sum(1 for w in words if w["x0"] >= mid - 3)
    two_columns = crossing / len(words) < 0.02 and min(left, right) / len(words) > 0.15
    if not two_columns:
        return page.extract_text() or ""
    halves = [page.crop((0, 0, mid, page.height)), page.crop((mid, 0, page.width, page.height))]
    return "\n".join(h.extract_text() or "" for h in halves)


def ingest_book(pdf_path: Path, out: Path, scale: float, max_pages: int = 0) -> list[dict]:
    slug = pdf_path.stem.lower()
    pages_dir, text_dir = out / "pages" / slug, out / "text" / slug
    pages_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    pdf = pdfium.PdfDocument(str(pdf_path))
    n_pages = min(len(pdf), max_pages) if max_pages else len(pdf)
    with pdfplumber.open(pdf_path) as plumber:
        for i in range(n_pages):
            number = i + 1  # numeracao 1-based do PDF (indice fisico, nao o numero impresso na pagina)
            image_path = pages_dir / f"{number:04d}.png"
            text_path = text_dir / f"{number:04d}.txt"
            if not image_path.exists():
                page = pdf[i]
                page.render(scale=scale).to_pil().convert("RGB").save(image_path)
                page.close()
            if not text_path.exists():
                text_path.write_text(extract_text(plumber.pages[i]), encoding="utf-8")
            rows.append(
                {
                    "book": slug,
                    "book_name": BOOKS.get(slug, slug),
                    "page": number,
                    "image": image_path.relative_to(out).as_posix(),
                    "text": text_path.relative_to(out).as_posix(),
                    "n_chars": len(text_path.read_text(encoding="utf-8")),
                }
            )
            if number % 25 == 0 or number == n_pages:
                print(f"  {slug}: {number}/{n_pages}", flush=True)
    pdf.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf-dir", type=Path, default=ROOT / "ArquivosEFontes")
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    parser.add_argument("--scale", type=float, default=2.0, help="escala de render (1.0 = 72 dpi)")
    parser.add_argument("--max-pages", type=int, default=0, help="so as N primeiras paginas de cada livro (teste)")
    args = parser.parse_args()

    manifest = []
    for pdf_path in sorted(args.pdf_dir.glob("*.pdf")):
        print(f"{pdf_path.name}", flush=True)
        manifest += ingest_book(pdf_path, args.out, args.scale, args.max_pages)

    manifest_path = args.out / "manifest.jsonl"
    manifest_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in manifest) + "\n", encoding="utf-8")
    print(f"{len(manifest)} paginas -> {manifest_path}")


if __name__ == "__main__":
    main()
