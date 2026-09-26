"""Gera perguntas sinteticas em pt-BR para cada pagina (dados de treino do retriever).

O Qwen3-VL-4B local le a pagina (imagem + texto extraido) e escreve perguntas que so podem ser
respondidas com aquela pagina. Saida: data/synth/queries.jsonl, uma linha por pagina:
    {"book": ..., "page": ..., "queries": ["...", ...]}

Retomavel: paginas ja presentes no arquivo sao puladas (Ctrl+C e rode de novo). A ordem das
paginas e embaralhada (seed fixa) para que um run parcial ja cubra os tres livros.

Uso:  python -m dnd_rag.train.make_synthetic [--questions-per-page 5] [--limit N]
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
from PIL import Image

from dnd_rag.generate.vlm import DEFAULT_MODEL, fit_visual_tokens, load_qwen3vl_4bit

ROOT = Path(__file__).resolve().parents[3]

PROMPT = """Voce esta montando um conjunto de treino para um sistema de busca sobre livros de Dungeons & Dragons 5a edicao, em portugues do Brasil.

Leia a pagina (imagem e texto extraido abaixo) e escreva {n} perguntas em portugues que um jogador ou mestre poderia fazer e que SO podem ser respondidas com o conteudo desta pagina.

Regras:
- Cada pergunta deve ser autossuficiente: cite o nome da magia, criatura, classe, regra ou item. Nunca diga "esta pagina", "o texto", "a imagem", "acima" ou "abaixo".
- Varie os tipos: valor numerico (CA, PV, dano, alcance, CD), regra ou condicao, comparacao, descricao, lista.
- Use apenas informacoes que estao na pagina. Nao invente.
- Se a pagina nao tiver conteudo de jogo aproveitavel (capa, indice, pagina em branco, creditos), responda com uma lista vazia.

Texto extraido da pagina (pode estar fora de ordem):
\"\"\"
{text}
\"\"\"

Responda somente com JSON no formato {{"perguntas": ["...", "..."]}}."""

BANNED = re.compile(r"\b(esta p[aá]gina|nesta p[aá]gina|o texto|na imagem|a imagem|acima|abaixo|no trecho)\b", re.I)


def parse_questions(raw: str) -> list[str]:
    match = re.search(r"\{.*\}", raw, re.S)
    if match:
        try:
            items = json.loads(match.group(0)).get("perguntas", [])
        except (json.JSONDecodeError, AttributeError):
            items = []
    else:
        items = []
    if not items:  # fallback: uma pergunta por linha
        items = [ln.strip(' -*"0123456789.') for ln in raw.splitlines() if "?" in ln]
    seen, out = set(), []
    for q in items:
        q = str(q).strip()
        key = q.lower()
        if len(q) >= 15 and q.endswith("?") and not BANNED.search(q) and key not in seen:
            seen.add(key)
            out.append(q)
    return out


def load_done(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            done.add((row["book"], row["page"]))
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--questions-per-page", type=int, default=5)
    parser.add_argument("--min-chars", type=int, default=300, help="ignora paginas com menos texto (capas, em branco)")
    parser.add_argument("--max-visual-tokens", type=int, default=1000)
    parser.add_argument("--max-text-chars", type=int, default=3500)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--limit", type=int, default=0, help="processa no maximo N paginas (0 = todas)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest = [json.loads(ln) for ln in (args.data / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    out_path = args.data / "synth" / "queries.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)

    todo = [r for r in manifest if r["n_chars"] >= args.min_chars and (r["book"], r["page"]) not in done]
    random.Random(args.seed).shuffle(todo)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(done)} paginas ja feitas | {len(todo)} a processar", flush=True)
    if not todo:
        return

    model, processor = load_qwen3vl_4bit(args.model)
    model.eval()
    torch.manual_seed(args.seed)
    started = time.time()
    total_q = 0
    with out_path.open("a", encoding="utf-8") as sink:
        for i, row in enumerate(todo, 1):
            image = fit_visual_tokens(Image.open(args.data / row["image"]).convert("RGB"), args.max_visual_tokens)
            text = (args.data / row["text"]).read_text(encoding="utf-8")[: args.max_text_chars]
            prompt = PROMPT.format(n=args.questions_per_page, text=text)
            conversation = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            inputs = processor.apply_chat_template(
                [conversation], tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt", processor_kwargs={"padding": True},
            ).to("cuda")
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)
            raw = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            queries = parse_questions(raw)
            total_q += len(queries)
            sink.write(json.dumps({"book": row["book"], "page": row["page"], "queries": queries}, ensure_ascii=False) + "\n")
            sink.flush()
            if i % 10 == 0 or i == len(todo):
                rate = (time.time() - started) / i
                print(f"[{i}/{len(todo)}] {total_q} perguntas | {rate:.1f}s/pagina | ETA {rate * (len(todo) - i) / 3600:.1f}h", flush=True)


if __name__ == "__main__":
    main()
