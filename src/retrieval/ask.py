"""CLI interativa: digite uma pergunta em pt-BR e receba uma resposta com citacao (livro + pagina).

Duas etapas por pergunta: o retriever (ColQwen3 + LoRA) rankeia as paginas do manifesto contra o
indice em cache; o gerador (Qwen3-VL-4B-Instruct, --no-generate desliga) le a(s) pagina(s) melhor
posicionada(s) (imagem + texto extraido) e escreve a resposta. Os dois modelos ficam carregados ao
mesmo tempo (juntos cabem nos 8 GB medidos em scripts/vram_smoke_test.py).

Na primeira vez, embeda todas as paginas do manifesto e guarda em cache (CACHE_DIR, ver config.yaml);
demora alguns minutos. Nas proximas, carrega do cache e a busca e instantanea.

Uso:
  python main.py ask                                              # adapter outputs/retriever-lora/best
  python main.py ask --adapter outputs\\retriever-lora\\final
  python main.py ask --no-adapter                                 # retriever original, sem fine-tuning
  python main.py ask --no-generate                                # so a busca, sem gerar resposta
  python main.py ask --gen-pages 3 --top-k 10 --rebuild-index
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

from src.chunking.chunker import fit_visual_tokens
from src.embeddings.embedder import embed_pages, embed_query, load_image, load_retriever
from src.prompts.prompt_templates import ANSWER_PROMPT
from src.retrieval.retriever import score_matrix
from src.train.data import load_pages
from src.utils.helpers import CACHE_DIR, CONFIG, ROOT

DEFAULT_ADAPTER = ROOT / "outputs" / "retriever-lora" / "best"


def printed_page(book: str, page: int) -> int:
    """Numero IMPRESSO na pagina (o que aparece no rodape do livro), nao o indice do manifesto.

    Os dois nao batem (capa/guardas/sumario empurram cada livro de um jeito); ver book_page_offsets
    em config.yaml. So usar para o que e mostrado ao usuario — o resto do codigo usa o indice do
    manifesto (chaves, nomes de arquivo em data/pages|text/).
    """
    return page + CONFIG.get("book_page_offsets", {}).get(book, 0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "data")
    p.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER, help="pasta do adapter LoRA (ex.: outputs/retriever-lora/best)")
    p.add_argument("--no-adapter", action="store_true", help="ignora --adapter: usa o retriever original, sem fine-tuning")
    p.add_argument("--top-k", type=int, default=5, help="quantas paginas mostrar no ranking")
    p.add_argument("--snippet-chars", type=int, default=160)
    p.add_argument("--rebuild-index", action="store_true", help="reembeda todas as paginas mesmo com cache valido")
    p.add_argument("--no-generate", action="store_true", help="so a busca: nao carrega o gerador nem escreve resposta")
    p.add_argument("--gen-pages", type=int, default=1, help="quantas paginas do topo do ranking o gerador le")
    p.add_argument("--gen-max-visual-tokens", type=int, default=768)
    p.add_argument("--gen-max-new-tokens", type=int, default=400)
    args = p.parse_args()
    if args.no_adapter:
        args.adapter = None
    elif not args.adapter.is_dir():
        sys.exit(f"Adapter nao encontrado: {args.adapter}\n(gere um com `python main.py train`, ou use --no-adapter)")
    return args


def resolve_model_options(adapter: Path | None) -> tuple[str, int]:
    """base/max_visual_tokens do retriever: do config.json do treino ao lado do adapter, senao os padroes do projeto."""
    base, max_visual_tokens = CONFIG["models"]["retriever"], 768
    if adapter:
        config_path = adapter.parent / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8-sig"))
            base = config.get("base", base)
            max_visual_tokens = int(config.get("max_visual_tokens", max_visual_tokens))
    return base, max_visual_tokens


def index_path(adapter: Path | None) -> Path:
    """Cache dos embeddings de todas as paginas do manifesto, especifico do adapter (ou do original).

    Fica fora do repositorio (CACHE_DIR, ver config.yaml): e grande (~470 MB para o corpus inteiro,
    medido em 57 MB para 120 paginas) e deriva de conteudo licenciado, como data/ e outputs/.
    """
    if adapter is None:
        return CACHE_DIR / "_original" / "page_index.pt"
    try:
        name = adapter.resolve().relative_to((ROOT / "outputs").resolve())
    except ValueError:  # adapter fora de outputs/ (ex.: caminho externo): identificador mais simples
        name = Path(adapter.resolve().parent.name) / adapter.resolve().name
    return CACHE_DIR / name / "page_index.pt"


def build_or_load_index(model, processor, args, pages: dict) -> tuple[list, list]:
    """(chaves, embeddings) de todas as paginas do manifesto, com cache em disco (uma lista de tensores, uma por pagina)."""
    path, keys = index_path(args.adapter), sorted(pages)
    if path.exists() and not args.rebuild_index:
        cached = torch.load(path, weights_only=False)
        if cached["keys"] == keys and cached["data"] == str(args.data):
            print(f"indice: {len(keys)} paginas (cache: {path})", flush=True)
            return keys, cached["embeddings"]
        print("indice em cache nao bate com os dados atuais; recalculando...", flush=True)

    print(f"indexando {len(keys)} paginas (uma vez; ~1s/pagina; fica em cache em {path})...", flush=True)
    embeddings = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, key in enumerate(keys, 1):
            emb, _ = embed_pages(model, processor, [load_image(args, pages, key)])
            embeddings.append(emb[0].to(torch.bfloat16).cpu())
            if i % 50 == 0 or i == len(keys):
                print(f"  {i}/{len(keys)}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"keys": keys, "data": str(args.data), "embeddings": embeddings}, path)
    return keys, embeddings


@torch.inference_mode()
def search(model, processor, query: str, keys, embeddings, top_k: int):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        q_emb = embed_query(model, processor, query)[0].to(torch.bfloat16)
    scores = score_matrix([q_emb], embeddings)[0]
    top = scores.argsort()[::-1][:top_k]
    return [(keys[i], float(scores[i])) for i in top]


@torch.inference_mode()
def generate_answer(gen_model, gen_processor, question: str, args, pages: dict, hits: list) -> str:
    """hits: lista de (book, page) com as paginas que o gerador vai ler (ja recortada em --gen-pages)."""
    blocks, images = [], []
    for book, page in hits:
        name = CONFIG["books"].get(book, book)
        text = (args.data / pages[(book, page)]["text"]).read_text(encoding="utf-8")
        blocks.append(f"Livro: {name}\nPagina: {printed_page(book, page)}\nTexto extraido:\n\"\"\"\n{text}\n\"\"\"")
        image = Image.open(args.data / pages[(book, page)]["image"]).convert("RGB")
        images.append(fit_visual_tokens(image, args.gen_max_visual_tokens))

    prompt = ANSWER_PROMPT.format(pages="\n\n".join(blocks), question=question)
    content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": prompt}]
    conversation = [{"role": "user", "content": content}]
    inputs = gen_processor.apply_chat_template(
        [conversation], tokenize=True, add_generation_prompt=True, return_dict=True,
        return_tensors="pt", processor_kwargs={"padding": True},
    ).to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = gen_model.generate(**inputs, max_new_tokens=args.gen_max_new_tokens, do_sample=False)
    return gen_processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def main():
    args = parse_args()
    base, args.max_visual_tokens = resolve_model_options(args.adapter)
    args.base = base

    pages = load_pages(args.data)
    if not pages:
        sys.exit(f"Nenhuma pagina em {args.data}: rode `python main.py ingest` antes.")

    print(f"carregando {args.base}" + (f" + adapter {args.adapter}" if args.adapter else " (sem adapter)") + "...", flush=True)
    model, processor = load_retriever(args)
    model.eval()
    if args.adapter:
        from src.eval.run_retrieval_eval import load_adapter

        model = load_adapter(model, args.adapter)

    keys, embeddings = build_or_load_index(model, processor, args, pages)

    gen_model = gen_processor = None
    if not args.no_generate:
        from src.llm.llm_client import DEFAULT_MODEL as GEN_MODEL
        from src.llm.llm_client import load_qwen3vl_4bit

        print(f"carregando {GEN_MODEL} (gerador)...", flush=True)
        gen_model, gen_processor = load_qwen3vl_4bit()
        gen_model.eval()

    print(f"\nPronto. {len(keys)} paginas indexadas. Digite uma pergunta (vazio ou 'sair' para encerrar).\n", flush=True)
    while True:
        try:
            query = input("Pergunta> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.lower() in {"sair", "exit", "quit"}:
            break

        t0 = time.perf_counter()
        hits = search(model, processor, query, keys, embeddings, args.top_k)
        search_s = time.perf_counter() - t0

        gen_s = None
        if gen_model is not None:
            t1 = time.perf_counter()
            answer = generate_answer(gen_model, gen_processor, query, args, pages, [h[0] for h in hits[: args.gen_pages]])
            gen_s = time.perf_counter() - t1
            print(f"\n{answer}\n")
            print("Fontes consultadas:")

        for (book, page), score in hits:
            name = CONFIG["books"].get(book, book)
            text = (args.data / pages[(book, page)]["text"]).read_text(encoding="utf-8")
            snippet = " ".join(text.split())[: args.snippet_chars]
            image = args.data / pages[(book, page)]["image"]
            print(f"  {score:5.2f}  {name}, p.{printed_page(book, page):>4}  {image}")
            print(f"          {snippet}...")

        timing = f"busca {search_s:.1f}s" + (f" | resposta {gen_s:.1f}s | total {search_s + gen_s:.1f}s" if gen_s is not None else "")
        print(f"[tempo: {timing}]\n")


if __name__ == "__main__":
    main()
