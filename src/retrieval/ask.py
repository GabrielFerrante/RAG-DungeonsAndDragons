"""CLI interativa: dois modos, escolhidos uma vez no inicio (ou via --mode).

  T2T  digite uma pergunta em pt-BR -> resposta com citacao (livro + pagina)
  I2T  escolha uma imagem de ImagesI2T/ (fora dos livros, ex.: arte avulsa de uma criatura) ->
       resposta dizendo do que se trata, citando a pagina correspondente

Em ambos: o retriever (ColQwen3 + LoRA) rankeia as paginas do manifesto contra o indice em cache —
em T2T a consulta e a pergunta (embed_query), em I2T e a propria imagem (embed_pages, o mesmo jeito
que as paginas sao embedadas: o retriever alinha a imagem da consulta com as imagens das paginas no
mesmo espaco). Em I2T precisa de um adapter treinado pra isso (outputs/retriever-i2t-lora/best, ver
`python main.py train-i2t`). O gerador (Qwen3-VL-4B-Instruct, --no-generate desliga) le a(s) pagina(s)
melhor posicionada(s) (imagem + texto extraido) e escreve a resposta. Os dois modelos ficam carregados
ao mesmo tempo (GPU ou CPU, ver src/utils/helpers.py::resolve_device_mode).

Na primeira vez, embeda todas as paginas do manifesto e guarda em cache (CACHE_DIR, ver config.yaml);
demora alguns minutos. Nas proximas, carrega do cache e a busca e instantanea.

Uso:
  python main.py ask                                              # pergunta o modo (T2T/I2T)
  python main.py ask --mode t2t --adapter outputs\\retriever-lora\\final
  python main.py ask --mode i2t --adapter outputs\\retriever-i2t-lora\\best
  python main.py ask --no-adapter                                 # retriever original, sem fine-tuning
  python main.py ask --no-generate                                # so a busca, sem gerar resposta
  python main.py ask --gen-pages 3 --top-k 10 --rebuild-index
"""
import argparse
import json
import sys
import time
from pathlib import Path
from threading import Thread

import torch
from PIL import Image
from transformers import TextIteratorStreamer

from src.chunking.chunker import fit_visual_tokens
from src.embeddings.embedder import embed_pages, embed_query, load_image, load_retriever
from src.prompts.prompt_templates import ANSWER_PROMPT, IMAGE_QUERY_PROMPT
from src.retrieval.retriever import score_matrix
from src.train.data import load_pages
from src.utils.helpers import CACHE_DIR, CONFIG, ROOT, resolve_device_mode

STREAM_TIMEOUT_S = 120.0  # se gen_model.generate() morrer sem chamar streamer.end(), nao trava pra sempre

DEFAULT_ADAPTER = ROOT / "outputs" / "retriever-lora" / "best"
DEFAULT_IMAGES_DIR = ROOT / "ImagesI2T"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


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
    p.add_argument("--mode", choices=["t2t", "i2t"], help="pula a pergunta do modo no inicio (T2T: texto: I2T: imagem)")
    p.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR, help="pasta com as imagens do modo I2T")
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


def build_or_load_index(model, processor, args, pages: dict, adapter: Path | None) -> tuple[list, list]:
    """(chaves, embeddings) de todas as paginas do manifesto, com cache em disco (uma lista de tensores, uma por pagina).

    adapter: caminho do adapter que esta ativo em `model` (so decide onde fica o cache)."""
    path, keys = index_path(adapter), sorted(pages)
    if path.exists() and not args.rebuild_index:
        cached = torch.load(path, weights_only=False)
        if cached["keys"] == keys and cached["data"] == str(args.data):
            print(f"indice: {len(keys)} paginas (cache: {path})", flush=True)
            return keys, cached["embeddings"]
        print("indice em cache nao bate com os dados atuais; recalculando...", flush=True)

    print(f"indexando {len(keys)} paginas (uma vez; ~1s/pagina; fica em cache em {path})...", flush=True)
    embeddings = []
    device = next(model.parameters()).device.type
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
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
    """T2T: a consulta e uma pergunta em texto."""
    device = next(model.parameters()).device.type
    with torch.autocast(device, dtype=torch.bfloat16):
        q_emb = embed_query(model, processor, query)[0].to(torch.bfloat16)
    scores = score_matrix([q_emb], embeddings, device=device)[0]
    top = scores.argsort()[::-1][:top_k]
    return [(keys[i], float(scores[i])) for i in top]


@torch.inference_mode()
def search_by_image(model, processor, image: Image.Image, args, keys, embeddings, top_k: int):
    """I2T: a consulta e uma imagem (fora dos livros), alinhada no mesmo espaco de embeddings das
    paginas (mesmo redimensionamento, args.max_visual_tokens do retriever)."""
    resized = fit_visual_tokens(image, args.max_visual_tokens)
    device = next(model.parameters()).device.type
    with torch.autocast(device, dtype=torch.bfloat16):
        img_emb, _ = embed_pages(model, processor, [resized])
    q_emb = img_emb[0].to(torch.bfloat16)
    scores = score_matrix([q_emb], embeddings, device=device)[0]
    top = scores.argsort()[::-1][:top_k]
    return [(keys[i], float(scores[i])) for i in top]


@torch.inference_mode()
def generate_answer(gen_model, gen_processor, args, pages: dict, hits: list, question: str | None = None, query_image: Image.Image | None = None) -> str:
    """hits: lista de (book, page) que o gerador vai ler (ja recortada em --gen-pages). Um entre
    question (T2T) e query_image (I2T) precisa ser dado; o outro fica None."""
    blocks, images = [], []
    for book, page in hits:
        name = CONFIG["books"].get(book, book)
        text = (args.data / pages[(book, page)]["text"]).read_text(encoding="utf-8")
        blocks.append(f"Livro: {name}\nPagina: {printed_page(book, page)}\nTexto extraido:\n\"\"\"\n{text}\n\"\"\"")
        image = Image.open(args.data / pages[(book, page)]["image"]).convert("RGB")
        images.append(fit_visual_tokens(image, args.gen_max_visual_tokens))

    pages_block = "\n\n".join(blocks)
    if query_image is not None:
        prompt = IMAGE_QUERY_PROMPT.format(pages=pages_block)
        ordered_images = [fit_visual_tokens(query_image, args.gen_max_visual_tokens), *images]  # a do usuario vem primeiro (ver o prompt)
    else:
        prompt = ANSWER_PROMPT.format(pages=pages_block, question=question)
        ordered_images = images

    content = [{"type": "image", "image": im} for im in ordered_images] + [{"type": "text", "text": prompt}]
    conversation = [{"role": "user", "content": content}]
    device = next(gen_model.parameters()).device
    inputs = gen_processor.apply_chat_template(
        [conversation], tokenize=True, add_generation_prompt=True, return_dict=True,
        return_tensors="pt", processor_kwargs={"padding": True},
    ).to(device)

    # streaming: gen_model.generate() tem que rodar em outra thread pra poder ser lido token a token
    # aqui enquanto gera. Uma thread engole excecao por padrao (so imprime no stderr e segue) — sem o
    # try/except abaixo, um erro no meio da geracao travaria o `for chunk in streamer` esperando pra
    # sempre por tokens que nunca mais vem (daí o timeout tambem, por garantia).
    streamer = TextIteratorStreamer(gen_processor, skip_prompt=True, skip_special_tokens=True, timeout=STREAM_TIMEOUT_S)
    error: list[BaseException] = []

    def run_generate():
        try:
            with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
                gen_model.generate(**inputs, max_new_tokens=args.gen_max_new_tokens, do_sample=False, streamer=streamer)
        except BaseException as exc:
            error.append(exc)

    thread = Thread(target=run_generate, daemon=True)
    thread.start()

    print(flush=True)
    chunks = []
    for chunk in streamer:
        print(chunk, end="", flush=True)
        chunks.append(chunk)
    print("\n", flush=True)
    thread.join()
    if error:
        raise error[0]
    return "".join(chunks).strip()


def list_i2t_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def show_results(hits, pages: dict, args, answer: str | None, search_s: float, gen_s: float | None):
    if answer is not None:  # o texto ja foi impresso em stream por generate_answer(); so falta o cabecalho
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


def choose_mode(preset: str | None) -> str:
    if preset:
        return preset
    while True:
        choice = input("Modo: [1] Pergunta em texto (T2T)  [2] Imagem (I2T) > ").strip().lower()
        if choice in ("1", "t2t", ""):
            return "t2t"
        if choice in ("2", "i2t"):
            return "i2t"
        print("Digite 1 (T2T) ou 2 (I2T).")


def run_t2t(model, processor, gen_model, gen_processor, args, pages, keys, embeddings):
    print("\nDigite uma pergunta (vazio ou 'sair' para encerrar).\n", flush=True)
    while True:
        try:
            query = input("Pergunta> ").strip().lstrip("﻿")  # BOM: sobra de entrada redirecionada/em pipe
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.lower() in {"sair", "exit", "quit"}:
            break

        t0 = time.perf_counter()
        hits = search(model, processor, query, keys, embeddings, args.top_k)
        search_s = time.perf_counter() - t0

        answer, gen_s = None, None
        if gen_model is not None:
            t1 = time.perf_counter()
            answer = generate_answer(gen_model, gen_processor, args, pages, [h[0] for h in hits[: args.gen_pages]], question=query)
            gen_s = time.perf_counter() - t1

        show_results(hits, pages, args, answer, search_s, gen_s)


def run_i2t(model, processor, gen_model, gen_processor, args, pages, keys, embeddings):
    images = list_i2t_images(args.images_dir)
    if not images:
        print(f"Nenhuma imagem em {args.images_dir} (extensoes aceitas: {', '.join(sorted(IMAGE_EXTENSIONS))}).", flush=True)
        return

    print(f"\nImagens em {args.images_dir}:", flush=True)
    for i, p in enumerate(images, 1):
        print(f"  [{i}] {p.name}", flush=True)
    print("\nDigite o numero ou o nome do arquivo (vazio ou 'sair' para encerrar).\n", flush=True)

    while True:
        try:
            choice = input("Imagem> ").strip().lstrip("﻿")  # BOM: sobra de entrada redirecionada/em pipe
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not choice or choice.lower() in {"sair", "exit", "quit"}:
            break

        if choice.isdigit() and 1 <= int(choice) <= len(images):
            path = images[int(choice) - 1]
        else:
            matches = [p for p in images if choice in (p.name, p.stem)]
            path = matches[0] if matches else None
        if path is None:
            print(f"Nao encontrei '{choice}' na lista acima. Digite o numero ou o nome exato do arquivo.\n", flush=True)
            continue

        image = Image.open(path).convert("RGB")
        t0 = time.perf_counter()
        hits = search_by_image(model, processor, image, args, keys, embeddings, args.top_k)
        search_s = time.perf_counter() - t0

        answer, gen_s = None, None
        if gen_model is not None:
            t1 = time.perf_counter()
            answer = generate_answer(gen_model, gen_processor, args, pages, [h[0] for h in hits[: args.gen_pages]], query_image=image)
            gen_s = time.perf_counter() - t1

        show_results(hits, pages, args, answer, search_s, gen_s)


def main():
    args = parse_args()
    base, args.max_visual_tokens = resolve_model_options(args.adapter)
    args.base = base

    pages = load_pages(args.data)
    if not pages:
        sys.exit(f"Nenhuma pagina em {args.data}: rode `python main.py ingest` antes.")

    device_mode = resolve_device_mode()
    print(f"dispositivo: {device_mode}" + (" (sem GPU com VRAM suficiente; rode `python main.py setup` pra detalhes)" if device_mode == "cpu" else ""), flush=True)
    print(f"carregando {args.base}" + (f" + adapter {args.adapter}" if args.adapter else " (sem adapter)") + "...", flush=True)
    model, processor = load_retriever(args)
    model.eval()
    if args.adapter:
        from src.eval.run_retrieval_eval import load_adapter

        model = load_adapter(model, args.adapter)

    keys, embeddings = build_or_load_index(model, processor, args, pages, args.adapter)

    gen_model = gen_processor = None
    if not args.no_generate:
        from src.llm.llm_client import DEFAULT_MODEL as GEN_MODEL
        from src.llm.llm_client import load_qwen3vl_4bit

        print(f"carregando {GEN_MODEL} (gerador)...", flush=True)
        gen_model, gen_processor = load_qwen3vl_4bit()
        gen_model.eval()

    print(f"\nPronto. {len(keys)} paginas indexadas.", flush=True)
    mode = choose_mode(args.mode)
    if mode == "t2t":
        run_t2t(model, processor, gen_model, gen_processor, args, pages, keys, embeddings)
    else:
        run_i2t(model, processor, gen_model, gen_processor, args, pages, keys, embeddings)


if __name__ == "__main__":
    main()
