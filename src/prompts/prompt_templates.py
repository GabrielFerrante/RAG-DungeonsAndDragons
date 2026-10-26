"""Templates de prompt do projeto."""

# Perguntas sinteticas por pagina (dados de treino do retriever). Placeholders: {n} e {text}.
SYNTHETIC_QUESTIONS_PROMPT = """Voce esta montando um conjunto de treino para um sistema de busca sobre livros de Dungeons & Dragons 5a edicao, em portugues do Brasil.

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


# Resposta final no `main.py ask`, depois do retriever encontrar a(s) pagina(s). Placeholders:
# {pages} (bloco "Livro: ... / Pagina: ... / Texto extraido: ..." por pagina, ja formatado) e {question}.
ANSWER_PROMPT = """Voce e um assistente que responde perguntas sobre Dungeons & Dragons 5a edicao, em portugues do Brasil, usando SOMENTE as paginas fornecidas (imagem e texto extraido de cada uma).

{pages}

Pergunta: {question}

Regras:
- Responda em portugues, de forma direta e objetiva.
- Cite a fonte de cada afirmacao no formato (Livro, p. N).
- Use a imagem para confirmar valores exatos (CA, PV, dano, etc.); o texto extraido pode estar fora de ordem.
- Se as paginas fornecidas NAO tiverem a resposta, diga claramente que nao encontrou a informacao nelas. Nao invente."""


# Modo I2T do `main.py ask`: o usuario fornece uma imagem (fora dos livros, ex.: arte avulsa de uma
# criatura) em vez de uma pergunta em texto. Placeholder: {pages} (mesmo formato do ANSWER_PROMPT).
IMAGE_QUERY_PROMPT = """Voce e um assistente sobre Dungeons & Dragons 5a edicao, em portugues do Brasil. A PRIMEIRA imagem desta mensagem foi fornecida pelo usuario (nao e uma pagina de livro). Identifique do que se trata — criatura, item, magia ou outro conceito do D&D 5E — usando as paginas abaixo (demais imagens + texto extraido de cada uma) como referencia.

{pages}

Regras:
- Responda em portugues, de forma direta.
- Diga do que se trata a imagem fornecida e cite a fonte no formato (Livro, p. N).
- Compare visualmente a primeira imagem com a(s) imagem(ns) da(s) pagina(s); use o texto extraido para confirmar valores exatos.
- Se nenhuma das paginas corresponder a imagem fornecida, diga isso claramente — nao invente."""
