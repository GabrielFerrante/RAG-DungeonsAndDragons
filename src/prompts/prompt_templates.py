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
