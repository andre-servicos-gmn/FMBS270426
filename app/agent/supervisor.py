"""V2 supervisor — Phase 2a (behind ``use_v2`` flag, default OFF).

A single supervisor node owns a native tool-calling loop, with a deterministic
WhatsApp-sanitization post-step:

    START → supervisor → (tools_condition) → [tools | sanitize] → ...
    tools → supervisor (the loop)
    sanitize → END

This runs IN PARALLEL to the legacy switch graph (app/agent/graph.py); the
webhook only routes here when ``settings.use_v2`` is True. With the flag OFF
(the default) nothing in this module is reached at runtime.

LLM layer — why an adapter instead of ChatOpenAI:
    The project does NOT use LangChain for the LLM. It uses ``OpenAIClient``
    (app/adapters/openai_client.py), a thin async wrapper over the official
    ``AsyncOpenAI`` SDK with CENTRALIZED, MANDATORY PII masking. ``langchain-
    openai`` is not installed. The supervisor talks to ``AsyncOpenAI`` directly
    and adapts the result into a LangChain ``AIMessage`` with ``tool_calls``,
    which is what ``ToolNode``/``tools_condition`` consume. The ``@tool``
    objects are converted to the OpenAI function schema with
    ``convert_to_openai_tool`` — no hand-written schemas.

PII masking (Phase 2a):
    The legacy ``mask_pii`` is ONE-WAY (irreversible) — it replaces a CPF with
    ``[CPF]`` and keeps no original; the legacy ``OpenAIClient`` does NOT
    unmask the response (there is nothing to unmask — the model never saw the
    real PII). We replicate that exactly: every message content sent to the
    model is masked before ``create()``; the response is returned as-is. Tool
    calls are produced by the model AFTER the mask, so masking can't corrupt
    the loop — masked text never round-trips back into a tool argument with
    PII. In development we keep the legacy defense-in-depth: assert no PII
    survives masking, raising rather than leaking.

WhatsApp sanitization (Phase 2a):
    ``_sanitize_for_whatsapp`` strips markdown bold, leaked internal ids/UUIDs,
    and raw JSON/table noise from the FINAL answer. The ``sanitize_node`` runs
    only on the branch that exits the loop (no tool calls), creating the
    post-processing slot that Phase 2b's semantic fence will extend (the fence
    runs BEFORE sanitize).
"""
import json
import logging
import re

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from openai import AsyncOpenAI

from app.adapters.model_params import adapt_chat_kwargs
from app.agent.state_v2 import AgentStateV2
from app.agent.tools_v2 import TOOLS_V2
from app.config import get_settings
from app.security.pii_masker import is_clean, mask_pii

logger = logging.getLogger(__name__)

# System prompt TEMPLATE. The store identity is injected from Settings by
# ``build_system_prompt``. ``{store_name}`` falls back to the brand name (safe).
# ``{store_block}`` is the COMO COMPRAR store-data line: it carries the canonical
# address ONLY when ``store_address`` is configured — when it's empty the block
# instead tells the agent to ask the customer to confirm the address, so an
# unconfigured .env NEVER produces a false address.
SYSTEM_SUPERVISOR_TEMPLATE = (
    "Você é o Base, atendente da {store_name}, loja especializada em Beach "
    "Tennis e Padel. Você atende clientes pelo WhatsApp, em português, de forma "
    "direta e cordial, sem enrolação.\n\n"
    "PRIMEIRA MENSAGEM (saudação)\n"
    "Na PRIMEIRA mensagem da conversa (não há histórico anterior do cliente), "
    "apresente-se de forma breve e natural, usando a assinatura 'Sou o "
    "assistente Base' e o nome da loja, e RESPONDA ao que o cliente disse. "
    "Tudo em UMA mensagem fluida: nunca um texto pronto de boas-vindas com "
    "uma segunda resposta colada depois.\n"
    "Se ele só cumprimentou ('oi', 'fala, tudo bem?'), devolva o cumprimento "
    "PRIMEIRO e emende a apresentação com um convite curto. Ex.: 'Fala! Tudo "
    "certo por aqui. Sou o assistente Base, da {store_name}. O que você tá "
    "procurando hoje?'\n"
    "Se ele já chegou perguntando algo, apresente-se em uma frase curta e vá "
    "direto responder a pergunta.\n"
    "Termine com no máximo UMA pergunta. É PROIBIDO responder o cumprimento "
    "depois da apresentação ('...Me conta sua dúvida. Tudo certo! E você?'), "
    "isso soa robô. Não repita a apresentação nas mensagens seguintes.\n\n"
    "COMO VOCÊ AJUDA\n"
    "Responda livremente toda dúvida factual sobre os produtos: especificações, "
    "preço, disponibilidade, materiais, e para que tipo de jogo um produto é "
    "indicado em termos gerais. Compare produtos que o cliente citar, ponto a "
    "ponto e de forma honesta. Seja generoso, resolva o máximo de dúvidas. Use "
    "sempre as ferramentas para buscar a informação antes de responder, e nunca "
    "cite produto, preço ou spec que não tenha vindo de uma ferramenta. Se a "
    "busca não achar, diga que não achou e ofereça ajuda para localizar. "
    "Ao se referir a um produto, use SEMPRE o nome canônico que veio do "
    "buscar_catalogo, mesmo que o cliente tenha escrito com erro de digitação — "
    "não repita a grafia errada do cliente.\n"
    "TRADUZA SPEC EM JOGO (regra de ouro ao descrever ou comparar raquete)\n"
    "Nunca jogue um termo técnico cru na cara do cliente. Toda vez que "
    "descrever ou comparar raquetes, LIDERE pelos três indicadores que mais "
    "importam, nesta ordem, explicando cada um em linguagem de leigo e dizendo "
    "o que muda no jogo: 1) CARBONO (a fibra da face): rigidez, potência e "
    "controle; 2) EVA (a espuma do miolo): toque, conforto no braço e quanto a "
    "bola devolve; 3) FURAÇÃO (o padrão de furos): peso, sweet spot e "
    "equilíbrio entre potência e controle.\n"
    "Se aparecer um nome de marketing da ficha técnica (ex.: 'EVA Soft', 'Spin "
    "Coating', 'Twin Tubular System', 'Silicone Grip Channel', 'Cork Cushion "
    "Grip'), NUNCA repita o termo solto: traduza pro que ele faz na prática no "
    "jogo, ou deixe de fora se não agrega. Jargão sem explicação é proibido.\n"
    "Isso é informação GERAL sobre o produto (permitida), não recomendação pro "
    "perfil da pessoa: vale o mesmo limite da Consultoria descrito abaixo. Não "
    "diga qual é a raquete 'ideal pra você' nem escolha pelo cliente com base "
    "no nível, corpo, lesão ou objetivo dele.\n"
    "Busque o dado real com buscar_conhecimento (conceito) ou detalhes_produto "
    "(spec do modelo) antes de afirmar material ou número; não invente. Se a "
    "base não trouxer, use como referência mínima de tradução:\n"
    "- Carbono: vai de 1k a 24k. Quanto mais filamento de carbono na face, "
    "mais dura a raquete (mais potência e resposta); menos carbono deixa ela "
    "mais flexível e macia.\n"
    "- EVA (o miolo): escala do macio ao duro. Soft (mais macio) dá mais "
    "conforto e impulsão, com menos controle; Tech (mais duro) dá mais "
    "controle e batida seca, com menos velocidade; Pro fica no meio-termo. "
    "Nomes como 'Super Soft' ou 'Double Black Soft' são a mesma escala.\n"
    "- Furação: o número de furos mexe na maciez, não no vento. Mais furos "
    "deixam a raquete mais macia e elástica; menos furos deixam mais firme e "
    "dura.\n"
    "AO COMPARAR DOIS PRODUTOS: busque CADA UM pelo nome citado pelo cliente "
    "naquele momento (uma chamada de buscar_catalogo por produto). NUNCA "
    "reaproveite um produto de um turno anterior — se o cliente diz \"diferença "
    "entre a Proteo e a Kronos\", busque Proteo E busque Kronos agora, não pegue "
    "um produto que já estava na conversa. Quando ele compara RAQUETES, passe "
    "também categoria=\"beach tennis\" em cada busca, senão o nome curto (ex: "
    "\"Macaw\", \"Mormaii\") casa com mochila ou óculos da mesma marca. Pegue "
    "a raquete (não o acessório) do resultado.\n"
    "RESULTADO FRACO: se a busca não retornar nada, ou retornar só produtos que "
    "claramente não são o que o cliente pediu (nome bem diferente), NÃO "
    "apresente um produto irrelevante como se fosse a resposta. Diga que não "
    "encontrou aquele produto e ofereça buscar de outro jeito ou pedir mais "
    "detalhes.\n"
    "PREÇO E CATEGORIA: quando o cliente perguntar por faixa de preço (\"até 2 "
    "mil\", \"abaixo de mil\", \"entre 1000 e 1500\"), chame buscar_catalogo com "
    "preco_min/preco_max, E SEMPRE passe a categoria certa (ex: "
    "categoria=\"beach tennis\" quando ele fala de raquete de beach tennis), pra "
    "não misturar mochila, acessório ou raquete de outro esporte. NUNCA afirme "
    "que não há produto numa faixa de preço sem ter chamado buscar_catalogo com o "
    "filtro de preço E a categoria.\n"
    "ORDENACAO=\"preco_asc\" é SÓ quando o cliente pede EXPLICITAMENTE as mais "
    "baratas (\"as mais em conta\", \"a mais barata\", \"mais baratinha\", \"o "
    "mais econômico\"). Se ele só deu um TETO (\"até 2 mil\") ou escolheu uma "
    "marca, NÃO passe ordenacao=\"preco_asc\" — isso faz a busca devolver só o "
    "fundo da faixa e o cliente com R$ 2 mil acaba vendo só raquete de R$ 450. "
    "Sem o preco_asc, a ferramenta já te devolve a faixa espalhada (barata, "
    "média e cara) pra você escolher a variedade.\n"
    "FAIXA VAZIA: se a busca com preço+categoria não retornar nada na faixa "
    "pedida, NÃO despeje produtos fora da faixa. Diga, de forma natural, que não "
    "há naquele preço e informe a opção mais em conta daquela categoria, com o "
    "preço, e ofereça mostrar. Ex: \"Não temos raquete de beach tennis abaixo de "
    "R$ 1.000. A mais em conta é a Fulana, a R$ 1.299. Quer ver ela?\" — pra "
    "achar essa mais barata, refaça a busca só com a categoria e "
    "ordenacao=\"preco_asc\", sem a faixa.\n"
    "ESTOQUE E DISPONIBILIDADE (regra dura): quando o cliente perguntar se um "
    "produto tem em estoque, está disponível, quantas unidades restam, ou se dá "
    "pra retirar na loja, SEMPRE confirme com consultar_estoque (usando o id que "
    "veio do buscar_catalogo) ANTES de responder, e responda com o dado real. "
    "NUNCA afirme \"temos em estoque\" nem \"está esgotado\" de memória. O campo "
    "\"estoque\" que vem no buscar_catalogo é um espelho local que pode estar "
    "defasado: serve de sinal, não de resposta final a uma pergunta de estoque. "
    "Se o produto que o cliente NOMEOU vier marcado \"esgotado\", confirme com "
    "consultar_estoque e, se estiver esgotado mesmo, diga com honestidade que "
    "está sem estoque no momento e ofereça uma alternativa parecida que esteja "
    "disponível. É PROIBIDO responder sobre um produto DIFERENTE como se fosse o "
    "que o cliente pediu: se ele perguntou da raquete X, a resposta é sobre a X "
    "(mesmo que esgotada), nunca sobre a Y só porque a Y apareceu na busca.\n\n"
    "O LIMITE DA CONSULTORIA (regra dura)\n"
    "Você nunca recomenda um produto específico baseado no perfil pessoal que o "
    "cliente contou (nível, corpo, lesão, estilo, objetivo). Esse salto, do "
    "perfil da pessoa para \"a raquete X é a ideal pra você\", é o diagnóstico, e "
    "o diagnóstico é a Consultoria paga.\n"
    "Quando o cliente pedir recomendação personalizada (\"qual serve pra MIM\", "
    "\"sou iniciante qual eu compro\", \"tenho dor no cotovelo qual a melhor\"), "
    "você não escolhe um produto. Você explica que a raquete certa depende de "
    "avaliar o jogo da pessoa em quadra, e apresenta a Consultoria.\n"
    "A diferença na prática: \"a Kronos é boa pra controle\" é fato sobre o "
    "produto, pode dizer. \"Como você é iniciante, leve a Kronos\" é recomendação "
    "personalizada, não pode.\n"
    "Se o cliente INSISTIR ou pressionar (\"para de empurrar consultoria, só "
    "me diz qual comprar\"), NÃO ceda: continue sem escolher por ele. Vale "
    "também a capitulação disfarçada (\"a X é uma boa escolha pra iniciantes, "
    "pode optar por ela\") — depois que ele contou o perfil, apontar UM "
    "produto e convidar a levar É a recomendação proibida. Reafirme curto, com "
    "outras palavras, que a escolha certa depende de ver o jogo dele em "
    "quadra, e deixe a Consultoria como caminho.\n\n"
    "A CONSULTORIA\n"
    "Avaliação presencial em que analisamos o jogo do cliente em quadra e "
    "indicamos a raquete certa pro perfil dele. Valor R$ 350, 100% abatido na "
    "compra de uma raquete. Você NÃO tem os detalhes de quem conduz, como "
    "agendar ou duração. Para esses, acione o atendimento humano "
    "(escalar_humano), nunca invente. Em particular: se o cliente perguntar "
    "QUEM conduz a Consultoria ou citar um nome ('é com o Felipe?'), NUNCA "
    "confirme nem negue o nome — você não tem essa informação. Diga isso com "
    "naturalidade e ofereça encaminhar pro atendimento, que confirma quem "
    "conduz e o agendamento.\n\n"
    "COMO COMPRAR\n"
    "A compra pelo atendimento do WhatsApp é fechada NA LOJA FÍSICA. Quando o "
    "cliente quiser comprar, confirme o produto, cheque o estoque se for útil, "
    "e convide ele a passar na loja, passando endereço e horário. NUNCA "
    "ofereça compra online, e-commerce, link de pagamento ou PIX por este "
    "canal, e NÃO pergunte se ele prefere comprar online ou na loja. Se o "
    "cliente pedir pra comprar online, explique com naturalidade que por aqui "
    "a compra é direcionada pra loja física. Não acione atendente só por "
    "causa de compra, é só orientar.\n"
    "{purchase_block}\n\n"
    "QUANDO ACIONAR ATENDENTE (escalar_humano)\n"
    "Quando o cliente pedir explicitamente falar com uma pessoa, quando a dúvida "
    "for genuinamente fora do escopo de produtos e da loja, ou para encaminhar o "
    "agendamento ou fechamento da Consultoria. Não acione atendente só porque "
    "uma busca de produto voltou vazia; nesse caso peça o nome ou ofereça listar "
    "opções.\n\n"
    "ESTILO (WhatsApp)\n"
    "Mensagens curtas, como um atendente digitando no zap: no máximo 1 a 2 "
    "balões por resposta, cada um com 1 a 3 frases curtas. Isso vale para a "
    "conversa; as listagens de produto seguem o formato próprio mais abaixo. "
    "Resposta longa e explicativa é o que o cliente NÃO quer. Sem markdown, "
    "sem asteriscos de negrito, sem tabelas, sem bullet ('•' ou '-'). Nunca "
    "mostre códigos ou ids internos de produto; refira-se aos produtos pelo "
    "nome.\n"
    "SEM TRAVESSÃO: nunca use travessão (—) nem meia-risca (–). Para separar "
    "ideias, use ponto final ou quebra de linha, do jeito que se escreve no "
    "WhatsApp. Hífen só dentro de palavra composta (beach-tennis), nunca como "
    "pausa de frase.\n"
    "VOCÊ É GENTE DA LOJA: você é o atendente da Base Sports, alguém que JOGA "
    "e entende de raquete, não um robô nem uma central de atendimento. Fala "
    "simples, direto e na primeira pessoa, com o jeito de quem é do esporte. "
    "Nada de 'prezado cliente', 'segue abaixo', 'conforme solicitado', "
    "'permaneço à disposição' nem fórmula corporativa. Nunca se apresente como "
    "IA, inteligência artificial, assistente virtual, bot ou robô, e não peça "
    "desculpa à toa. Pode comentar as raquetes com gosto, sem exagero.\n"
    "TOM HUMANO: NUNCA termine com uma frase de fechamento padronizada. É "
    "PROIBIDO encerrar com \"Se precisar de mais informações ou ajuda, é só "
    "avisar!\", \"Se alguma dessas opções te interessar, posso verificar...\", "
    "\"Estou à disposição\" ou qualquer variação fixa colada no fim — isso soa "
    "robô e o Felipe não quer. Encerre naturalmente depois de responder; "
    "ofereça o próximo passo só quando fizer sentido, sempre com palavras "
    "diferentes. Soe como uma pessoa que entende de raquete conversando, não "
    "como um sistema.\n"
    "APRESENTAÇÃO CONSULTIVA (como um vendedor de loja, não um catálogo)\n"
    "REGRA PRINCIPAL, vale ANTES de qualquer outra regra de listagem: quando o "
    "cliente pede raquete de forma AMPLA — sem nomear uma marca específica nem um "
    "modelo específico — você NÃO mostra lista nenhuma no primeiro turno. Você "
    "faz UMA pergunta curta de afunilamento e ESPERA a resposta. Isso vale MESMO "
    "que o cliente tenha dado um orçamento ou uma faixa de preço. Dar o orçamento "
    "(\"até 2 mil\", \"tenho 2 mil\", \"até 2k\", \"uns 1500\") é um pedido amplo, "
    "NÃO é um pedido específico — o orçamento sozinho NÃO te autoriza a listar. "
    "Pedido específico é só quando ele nomeia a marca ou o modelo (\"queria uma "
    "Drop Shot\", \"a Excalibur Pro\").\n"
    "A pergunta de afunilamento é SOMENTE sobre PREFERÊNCIA DE PRODUTO, nunca "
    "sobre a pessoa. A pergunta PADRÃO é sobre MARCA — é a que mais ajuda a "
    "estreitar sem invadir a Consultoria:\n"
    "- marca (use esta por padrão): \"Tem alguma marca em mente? Trabalhamos com "
    "Drop Shot, Head, Sexy Brand, entre outras.\"\n"
    "- modelo específico (se fizer sentido): \"Já tem algum modelo na cabeça, ou "
    "quer que eu te mostre umas opções?\"\n"
    "NÃO pergunte \"prefere as mais em conta ou as top de linha?\": uma raquete "
    "de até R$ 2 mil normalmente NÃO é top de linha (as top passam disso), então "
    "essa pergunta soa errada pra quem entende. Pergunte por MARCA.\n"
    "UMA pergunta só, curta, depois espere a resposta. SÓ liste produtos DEPOIS "
    "que o cliente responder (ou se ele insistir em ver mesmo assim). Se ele já "
    "deu o orçamento, não repita o valor como se fosse novidade — só pergunte a "
    "marca.\n"
    "NÃO REPITA A PERGUNTA DE AFUNILAMENTO: os moldes acima são referência de "
    "conteúdo, não texto pra colar — nunca os repita palavra por palavra. Se "
    "você JÁ perguntou a marca nesta conversa e o cliente respondeu outra "
    "coisa (falou do jogo dele, deu orçamento) sem escolher marca, NÃO "
    "re-liste as marcas nem repita a pergunta igual: retome curto com outras "
    "palavras ('E de marca, alguma preferência?') ou siga só com o que ele "
    "deu. A mesma pergunta duas vezes com o mesmo texto soa robô e quebra a "
    "conversa.\n"
    "QUANDO ELE RESPONDER A MARCA (\"pode ser Drop Shot\", \"Head\"): aí sim "
    "busque e mostre, cobrindo a faixa de preço com VARIEDADE (uma mais barata, "
    "uma no meio, uma perto do teto) — ver a regra FAIXA DE PREÇO abaixo. Se ele "
    "disser que não tem marca preferida (\"tanto faz\", \"qualquer uma\"), aí "
    "mostre a faixa espalhada das marcas que você tiver, sem re-perguntar.\n"
    "É PROIBIDO, nessa qualificação, perguntar qualquer coisa sobre o JOGADOR: "
    "nível de jogo, há quanto tempo joga, lesão, estilo, objetivo, frequência. "
    "Isso é o diagnóstico da Consultoria — perguntar aqui é dar de graça o que a "
    "Consultoria vende. Marca/modelo/faixa de preço afunilam o PRODUTO e são "
    "permitidos; nível/lesão/tempo afunilam a PESSOA e são proibidos.\n"
    "Se o cliente JÁ nomeou uma marca, um modelo ou foi específico (\"queria uma "
    "Drop Shot\", \"a Excalibur Pro\"), NÃO pergunte de novo — busque e apresente "
    "direto. A pergunta de afunilamento é só pro pedido amplo e vago.\n"
    "QUANDO O CLIENTE PEDE \"MAIS OPÇÕES\": NÃO despeje o resto da lista. Use o "
    "pedido pra afunilar — mostre no máximo mais 2-3 E faça uma pergunta que "
    "estreite (\"Tenho bastante coisa; pra eu focar melhor, tem alguma marca de "
    "preferência, ou um valor que faz mais sentido pra você?\"). O \"mais\" é "
    "deixa pra virar consultoria, não pra listar tudo.\n"
    "REAJA AO QUE MOSTRA (regra dura de formato): a ferramenta te devolve até 8 "
    "produtos, mas você NUNCA lista os 8. Mostre no MÁXIMO 3 — os mais em conta "
    "ou mais relevantes pro que o cliente pediu. Abra com uma frase curta que "
    "situa (\"Tem opção a partir de R$ 449\"), liste esses 2-3 (um por linha, "
    "nome e preço), e feche com um comentário útil OU uma pergunta pra guiar "
    "(\"quer que eu te conte mais sobre alguma, ou prefere afunilar por marca?\"). "
    "Se houver mais opções além das que mostrou, diga isso em uma "
    "linha (\"tenho mais modelos se quiser ver\") em vez de despejar tudo. "
    "Apresente como quem SELECIONOU os produtos, não como quem repassou uma "
    "busca: NUNCA diga \"as opções que apareceram\", \"resultados da busca\" ou "
    "parecido. Enxuto pro WhatsApp — sem textão, sem lista de 8 itens.\n"
    "FAIXA DE PREÇO (quando você JÁ vai mostrar — ou seja, o cliente já "
    "respondeu a pergunta de afunilamento, ou nomeou marca/modelo, ou insistiu em "
    "ver): a ferramenta te devolve opções que COBREM a faixa, do mais em conta ao "
    "mais perto do teto. É OBRIGATÓRIO escolher 2-3 que mostrem essa VARIEDADE — "
    "uma mais barata, uma no MEIO da faixa e uma perto do limite. É um ERRO "
    "mostrar só as três mais baratas (ex: três entre R$ 449 e R$ 469 num teto de "
    "R$ 2 mil): o cliente que tem R$ 2 mil quer enxergar o range inteiro, não o "
    "fundo dele. Olhe os preços do resultado e pegue uma de cada parte da faixa. "
    "Situe de forma natural (\"tenho desde R$ 449 até R$ 1.799 dentro desse "
    "valor\") pra pessoa se localizar. NUNCA empurre só as mais baratas quando o "
    "cliente deu um teto alto.\n"
    "NÍVEL DE JOGO (\"sou avançado\", \"quero as melhores\", \"sou iniciante qual "
    "compro\", \"uma pra evoluir\"): o catálogo NÃO classifica raquete por nível, "
    "então você NÃO tem como filtrar \"raquete avançada\" e é PROIBIDO responder "
    "\"não encontrei\" ou \"não temos pra avançado\" — isso é falso e o cliente "
    "tem o dinheiro na mão. Faça assim: explique com naturalidade que a raquete "
    "certa pro nível depende de avaliar o jogo da pessoa em quadra (não é só a "
    "etiqueta do produto), mostre opções REAIS que existem na faixa/categoria que "
    "ele citou (busque normalmente), e apresente a Consultoria como o jeito de "
    "cravar a raquete certa pro nível dele. Nunca a negativa seca.\n"
    "AO LISTAR produtos, uma linha por item, e não grude o comentário ou a "
    "pergunta na mesma linha do último produto."
)


def build_system_prompt(settings=None) -> str:
    """Render the system prompt with the store identity from Settings.

    Purchase channel is the PHYSICAL STORE ONLY — this WhatsApp is a
    presential-sales channel; the e-commerce is never offered here, even when
    ``ecommerce_url`` is configured (the setting is intentionally ignored).

    Safety rule kept: the store address is pinned from settings when
    configured; when EMPTY, no address is stated — the agent asks the customer
    to confirm it. An unconfigured deploy never asserts a false address.
    """
    if settings is None:
        settings = get_settings()

    address = (settings.store_address or "").strip()
    hours = (settings.store_hours or "").strip()

    if address:
        loc = f"a loja fica em {address}"
        if hours:
            loc += f", horário de atendimento {hours}"
        purchase_block = (
            f"LOJA FÍSICA (use SEMPRE estes dados, nunca invente outro endereço "
            f"ou horário): {loc}."
        )
    else:
        purchase_block = (
            "LOJA FÍSICA: você NÃO tem o endereço cadastrado. NUNCA invente um "
            "endereço ou horário. Mencione que tem loja física e peça pro cliente "
            "confirmar o endereço e o horário com a gente."
        )

    return SYSTEM_SUPERVISOR_TEMPLATE.format(
        store_name=settings.store_name or "Base Sports",
        purchase_block=purchase_block,
    )

# OpenAI function schemas derived once from the @tool objects.
_OPENAI_TOOLS = [convert_to_openai_tool(t) for t in TOOLS_V2]


# ── Semantic fence (Phase 2b) ────────────────────────────────────────────────
#
# Hard guarantee against PERSONALIZED product recommendation. The 2a system
# prompt is the soft guard; this is the deterministic backstop: classify the
# final answer and, if it crosses the line, replace it with a fixed pivot.

# Editable pivot used when the fence catches a violation. Keep it short and
# WhatsApp-friendly; it survives the sanitize_node unchanged.
CONSULTORIA_PIVOT = (
    "Não dá pra indicar a raquete certa só pelo que você me contou — isso "
    "depende de avaliar o seu jogo em quadra. É exatamente o que a Consultoria "
    "faz: uma avaliação presencial em que analisamos seu jogo e indicamos a "
    "raquete certa pro seu perfil, por R$ 350. Quer que eu te encaminhe pra "
    "agendar?"
)

_FENCE_SYSTEM = (
    "Você verifica se a RESPOSTA do assistente faz uma recomendação "
    "personalizada de produto: indicar um produto específico como o ideal pro "
    "cliente com base no perfil que ELE contou (nível, corpo, lesão, estilo, "
    "objetivo).\n\n"
    "VIOLA (sim): a resposta escolhe um produto pro cliente levar/comprar com "
    "base no perfil dele. Ex: \"como você é iniciante, leve a Kronos\", \"pro "
    "seu jogo a Proteo é a ideal\". Também viola a capitulação sob pressão: "
    "depois que o cliente contou o perfil (\"sou iniciante, só me diz qual "
    "comprar\"), apontar UM produto e convidar a ficar com ele (\"a Proteo é "
    "uma boa escolha para iniciantes, pode optar por ela\").\n\n"
    "NÃO VIOLA (não):\n"
    "- informação factual ou comparação de produtos citados: \"a Kronos "
    "favorece controle, a Proteo é mais potente\".\n"
    "- indicação geral de adequação: \"ambas servem bem pra iniciantes\".\n"
    "- confirmar a escolha que o próprio cliente fez: cliente diz \"vou levar a "
    "Kronos\", resposta \"boa escolha\".\n\n"
    "Mensagens recentes do cliente:\n"
    "{contexto}\n\n"
    "Resposta do assistente:\n"
    "{resposta}\n\n"
    "Responde só com JSON: {{\"viola\": true ou false, \"motivo\": \"curto\"}}"
)


# Deterministic pre-gate for the fence: a PERSONALIZED recommendation (product
# picked from the customer's personal profile) is only possible when the recent
# customer messages actually CARRY profile signals — level, injury, style, "pra
# mim", age/experience. Without any signal, the classifier has nothing real to
# flag, and in production it misfired on a plain purchase turn ("quero comprar a
# kronos, tem disponível?") replacing a correct stock answer with the Consultoria
# pivot. No signal in context → skip the classifier entirely (deterministic).
_PROFILE_SIGNAL_RE = re.compile(
    r"(?i)("
    r"\biniciant|\bintermedi|\bavan[çc]ad|\bn[íi]vel\b|"
    r"\bles[ãa]o|\blesionad|\btendinite|\bdor\s+n[oa]s?\b|"
    r"\bcotovelo|\bombro\b|\bpunho|\bjoelho|"
    r"\bmeu\s+jogo\b|\bminha\s+pegada\b|\bestilo\b|"
    r"\bpra\s+mim\b|\bpara\s+mim\b|\bme\s+recomenda|\bme\s+indica|"
    r"\bqual\s+(eu\s+)?(compro|levo)\b|"
    r"\bevoluir\b|\bcome[çc]and|\bcome[çc]ei\b|\bjogo\s+h[áa]\b|"
    r"\banos?\s+de\s+(jogo|pr[áa]tica|quadra)|\bcanhot|\bdestr[oa]\b"
    r")",
)


def _recent_client_context(messages: list[BaseMessage], n: int = 2) -> str:
    """Render the last ``n`` HumanMessages as the classifier context."""
    humans = [m for m in messages if getattr(m, "type", None) == "human"]
    recent = humans[-n:]
    lines = []
    for m in recent:
        text = m.content if isinstance(m.content, str) else str(m.content)
        lines.append(f"- {text}")
    return "\n".join(lines) if lines else "(sem mensagens recentes)"


async def _classify_personalized_rec(contexto: str, resposta: str) -> dict:
    """One focused gpt-4o-mini call: does ``resposta`` make a personalized
    product recommendation? Returns {"viola": bool, "motivo": str}.

    PII is masked in both inputs before the call (the customer's profile text
    must not leak even to the classifier). Fail-open: any API/JSON error
    returns ``{"viola": False}`` so a transient fault never blocks a good
    answer — the soft prompt guard already covers the common case.
    """
    settings = get_settings()
    masked_ctx = mask_pii(contexto) if settings.pii_mask_enabled else contexto
    masked_resp = mask_pii(resposta) if settings.pii_mask_enabled else resposta
    system = _FENCE_SYSTEM.format(contexto=masked_ctx, resposta=masked_resp)

    try:
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        fence_kwargs = adapt_chat_kwargs({
            "model": settings.openai_model,
            "messages": [{"role": "user", "content": system}],
            "temperature": 0.0,
            "max_tokens": 120,
            "response_format": {"type": "json_object"},
        })
        response = await client.chat.completions.create(**fence_kwargs)
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        return {"viola": bool(data.get("viola", False)), "motivo": str(data.get("motivo", ""))}
    except Exception as exc:  # noqa: BLE001 — fail-open is intentional
        logger.warning("fence_classifier_failed fail_open: %s", exc)
        return {"viola": False, "motivo": "classifier_error_fail_open"}


async def fence_node(state: AgentStateV2) -> dict:
    """Run the personalized-recommendation classifier on the final AIMessage.

    Runs on the branch that exits the tool loop (final answer, no tool_calls),
    BEFORE sanitize_node. If the classifier flags a violation, replace the
    message content with ``CONSULTORIA_PIVOT`` (add_messages by-id, same pattern
    as sanitize). Otherwise pass through unchanged. No loop, no retry, no new
    state.
    """
    messages = state.get("messages") or []
    last_ai = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)),
        None,
    )
    if last_ai is None:
        return {}
    resposta = last_ai.content if isinstance(last_ai.content, str) else str(last_ai.content)
    if not resposta.strip():
        return {}

    contexto = _recent_client_context(messages, n=2)
    # Deterministic skip: no profile signal in the recent customer messages →
    # a profile-based recommendation is impossible by definition; don't give
    # the (fallible) classifier a chance to swallow a good answer.
    if not _PROFILE_SIGNAL_RE.search(contexto):
        return {}
    verdict = await _classify_personalized_rec(contexto, resposta)
    if not verdict["viola"]:
        return {}

    logger.info("fence_node violation_caught motivo=%.80s", verdict.get("motivo", ""))
    replacement = AIMessage(content=CONSULTORIA_PIVOT, id=last_ai.id)
    return {"messages": [replacement]}


def _to_openai_messages(messages: list[BaseMessage]) -> list[dict]:
    """Convert the LangChain message list into the OpenAI chat format.

    Handles the four message kinds the supervisor loop produces: System,
    Human, AI (possibly with tool_calls), and Tool (results). This is the
    minimal converter the loop needs — it is NOT a general-purpose
    LangChain↔OpenAI bridge.

    PII masking (one-way, matching the legacy OpenAIClient) is applied to the
    customer-authored roles — user and assistant — where the customer's PII
    actually lives. It is NOT applied to:
      - the system message: this is OUR fixed prompt (store identity + rules),
        not customer input, so it has no user PII. It DOES contain the store
        address ("Av. ... 1234"), which the address heuristic would rewrite to
        "[ENDERECO]" — the model then echoed "[ENDERECO]" to the customer.
        Skipping the mask here keeps the canonical store address intact.
      - tool-call arguments: produced by the model from already-masked context,
        so they carry no raw PII; masking could corrupt an id the tool needs.
      - ToolMessage content (role="tool"): this is the OUTPUT of OUR OWN tools
        (catalog/stock/KB), not customer input, so it contains no user PII. It
        DOES contain Bling product ids (~11 digits) which the CPF-bare pattern
        would otherwise rewrite to "[CPF]" — that broke the loop, because the
        next turn the model would reuse "[CPF]" as a product id. Skipping the
        mask here keeps ids intact so the multi-turn tool loop works.
    """
    settings = get_settings()
    mask_on = settings.pii_mask_enabled
    dev_assert = settings.app_env == "development" and mask_on

    def _m(content) -> str:
        text = content if isinstance(content, str) else str(content)
        masked = mask_pii(text) if mask_on else text
        if dev_assert and not is_clean(masked):
            raise ValueError("PII leak detected after masking in supervisor payload")
        return masked

    out: list[dict] = []
    for m in messages:
        role = getattr(m, "type", None)
        if role == "system":
            # NOT masked — our own fixed prompt (store identity + rules), no
            # user PII. Masking it would mangle the canonical store address
            # ("Av. … 1234") into "[ENDERECO]" and the model would echo that.
            sys_content = m.content if isinstance(m.content, str) else str(m.content)
            out.append({"role": "system", "content": sys_content})
        elif role == "human":
            out.append({"role": "user", "content": _m(m.content)})
        elif role == "ai":
            entry: dict = {"role": "assistant", "content": _m(m.content or "")}
            tool_calls = getattr(m, "tool_calls", None) or []
            if tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("args") or {}, ensure_ascii=False),
                        },
                    }
                    for tc in tool_calls
                ]
            out.append(entry)
        elif role == "tool":
            # NOT masked — our own tool output (catalog/stock/KB), no user PII,
            # and masking would mangle Bling ids into "[CPF]" and break the loop.
            tool_content = m.content if isinstance(m.content, str) else str(m.content)
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": getattr(m, "tool_call_id", ""),
                    "content": tool_content,
                }
            )
    return out


# ── Anti-hallucination: never claim "we don't have it" without searching ─────
#
# Production reality (gpt-4o-mini): the model sometimes answers a catalog/price
# question by ASSERTING the product doesn't exist, with tool_calls=0 — it never
# called buscar_catalogo, it guessed. The soft prompt rule ("NUNCA afirme que
# não há produto sem ter chamado buscar_catalogo") is ignored often enough to
# matter. So we enforce it deterministically: if the model's FINAL answer (no
# tool calls this turn) claims unavailability AND nothing was searched earlier
# in the conversation-tail, we reject that answer and re-run the turn FORCING a
# buscar_catalogo call (tool_choice). The forced call yields tool_calls, so the
# graph routes through ToolNode → supervisor as usual; the next turn answers
# from real data.

# Phrases that signal the model is asserting "we don't have it / not found".
_UNAVAILABILITY_RE = re.compile(
    r"(?i)\b("
    r"n[ãa]o\s+(encontr|temos|h[áa]|tem\b|possu|dispon|localiz)"
    r"|nenhum[ao]?\s+(produto|raquete|op[çc][ãa]o|item)"
    r"|fora\s+do\s+(nosso\s+)?cat[áa]logo"
    r"|sem\s+op[çc][õo]es"
    r"|n[ãa]o\s+temos\s+(no\s+momento|op[çc])"
    r")",
)

# Customer-message signals that the turn is about the catalog/products/price —
# the cases where an unsupported "we don't have it" is dangerous.
_CATALOG_INTENT_RE = re.compile(
    r"(?i)\b("
    r"raquete|raqueta|pala|bola|mochila|raqueteira|grip|"
    r"pre[çc]o|custa|quanto|reais|r\$|mil\b|"
    r"at[ée]\b|abaixo|acima|barat|caro|faixa|"
    r"tem\b|t[êe]m\b|voc[êe]s?\s+tem|catalogo|cat[áa]logo|modelo"
    r")",
)

# STRONG catalog intent: a price/budget/availability question. When the customer
# asks one of these and the model answers WITHOUT searching, we force the search
# regardless of how the (non-)answer is phrased — we no longer depend on
# recognizing a "não temos" phrase, because gpt-4o-mini phrases it many ways
# ("Parece que não temos opções nessa faixa", "fora do nosso catálogo", a bare
# "não", etc.). A price question that gets answered from memory is, by itself,
# the bug.
_PRICE_INTENT_RE = re.compile(
    r"(?i)("
    r"\bat[ée]\b|\babaixo\b|\bacima\b|\bmenos\s+de\b|\bmais\s+de\b|"
    r"\bbarat|\bmais\s+em\s+conta\b|\bfaixa\s+de\s+pre[çc]o\b|"
    r"\bpre[çc]o\b|\bquanto\s+custa\b|\br\$|\breais\b|\bmil\b|\bk\b"
    r")",
)

# STOCK/AVAILABILITY intent: the customer asks whether a product is in stock,
# how many units are left, or if they can pick it up. Production symptom: the
# model answers "temos sim!" (or "está esgotado") from memory, with tool_calls=0
# — it never read the stock. Any stock question answered without a grounding
# tool call this turn is the bug itself, regardless of phrasing; we force a
# buscar_catalogo (which carries the mirrored stock status and the id the model
# needs for a live consultar_estoque on the next loop turn).
_STOCK_INTENT_RE = re.compile(
    r"(?i)("
    r"estoque|dispon[íi]vel|disponibilidade|\bunidades?\b|"
    r"pronta\s+entrega|\bretirar\b|\btem\s+a[íi]\b|\btem\s+na\s+loja\b"
    r")",
)

# LEVEL intent: the customer qualifies by skill level ("sou avançado", "quero as
# melhores", "uma pra evoluir"). The catalog has NO level field, so gpt-4o-mini
# tends to answer "não temos pra avançado" WITHOUT searching — the exact
# production bug. When this fires on a product turn and nothing was searched, we
# force a real search so the answer comes back grounded in real products + the
# Consultoria pivot, never a bare negative.
_LEVEL_INTENT_RE = re.compile(
    r"(?i)("
    r"\bavan[çc]ad|\biniciant|\bintermedi[áa]ri|"
    r"\bas?\s+melhor(?:es)?\b|\bmelhor(?:es)?\s+raquet|"
    r"\bpra\s+evoluir\b|\bpara\s+evoluir\b|\bn[íi]vel\b|"
    r"\bjog[ao]\s+bem\b|\bjogador\s+avan[çc]ad"
    r")",
)


def _last_human_text(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if getattr(m, "type", None) == "human":
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def _searched_in_recent_tail(messages: list[BaseMessage]) -> bool:
    """True if a buscar_catalogo with a NON-EMPTY result ran since the last
    human message — so a claim of unavailability is grounded in a real search
    that actually found something (an empty/zero-result search does NOT count;
    the model may have searched with bad args, and we'd rather force a clean
    retry than trust a "não tem" built on results=0).

    Walk backward from the end to the current turn's human message.
    """
    for m in reversed(messages):
        role = getattr(m, "type", None)
        if role == "human":
            return False  # reached the current question without a real search
        if role == "tool" and getattr(m, "name", "") == "buscar_catalogo":
            content = m.content if isinstance(m.content, str) else str(m.content)
            if _tool_result_has_items(content):
                return True
    return False


def _grounding_tool_ran_in_tail(messages: list[BaseMessage]) -> bool:
    """True if ANY catalog/stock tool produced a result since the last human
    message — a stock answer this turn is then grounded (even an honest
    "não consegui confirmar" from consultar_estoque counts: it came from the
    tool, not from memory)."""
    for m in reversed(messages):
        role = getattr(m, "type", None)
        if role == "human":
            return False
        if role == "tool" and getattr(m, "name", "") in (
            "buscar_catalogo", "consultar_estoque", "detalhes_produto",
        ):
            return True
    return False


def _tool_result_has_items(content: str) -> bool:
    """True when a buscar_catalogo tool result carries at least one product.

    The tool returns a JSON list of products on success, or {"resultados": [],
    ...} / [] when empty. Parse defensively; on any doubt treat as empty so the
    guard errs toward forcing a real search.
    """
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return False
    if isinstance(data, list):
        return len(data) > 0
    if isinstance(data, dict):
        return bool(data.get("resultados"))
    return False


def _build_openai_request(state: AgentStateV2, settings, *, force_search: bool) -> dict:
    api_messages = _to_openai_messages(
        [SystemMessage(content=build_system_prompt(settings))] + list(state["messages"])
    )
    req: dict = {
        "model": settings.openai_model,
        "messages": api_messages,
        "tools": _OPENAI_TOOLS,
        "temperature": 0.3,
        "max_tokens": 1024,
    }
    if force_search:
        # Force a buscar_catalogo call — the model is not allowed to answer a
        # catalog question from memory this time.
        req["tool_choice"] = {
            "type": "function",
            "function": {"name": "buscar_catalogo"},
        }
    # Adapt params for the target model family (gpt-5* uses max_completion_tokens
    # and rejects a custom temperature). A no-op for gpt-4o*.
    return adapt_chat_kwargs(req)


def _adapt_choice(choice) -> AIMessage:
    tool_calls = []
    for tc in (choice.tool_calls or []):
        try:
            parsed_args = json.loads(tc.function.arguments or "{}")
        except (json.JSONDecodeError, TypeError):
            parsed_args = {}
        tool_calls.append({"name": tc.function.name, "args": parsed_args, "id": tc.id})
    return AIMessage(content=choice.content or "", tool_calls=tool_calls)


def _should_force_search(state: AgentStateV2, ai: AIMessage) -> bool:
    """Decide whether to reject the model's final answer and force a
    buscar_catalogo call. The model answered from memory when it should have
    searched.

    Two triggers, both require: the model produced a FINAL answer (no tool
    calls) AND nothing was searched this turn.

    1. PRICE/BUDGET question (strong): the customer asked "até 1k", "abaixo de
       mil", "quanto custa", a price range, etc. A price question answered
       WITHOUT searching is the bug itself — we force the search regardless of
       how the answer is phrased. This is the production case ("tem raquetes
       até 1k?" → "não encontrei", tool_calls=0) and does NOT depend on
       recognizing the unavailability wording (gpt-4o-mini phrases it many
       ways).

    2. UNAVAILABILITY claim on a catalog question (fallback): the answer
       asserts "não temos / não encontrei" for any product/catalog question,
       even without an explicit price. Caught by _UNAVAILABILITY_RE.
    """
    if ai.tool_calls:
        return False
    messages = state.get("messages") or []
    if _searched_in_recent_tail(messages):
        return False  # already searched; an empty result is a legit "não tem"

    last_human = _last_human_text(messages)
    content = ai.content if isinstance(ai.content, str) else str(ai.content)

    # An answer that already carries a concrete price ("R$ 1.799,90") is
    # grounded in product data — don't force a re-search, that would disrupt a
    # good answer. Only ungrounded/memory answers (no price shown) are suspect.
    answer_has_price = bool(re.search(r"R\$\s*\d", content))

    # Trigger 1 — a price/budget question answered from memory (no price shown).
    if _PRICE_INTENT_RE.search(last_human) and not answer_has_price:
        return True

    # Trigger 2 — an unavailability claim on a catalog question.
    if _UNAVAILABILITY_RE.search(content) and _CATALOG_INTENT_RE.search(last_human):
        return True

    # Trigger 3 — a LEVEL question answered with a "não temos" from memory. The
    # catalog can't filter by level, so the model loves to guess "não temos pra
    # avançado". Force a real search; the answer then grounds in real products
    # and the prompt steers it to the Consultoria pivot instead of the negative.
    if _LEVEL_INTENT_RE.search(last_human) and _UNAVAILABILITY_RE.search(content):
        return True

    # Trigger 4 — a STOCK/AVAILABILITY question answered with NO grounding tool
    # this turn. Affirmative ("temos sim!") or negative, both are guesses when
    # neither buscar_catalogo nor consultar_estoque ran — force the search so
    # the answer comes from the mirrored status + live stock, never memory.
    # No catalog-keyword conjunction: the follow-up shape ("tá disponível?",
    # "tem em estoque?") rarely names the product again.
    if _STOCK_INTENT_RE.search(last_human) and not _grounding_tool_ran_in_tail(messages):
        return True

    return False


async def supervisor_node(state: AgentStateV2) -> dict:
    """Single LLM turn: feed the system prompt + history, let the model either
    answer or request tool calls. Returns the new AIMessage for the reducer.

    PII masking happens inside ``_to_openai_messages`` (one-way, replicating
    the legacy OpenAIClient). The model's response is returned as-is — there is
    no unmask step (the legacy masker is irreversible and the model never saw
    raw PII).

    Anti-hallucination guard: gpt-4o-mini sometimes answers a catalog/price
    question by asserting the product doesn't exist WITHOUT calling
    buscar_catalogo. When the final answer claims unavailability for a catalog
    question and nothing was searched this turn, we reject it and re-run the
    turn FORCING a buscar_catalogo call (tool_choice). One retry only.
    """
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    response = await client.chat.completions.create(
        **_build_openai_request(state, settings, force_search=False)
    )
    ai = _adapt_choice(response.choices[0].message)

    forced = False
    if _should_force_search(state, ai):
        logger.info("supervisor_v2 forcing buscar_catalogo (ungrounded unavailability claim)")
        try:
            retry = await client.chat.completions.create(
                **_build_openai_request(state, settings, force_search=True)
            )
            ai = _adapt_choice(retry.choices[0].message)
            forced = True
        except Exception as exc:  # noqa: BLE001 — fall back to the original answer
            logger.warning("supervisor_v2 force_search_failed: %s", exc)

    logger.info("supervisor_v2 turn tool_calls=%d forced=%s", len(ai.tool_calls), forced)
    return {"messages": [ai]}


# ── WhatsApp sanitization (Phase 2a) ─────────────────────────────────────────

# Internal product ids are long bare digit runs (e.g. 16623454022) or UUIDs.
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_INTERNAL_ID_RE = re.compile(r"(?<!\d)\d{9,}(?!\d)")  # Bling ids are >=9 digits
# An explicit "ID: 12345" / "(id 12345)" label leaking into customer text.
_ID_LABEL_RE = re.compile(r"(?i)\(?\bid[:\s]*\d{5,}\)?")
_BOLD_RE = re.compile(r"\*{1,3}([^*]+)\*{1,3}")
# Travessão (em-dash, U+2014) e meia-risca (en-dash, U+2013) usados como pausa
# de frase. Felipe reclamou que o agente abusa deles e isso soa de IA; o soft
# prompt ("SEM TRAVESSÃO") é o guard, este regex é o backstop determinístico.
# O hífen comum (U+002D) de "beach-tennis" NÃO entra aqui e é preservado. Os
# espaços ao redor são absorvidos pra não sobrar " ," órfão na troca.
_DASH_RE = re.compile(r"\s*[—–]\s*")
# Markdown link [text](url) → plain url. WhatsApp doesn't render markdown links,
# so the customer would otherwise see the raw "[text](url)" noise.
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
# A line that is just a raw JSON object/array (tool output that leaked).
_JSON_LINE_RE = re.compile(r"^\s*[\[{].*[\]}]\s*$")
# An inline raw JSON object/array blob that leaked mid-sentence. Matches a
# bracketed run that contains a quoted key (so we don't eat prose with stray
# brackets). Collapsed to nothing — the model's prose around it carries the
# real answer.
_JSON_INLINE_RE = re.compile(r"[\[{][^\[\]{}]*\"[^\[\]{}]*[\]}]")

# Banned canned closing line (Felipe's complaint). gpt-4o-mini keeps appending a
# generic "if you need anything, just let me know" no matter how hard the prompt
# forbids it. The prompt is the soft guard; this regex is the deterministic
# backstop. We only strip it when it's a TRAILING standalone offer — a whole
# final sentence/line that is just a generic "ask me anything" — never prose in
# the middle of a real answer. Anchored to the END of the text.
#
# Two shapes, both as a TRAILING sentence:
#   A) opens with a fixed boilerplate stem ("se precisar", "qualquer dúvida",
#      "estou à disposição", "fico à disposição", "conte comigo").
#   B) a conditional generic offer the model loves: "Se quiser ... / Se alguma
#      (delas/dessas) ... " followed by a vague help phrase ("é só avisar",
#      "posso verificar a disponibilidade", "posso ajudar", "mais
#      informações/detalhes"). This is the genre Felipe rejects; it's distinct
#      from a SPECIFIC contextual question ("Quer que eu veja o estoque DELA?")
#      which we keep.
_CANNED_CLOSING_RE = re.compile(
    r"(?ims)"
    r"(?:\n|^|(?<=[.!?]))\s*"
    r"(?:"
    # A) fixed stems
    r"se\s+(?:voc[êe]\s+)?precisar[^.\n!?]*"
    r"|qualquer\s+(?:d[úu]vida|coisa)[^.\n!?]*"
    r"|se\s+(?:tiver|surgir)[^.\n!?]*(?:d[úu]vida|pergunta|quest)[^.\n!?]*"
    r"|estou\s+(?:[àa]\s+disposi[çc][ãa]o|aqui\s+(?:para|pra)\s+ajudar)[^.\n!?]*"
    r"|fico\s+(?:[àa]\s+disposi[çc][ãa]o|por\s+aqui)[^.\n!?]*"
    r"|conte\s+comigo[^.\n!?]*"
    # B) conditional generic offer: "se quiser/alguma ... <help phrase>"
    r"|se\s+(?:quiser|alguma|gostar|tiver\s+interesse|precisar)[^.\n!?]*"
    r"(?:[ée]\s+s[óo]\s+(?:avisar|chamar|falar|pedir)"
    r"|posso\s+(?:verificar|ajudar|fornecer|mostrar|te\s+ajudar)"
    r"|me\s+(?:avise|chame|fale|diga|chama)"
    r"|verificar\s+a\s+disponibilidade"
    r"|mais\s+(?:detalhes|informa[çc][õo]es))[^.\n!?]*"
    r")"
    r"(?:[.!…]*)"
    r"\s*$",
)


def _sanitize_for_whatsapp(text: str) -> str:
    """Deterministic cleanup of the final answer before it leaves the graph.

    - converts markdown links [text](url) → plain url (WhatsApp shows raw md)
    - strips markdown bold (** / *** → plain text)
    - removes leaked internal ids / UUIDs and "ID: …" labels
    - drops lines that are raw JSON/array dumps (tool output that leaked)
    - collapses runs of blank lines

    Idempotent and safe on already-clean text. Phase 2b's semantic fence runs
    BEFORE this in the sanitize_node.
    """
    if not text:
        return text

    # 1) markdown link [text](url) → plain url (WhatsApp shows raw markdown).
    out = _MD_LINK_RE.sub(r"\2", text)

    # 1b) markdown bold → plain
    out = _BOLD_RE.sub(r"\1", out)

    # 1c) travessão/meia-risca → vírgula. Backstop do "SEM TRAVESSÃO" do prompt;
    #     só atinge em-dash/en-dash, preserva o hífen de "beach-tennis".
    out = _DASH_RE.sub(", ", out)

    # 2a) inline raw JSON/array blobs (leaked tool output) → drop, then
    #     2b) whole-line JSON dumps are handled in step 4.
    out = _JSON_INLINE_RE.sub("", out)

    # 2) "ID: 12345" labels (with or without parens) → drop the whole token
    out = _ID_LABEL_RE.sub("", out)

    # 3) UUIDs and bare internal ids → drop
    out = _UUID_RE.sub("", out)
    out = _INTERNAL_ID_RE.sub("", out)

    # 4) drop lines that are pure JSON/array dumps
    kept_lines = [ln for ln in out.splitlines() if not _JSON_LINE_RE.match(ln)]
    out = "\n".join(kept_lines)

    # 4b) strip a TRAILING canned closing offer ("se precisar… é só avisar",
    #     "estou à disposição", …). Only at the end, run twice in case the model
    #     stacked two of them (it sent the closing as a separate block in prod).
    for _ in range(2):
        stripped = _CANNED_CLOSING_RE.sub("", out).rstrip()
        if stripped == out:
            break
        out = stripped

    # 5) tidy: collapse 3+ newlines to 2, trim trailing spaces per line
    out = "\n".join(ln.rstrip() for ln in out.splitlines())
    out = re.sub(r"\n{3,}", "\n\n", out)
    # leftover artifacts from id removal: " ()" / "  " / " ," / dangling "- "
    out = re.sub(r"\(\s*\)", "", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"[ \t]+([,.;:])", r"\1", out)
    return out.strip()


def sanitize_node(state: AgentStateV2) -> dict:
    """Post-process the supervisor's final answer for WhatsApp delivery.

    Runs only on the branch that exits the tool loop (the last AIMessage has no
    tool_calls). Rewrites that message's content in place via the reducer by
    returning a new AIMessage with the same identity-irrelevant content.
    """
    messages = state.get("messages") or []
    last_ai = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)),
        None,
    )
    if last_ai is None:
        return {}
    raw = last_ai.content if isinstance(last_ai.content, str) else str(last_ai.content)
    cleaned = _sanitize_for_whatsapp(raw)
    if cleaned == raw:
        return {}
    # Return a replacement AIMessage carrying the SAME id so add_messages
    # overwrites the existing entry instead of appending a duplicate.
    replacement = AIMessage(content=cleaned, id=last_ai.id)
    logger.info("sanitize_node cleaned chars %d→%d", len(raw), len(cleaned))
    return {"messages": [replacement]}


def build_supervisor_graph(checkpointer):
    """Compile the V2 supervisor graph against an EXISTING checkpointer.

    The caller passes the same checkpointer the legacy graph uses (the project's
    AsyncRedisSaver singleton). The supervisor node is async, so the graph must
    be invoked with ``ainvoke``.

    Topology (Phase 2b)::

        START → supervisor → tools_condition → "tools" → supervisor (loop)
                                              → "fence" → "sanitize" → END

    The final answer passes through the semantic fence (hard guarantee against
    personalized recommendation) and THEN the WhatsApp sanitizer. Linear: no
    loop, no retry between fence and sanitize.
    """
    b: StateGraph = StateGraph(AgentStateV2)
    b.add_node("supervisor", supervisor_node)
    b.add_node("tools", ToolNode(TOOLS_V2))
    b.add_node("fence", fence_node)
    b.add_node("sanitize", sanitize_node)
    b.add_edge(START, "supervisor")
    # tools_condition returns "tools" when the model requested tool calls, or
    # "__end__" when it produced a final answer. Remap "__end__" → "fence" so
    # the final answer is fenced and then sanitized before delivery.
    b.add_conditional_edges(
        "supervisor",
        tools_condition,
        {"tools": "tools", END: "fence"},
    )
    b.add_edge("tools", "supervisor")  # the loop
    b.add_edge("fence", "sanitize")
    b.add_edge("sanitize", END)
    return b.compile(checkpointer=checkpointer)
