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
    "Você é o assistente virtual da {store_name}, loja especializada em Beach "
    "Tennis e Padel. Você atende clientes pelo WhatsApp, em português, de forma "
    "direta e cordial, sem enrolação.\n\n"
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
    "mil\", \"abaixo de mil\", \"entre 1000 e 1500\") ou pelas \"mais baratas\", "
    "chame buscar_catalogo com preco_min/preco_max e/ou ordenacao=\"preco_asc\", "
    "E SEMPRE passe a categoria certa (ex: categoria=\"beach tennis\" quando ele "
    "fala de raquete de beach tennis), pra não misturar mochila, acessório ou "
    "raquete de outro esporte. NUNCA afirme que não há produto numa faixa de "
    "preço sem ter chamado buscar_catalogo com o filtro de preço E a categoria.\n"
    "FAIXA VAZIA: se a busca com preço+categoria não retornar nada na faixa "
    "pedida, NÃO despeje produtos fora da faixa. Diga, de forma natural, que não "
    "há naquele preço e informe a opção mais em conta daquela categoria, com o "
    "preço, e ofereça mostrar. Ex: \"Não temos raquete de beach tennis abaixo de "
    "R$ 1.000. A mais em conta é a Fulana, a R$ 1.299. Quer ver ela?\" — pra "
    "achar essa mais barata, refaça a busca só com a categoria e "
    "ordenacao=\"preco_asc\", sem a faixa.\n\n"
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
    "personalizada, não pode.\n\n"
    "A CONSULTORIA\n"
    "Avaliação presencial em que analisamos o jogo do cliente em quadra e "
    "indicamos a raquete certa pro perfil dele. Valor R$ 350, 100% abatido na "
    "compra de uma raquete. Você NÃO tem os detalhes de quem conduz, como "
    "agendar ou duração. Para esses, acione o atendimento humano "
    "(escalar_humano), nunca invente.\n\n"
    "COMO COMPRAR\n"
    "A {store_name} vende em DOIS canais: online pelo e-commerce (com PIX) e na "
    "loja física. Quando o cliente quiser comprar, ofereça as duas opções: "
    "confirme o produto, se for útil cheque o estoque, e apresente os dois "
    "caminhos. Não acione atendente só por causa de compra, é só orientar.\n"
    "{purchase_block}\n\n"
    "QUANDO ACIONAR ATENDENTE (escalar_humano)\n"
    "Quando o cliente pedir explicitamente falar com uma pessoa, quando a dúvida "
    "for genuinamente fora do escopo de produtos e da loja, ou para encaminhar o "
    "agendamento ou fechamento da Consultoria. Não acione atendente só porque "
    "uma busca de produto voltou vazia; nesse caso peça o nome ou ofereça listar "
    "opções.\n\n"
    "ESTILO (WhatsApp)\n"
    "Mensagens curtas e diretas. Sem markdown, sem asteriscos de negrito, sem "
    "tabelas. Nunca mostre códigos ou ids internos de produto; refira-se aos "
    "produtos pelo nome.\n"
    "TOM HUMANO: NUNCA termine com uma frase de fechamento padronizada. É "
    "PROIBIDO encerrar com \"Se precisar de mais informações ou ajuda, é só "
    "avisar!\", \"Se alguma dessas opções te interessar, posso verificar...\", "
    "\"Estou à disposição\" ou qualquer variação fixa colada no fim — isso soa "
    "robô e o Felipe não quer. Encerre naturalmente depois de responder; "
    "ofereça o próximo passo só quando fizer sentido, sempre com palavras "
    "diferentes. Soe como uma pessoa que entende de raquete conversando, não "
    "como um sistema.\n"
    "REAJA AO QUE MOSTRA: ao apresentar produtos, NÃO cuspa uma lista numerada "
    "longa seguida de um convite genérico. Quando a busca traz muitos itens "
    "parecidos, mostre uns 2 ou 3 (os mais relevantes ou os mais em conta), não "
    "os oito. Faça um comentário curto e útil sobre o que mostrou (\"a partir de "
    "R$ 449 já tem opção de iniciante\", \"essas Drop Shot são todas de linha "
    "iniciante\", \"as duas são de carbono\") e, quando fizer sentido, devolva "
    "UMA pergunta pra guiar (faixa de preço, nível, o que ele procura). "
    "Apresente como quem selecionou os produtos, não como quem repassou uma "
    "busca: NUNCA diga \"as opções que apareceram\", \"resultados da busca\" ou "
    "parecido. Continue enxuto pro WhatsApp — comentário curto, não textão.\n"
    "AO LISTAR produtos ou specs, formate de forma limpa e legível pro WhatsApp "
    "(uma linha por item), e não grude o comentário ou a pergunta na mesma linha "
    "do último produto."
)


def build_system_prompt(settings=None) -> str:
    """Render the system prompt with the store/e-commerce identity from Settings.

    The purchase block describes BOTH channels (online e-commerce with PIX, and
    the physical store). Two safety rules, same shape as before:

    - store address: pinned from settings when configured; when EMPTY, no
      address is stated — the agent asks the customer to confirm it.
    - e-commerce url: pinned when configured; when EMPTY, the e-commerce is
      still mentioned but WITHOUT a (broken/invented) link — the agent asks the
      customer to confirm the link. An unconfigured deploy never asserts a false
      address or a false URL.
    """
    if settings is None:
        settings = get_settings()

    address = (settings.store_address or "").strip()
    hours = (settings.store_hours or "").strip()
    ecommerce = (settings.ecommerce_url or "").strip()

    # ── Online channel ───────────────────────────────────────────────────────
    if ecommerce:
        online = (
            f"ONLINE: o e-commerce da loja é {ecommerce}. Lá dá pra comprar com "
            "PIX e ganhar 5% de desconto no PIX."
        )
    else:
        online = (
            "ONLINE: a loja TEM e-commerce com pagamento via PIX (5% de desconto "
            "no PIX), mas você NÃO tem o link cadastrado. NUNCA invente uma URL. "
            "Mencione que dá pra comprar online e peça pro cliente confirmar o "
            "link do e-commerce com a gente."
        )

    # ── Physical channel ─────────────────────────────────────────────────────
    if address:
        loc = f"a loja fica em {address}"
        if hours:
            loc += f", horário de atendimento {hours}"
        physical = (
            f"LOJA FÍSICA (use SEMPRE estes dados, nunca invente outro endereço "
            f"ou horário): {loc}."
        )
    else:
        physical = (
            "LOJA FÍSICA: você NÃO tem o endereço cadastrado. NUNCA invente um "
            "endereço ou horário. Mencione que tem loja física e peça pro cliente "
            "confirmar o endereço e o horário com a gente."
        )

    purchase_block = online + "\n" + physical

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
    "seu jogo a Proteo é a ideal\".\n\n"
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
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": system}],
            temperature=0.0,
            max_tokens=120,
            response_format={"type": "json_object"},
        )
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


async def supervisor_node(state: AgentStateV2) -> dict:
    """Single LLM turn: feed the system prompt + history, let the model either
    answer or request tool calls. Returns the new AIMessage for the reducer.

    PII masking happens inside ``_to_openai_messages`` (one-way, replicating
    the legacy OpenAIClient). The model's response is returned as-is — there is
    no unmask step (the legacy masker is irreversible and the model never saw
    raw PII).
    """
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    api_messages = _to_openai_messages(
        [SystemMessage(content=build_system_prompt(settings))] + list(state["messages"])
    )

    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=api_messages,
        tools=_OPENAI_TOOLS,
        temperature=0.3,
        max_tokens=1024,
    )
    choice = response.choices[0].message

    # Adapt the OpenAI response into a LangChain AIMessage. ToolNode and
    # tools_condition key off AIMessage.tool_calls (list of {name,args,id}).
    tool_calls = []
    for tc in (choice.tool_calls or []):
        try:
            parsed_args = json.loads(tc.function.arguments or "{}")
        except (json.JSONDecodeError, TypeError):
            parsed_args = {}
        tool_calls.append({"name": tc.function.name, "args": parsed_args, "id": tc.id})

    ai = AIMessage(content=choice.content or "", tool_calls=tool_calls)
    logger.info("supervisor_v2 turn tool_calls=%d", len(tool_calls))
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
