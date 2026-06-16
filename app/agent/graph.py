"""Sprint 2.6 — simplified LangGraph topology.

The strategic refactor removed the ``diagnose`` node from the live graph.
Triage now picks one of 9 declarative intents and the router maps each to
a single node. No more determined/exploring forks, no post-recommendation
state machine, no rerun branches.

Topology::

    START → triage → (intent) → smalltalk
                              → product_inquiry  → recommend
                              → price_inquiry    → price_inquiry
                              → purchase_intent  → product_selection
                              → scheduling_inquiry → scheduling_inquiry
                              → out_of_scope     → out_of_scope_handoff
                              → faq              → faq
                              → help_request     → help_request
                              → close            → smalltalk (graceful goodbye)

Every leaf node → END. No multi-hop routing, no recommend → handoff edge
(recommend is now a straight Q&A response; explicit handoff intents have
their own intent label).
"""
import json
import logging

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.adapters.openai_client import OpenAIClient
from app.agent.checkpointer import get_checkpointer
from app.agent.nodes.attribute_inquiry import attribute_inquiry_node
from app.agent.nodes.faq import faq_node
from app.agent.nodes.handoff import handoff_node
from app.agent.nodes.help_request import help_request_node
from app.agent.nodes.out_of_scope_handoff import out_of_scope_handoff_node
from app.agent.nodes.price_inquiry import price_inquiry_node
from app.agent.nodes.product_selection import product_selection_node
from app.agent.nodes.recommend import recommend_node
from app.agent.nodes.scheduling_inquiry import scheduling_inquiry_node
from app.agent.nodes.triage import recent_chat_history, triage_node
from app.agent.prompts import SYSTEM_NAME_EXTRACT, SYSTEM_SMALLTALK
from app.agent.state import AgentState

logger = logging.getLogger(__name__)


async def _smalltalk_node(state: AgentState) -> dict:
    """Smalltalk node — handles the name capture flow (3 phases) and the
    Sprint 2.4 canned brand greeting on the first interaction. Also serves
    as the graceful-goodbye destination when ``goodbye_pending`` is set.
    """
    messages = state.get("messages") or []
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)

    raw_content = last_human.content if last_human else ""
    content = raw_content if isinstance(raw_content, str) else str(raw_content)

    customer_name = state.get("customer_name")
    name_asked = bool(state.get("name_asked", False))

    client = OpenAIClient()
    update: dict = {}

    # Graceful goodbye after declining an offer.
    if state.get("goodbye_pending"):
        goodbye = "Tudo bem! Se mudar de ideia, é só me chamar."
        logger.info("smalltalk goodbye_pending → canned reply")
        return {
            "messages": [AIMessage(content=goodbye)],
            "response_blocks": [goodbye],
            "goodbye_pending": False,
        }

    # Sprint 2.6.2 — customer rejected our "Você quis dizer X?" suggestion.
    # Ask for the correct name and stop (no LLM call).
    if state.get("match_decline_pending"):
        decline = "Sem problemas! Pode me dar mais detalhes do que você procura?"
        logger.info("smalltalk match_decline_pending → canned reply")
        return {
            "messages": [AIMessage(content=decline)],
            "response_blocks": [decline],
            "match_decline_pending": False,
        }

    # Phase 1: name extraction.
    if name_asked and not customer_name:
        try:
            ext_raw = await client.chat(
                messages=[{"role": "user", "content": content}],
                system=SYSTEM_NAME_EXTRACT,
                max_tokens=50,
                temperature=0.0,
                json_mode=True,
            )
            extracted = (json.loads(ext_raw or "{}").get("extracted_name") or "")
            extracted = str(extracted).strip() if extracted else ""
            if extracted:
                customer_name = extracted
                update["customer_name"] = customer_name
                logger.info("smalltalk name_captured")
        except (json.JSONDecodeError, ValueError, AttributeError) as exc:
            logger.warning("smalltalk name_extract_failed: %s", exc)
        update["name_asked"] = False

    # Phase 2: first interaction → ask name (canned brand greeting).
    if not customer_name and not name_asked:
        ask_text = (
            "Olá! Bem-vindo à Base Sports 👋 Antes de tudo, qual seu nome?"
        )
        update["messages"] = [AIMessage(content=ask_text)]
        update["response_blocks"] = [ask_text]
        update["name_asked"] = True
        logger.info("smalltalk name_ask sent (brand canned)")
        return update

    # Phase 3: normal smalltalk reply.
    # Sprint 2.7.1 — pass the recent history so the LLM can respond
    # contextually instead of always producing the generic "E aí, Felipe!"
    # greeting. The name (if known) is prepended to the LAST user message
    # only, so the model sees a clean conversation transcript plus the
    # name hint where it matters.
    history = recent_chat_history(messages, window=4)
    if not history:
        # First-turn / no usable history — fall back to the legacy single
        # user-block format (same shape pre-2.7.1).
        user_block = (
            f"Nome do cliente: {customer_name}\n\n{content}"
            if customer_name else content
        )
        chat_messages: list[dict[str, str]] = [{"role": "user", "content": user_block}]
    else:
        # Inject the name hint into the LAST user turn so it doesn't get
        # echoed into a previous user message.
        if customer_name and history and history[-1]["role"] == "user":
            history[-1] = {
                "role": "user",
                "content": f"Nome do cliente: {customer_name}\n\n{history[-1]['content']}",
            }
        chat_messages = history

    response = await client.chat(
        messages=chat_messages,
        system=SYSTEM_SMALLTALK,
        max_tokens=150,
        temperature=0.7,
    )
    text = (response or "").strip()
    update["messages"] = [AIMessage(content=text)]
    update["response_blocks"] = [text]
    return update


# Sprint 2.6 — declarative intent → node mapping. Any intent not in this map
# falls through to smalltalk (safe default).
_INTENT_TO_NODE: dict[str, str] = {
    "smalltalk":          "smalltalk",
    "product_inquiry":    "recommend",
    "price_inquiry":      "price_inquiry",
    "purchase_intent":    "product_selection",
    "scheduling_inquiry": "scheduling_inquiry",
    "out_of_scope":       "out_of_scope",
    "faq":                "faq",
    "help_request":       "help_request",
    # Sprint 2.6.6 — pergunta sobre característica do produto ativo.
    "attribute_inquiry":  "attribute_inquiry",
    "close":              "smalltalk",  # graceful close goes through smalltalk
    # Legacy compatibility (state checkpointed from older sprints):
    "handoff":            "handoff",
}


def _triage_router(state: AgentState) -> str:
    intent = state.get("intent") or "smalltalk"
    target = _INTENT_TO_NODE.get(intent, "smalltalk")
    logger.info("triage_router intent=%s → node=%s", intent, target)
    return target


def build_graph(checkpointer=None) -> CompiledStateGraph:
    """Compile the Sprint 2.6 graph.

    Sprint 2.6 removed: diagnose, re_recommendation, product_detail (folded
    into recommend), ambiguous_selection, close, consultoria pitch node,
    legacy recommend post-rec branches. Anything that survives in
    ``app/agent/nodes/`` but isn't registered below is dead code kept for
    historical reference (e.g. diagnose.py — see its module docstring).
    """
    builder: StateGraph = StateGraph(AgentState)

    builder.add_node("triage", triage_node)
    builder.add_node("smalltalk", _smalltalk_node)
    builder.add_node("recommend", recommend_node)
    builder.add_node("price_inquiry", price_inquiry_node)
    builder.add_node("product_selection", product_selection_node)
    builder.add_node("scheduling_inquiry", scheduling_inquiry_node)
    builder.add_node("out_of_scope", out_of_scope_handoff_node)
    builder.add_node("faq", faq_node)
    builder.add_node("help_request", help_request_node)
    builder.add_node("attribute_inquiry", attribute_inquiry_node)
    builder.add_node("handoff", handoff_node)

    builder.add_edge(START, "triage")

    builder.add_conditional_edges(
        "triage",
        _triage_router,
        {
            "smalltalk":          "smalltalk",
            "recommend":          "recommend",
            "price_inquiry":      "price_inquiry",
            "product_selection":  "product_selection",
            "scheduling_inquiry": "scheduling_inquiry",
            "out_of_scope":       "out_of_scope",
            "faq":                "faq",
            "help_request":       "help_request",
            "attribute_inquiry":  "attribute_inquiry",
            "handoff":            "handoff",
        },
    )

    for leaf in (
        "smalltalk", "recommend", "price_inquiry", "product_selection",
        "scheduling_inquiry", "out_of_scope", "faq", "help_request",
        "attribute_inquiry", "handoff",
    ):
        builder.add_edge(leaf, END)

    if checkpointer is None:
        checkpointer = get_checkpointer()
    return builder.compile(checkpointer=checkpointer)
