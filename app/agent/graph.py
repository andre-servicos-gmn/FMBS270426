"""LangGraph agent graph for the Beach Tennis / Padel franchise."""
import json
import logging

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.adapters.openai_client import OpenAIClient
from app.agent.checkpointer import get_checkpointer
from app.agent.nodes._positional_reference import detect_positional_reference
from app.agent.nodes._product_match import match_product_tolerant
from app.agent.nodes._pronominal_reference import detect_pronominal_reference
from app.agent.nodes.ambiguous_selection import ambiguous_selection_node
from app.agent.nodes.close import close_node
from app.agent.nodes.diagnose import diagnose_node
from app.agent.nodes.faq import faq_node
from app.agent.nodes.handoff import handoff_node
from app.agent.nodes.out_of_scope_handoff import out_of_scope_handoff_node
from app.agent.nodes.pitch_consultoria import pitch_consultoria_node
from app.agent.nodes.price_inquiry import price_inquiry_node
from app.agent.nodes.product_detail import product_detail_node
from app.agent.nodes.product_selection import product_selection_node
from app.agent.nodes.re_recommendation import re_recommendation_node
from app.agent.nodes.recommend import recommend_node
from app.agent.nodes.scheduling_inquiry import scheduling_inquiry_node
from app.agent.nodes.triage import triage_node
from app.agent.prompts import SYSTEM_NAME_ASK, SYSTEM_NAME_EXTRACT, SYSTEM_SMALLTALK
from app.agent.state import AgentState

logger = logging.getLogger(__name__)


async def _smalltalk_node(state: AgentState) -> dict:
    """Sprint 2.0 — smalltalk with 3 phases for customer-name capture.

    Phase 1: if we asked for the name last turn (``name_asked=True``) and
             still don't have it, try to extract from the current message.
    Phase 2: if no name AND we haven't asked yet, send the name-ask reply
             and stop. Customer responds → next turn enters Phase 1.
    Phase 3: normal smalltalk reply; threads the name into the user block
             so the LLM can use it sparingly.
    """
    messages = state.get("messages") or []
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)

    raw_content = last_human.content if last_human else ""
    content = raw_content if isinstance(raw_content, str) else str(raw_content)

    customer_name = state.get("customer_name")
    name_asked = bool(state.get("name_asked", False))

    client = OpenAIClient()
    update: dict = {}

    # ── Sprint 2.4 — graceful goodbye after declining alternatives ───────
    if state.get("goodbye_pending"):
        goodbye = "Tudo bem! Se mudar de ideia, é só me chamar."
        logger.info("smalltalk goodbye_pending → canned reply")
        return {
            "messages": [AIMessage(content=goodbye)],
            "response_blocks": [goodbye],
            "goodbye_pending": False,
        }

    # ── Phase 1: name extraction ─────────────────────────────────────────
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
        # Close the asked window regardless — avoid looping if extraction fails.
        update["name_asked"] = False

    # ── Phase 2: first interaction → ask the name and return ─────────────
    # Sprint 2.4 — canned brand greeting. Deterministic, brand-consistent,
    # no LLM call needed for the first touch. (The LLM was rolling its
    # own greeting in 2.0, which made the brand presence inconsistent.)
    if not customer_name and not name_asked:
        ask_text = (
            "Olá! Bem-vindo à Base Sports 👋 Antes de tudo, qual seu nome?"
        )
        update["messages"] = [AIMessage(content=ask_text)]
        update["response_blocks"] = [ask_text]
        update["name_asked"] = True
        logger.info("smalltalk name_ask sent (brand canned)")
        return update

    # ── Phase 3: normal smalltalk (name optional in context) ─────────────
    user_block = (
        f"Nome do cliente: {customer_name}\n\n{content}"
        if customer_name else content
    )
    response = await client.chat(
        messages=[{"role": "user", "content": user_block}],
        system=SYSTEM_SMALLTALK,
        max_tokens=150,
        temperature=0.7,
    )
    text = (response or "").strip()
    update["messages"] = [AIMessage(content=text)]
    update["response_blocks"] = [text]
    return update


_CLASSICAL_INTENTS = {
    "faq", "diagnose", "recommend", "close", "consultoria", "handoff", "smalltalk",
    # scheduling_inquiry is classical-tier: routes to its own handoff node
    # regardless of post-recommendation state.
    "scheduling_inquiry",
}
_POST_REC_INTENTS = {
    "price_inquiry", "product_selection", "re_recommendation",
    "product_detail", "out_of_scope",
    # Sprint 1.15 — emitted only by the deterministic override in _triage_router
    # (never by the LLM directly) when a pronominal reference is ambiguous.
    "ambiguous_selection",
}


def _triage_router(state: AgentState) -> str:
    intent = state.get("intent") or "smalltalk"

    # Sprint 2.0 — bare_recommendation_request reuses the recommend path:
    # diagnose collects the profile, then recommend (PROFILE mode) delegates
    # to consultoria_offer. Same code path in both classical and post-rec.
    if intent == "bare_recommendation_request":
        intent = "recommend"

    # Sprint 2.1 — cliente determinado skips diagnose entirely. The customer
    # named a racket that exists in the catalog (triage already populated
    # modelo_desejado), so we go straight to recommend's REFERENCE branch.
    if (
        state.get("customer_intent_path") == "determined"
        and intent in ("recommend", "diagnose")
    ):
        logger.info("triage_router determined → recommend_determined (skip diagnose)")
        return "recommend_determined"

    has_products = bool(state.get("recommended_products"))
    is_post_rec = has_products and bool(state.get("last_recommendation_at"))

    # Post-recommendation intents only make sense when we actually have a
    # recommendation on the table. If triage emitted one outside that state
    # (shouldn't happen because we tell it Estado: pré-recomendação, but
    # defense in depth), demote to smalltalk.
    if intent in _POST_REC_INTENTS and not is_post_rec:
        intent = "smalltalk"

    if intent not in (_CLASSICAL_INTENTS | _POST_REC_INTENTS):
        return "smalltalk"

    # Sprint 1.14/1.15 — deterministic OVERRIDE in post-recommendation state.
    # Resolution order: tolerant name match → positional reference → pronominal
    # reference. We run this whenever the LLM gave us a "selection-adjacent"
    # intent (recommend / diagnose / product_selection) so that a multi-option
    # pronominal like "gostei dessa" is correctly demoted to
    # ``ambiguous_selection`` even when the LLM eagerly emitted product_selection.
    if is_post_rec and intent in ("recommend", "diagnose", "product_selection"):
        messages = state.get("messages") or []
        last_human = next(
            (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
        )
        last_text = (
            last_human.content
            if last_human and isinstance(last_human.content, str)
            else ""
        )
        products = state.get("recommended_products") or []
        original_intent = intent

        # 1) Tolerant name match — high or low confidence both win here.
        name_match = match_product_tolerant(last_text, products)
        if name_match.product is not None:
            logger.info(
                "triage_router override → product_selection method=%s confidence=%s name=%s",
                name_match.method,
                name_match.confidence,
                name_match.product.get("name"),
            )
            intent = "product_selection"
        else:
            # 2) Positional reference ("a primeira", "vou de segunda").
            idx = detect_positional_reference(last_text, len(products))
            if idx is not None:
                logger.info(
                    "triage_router override → product_selection method=positional idx=%d", idx
                )
                intent = "product_selection"
            elif detect_pronominal_reference(last_text):
                # 3) Pronominal — only auto-select when exactly 1 option is on
                # the table; otherwise ask the customer to disambiguate.
                if len(products) == 1:
                    logger.info(
                        "triage_router override → product_selection method=pronominal_single"
                    )
                    intent = "product_selection"
                else:
                    logger.info(
                        "triage_router override → ambiguous_selection method=pronominal_multi"
                    )
                    intent = "ambiguous_selection"
            elif original_intent == "product_selection":
                # The LLM thought this was a selection but we found no
                # resolvable reference. If there are multiple products,
                # asking is safer than fingers-crossed picking one.
                if len(products) > 1:
                    logger.info(
                        "triage_router override → ambiguous_selection "
                        "method=unresolved_with_multi_options"
                    )
                    intent = "ambiguous_selection"

    if is_post_rec:
        # Direct routing for the new follow-up intents.
        if intent == "price_inquiry":
            return "price_inquiry"
        if intent == "product_selection":
            return "product_selection"
        if intent == "re_recommendation":
            return "re_recommendation"
        if intent == "product_detail":
            return "product_detail"
        if intent == "out_of_scope":
            return "out_of_scope"
        if intent == "ambiguous_selection":
            return "ambiguous_selection"
        # Legacy intents in post-rec state:
        if intent == "close":
            return "close"
        if intent == "smalltalk":
            return "close"
        if intent in ("diagnose", "recommend"):
            # No explicit follow-up signal but customer is iterating — keep
            # the legacy "rerun recommend" path.
            return "recommend_rerun"
        # consultoria / faq / handoff fall through to the classical mapping.

    return intent


def _diagnose_router(state: AgentState) -> str:
    """After diagnose: if the LLM set intent=recommend, move on; otherwise wait for user."""
    if state.get("intent") == "recommend":
        return "recommend"
    return END


def _start_router(state: AgentState) -> str:
    """Skip triage when we're already mid-diagnose so short replies ('beach', 'sim', '300')
    are not misclassified as smalltalk by triage seeing them out of context."""
    if state.get("intent") == "diagnose":
        return "diagnose"
    return "triage"


def build_graph(checkpointer=None) -> CompiledStateGraph:
    """Build and compile the agent state-machine.

    Args:
        checkpointer: Optional checkpointer instance. When None (production),
            uses the Redis singleton initialized at app startup. Tests inject
            a ``MemorySaver`` so they can run in isolation without Redis.

    Topology:
        START →(mid-diagnose?)→ diagnose (skip triage)
              └──────────────→ triage →(intent)→ faq | diagnose | handoff | smalltalk | close
        faq        → END
        handoff    → END
        smalltalk  → END
        close      → END  (product confirmed, handoff triggered)
        diagnose →(complete?)→ recommend → END
                            └→ END  (waits for next user message)
    """
    builder: StateGraph = StateGraph(AgentState)

    builder.add_node("triage", triage_node)
    builder.add_node("diagnose", diagnose_node)
    builder.add_node("recommend", recommend_node)
    builder.add_node("close", close_node)
    builder.add_node("consultoria", pitch_consultoria_node)
    builder.add_node("faq", faq_node)
    builder.add_node("handoff", handoff_node)
    builder.add_node("smalltalk", _smalltalk_node)
    # Sprint 1.10 — follow-up nodes (post-recommendation state)
    builder.add_node("price_inquiry", price_inquiry_node)
    builder.add_node("product_selection", product_selection_node)
    builder.add_node("re_recommendation", re_recommendation_node)
    builder.add_node("product_detail", product_detail_node)
    builder.add_node("out_of_scope", out_of_scope_handoff_node)
    # Sprint 1.14 — scheduling handoff
    builder.add_node("scheduling_inquiry", scheduling_inquiry_node)
    # Sprint 1.15 — disambiguation when pronominal reference + multi-option
    builder.add_node("ambiguous_selection", ambiguous_selection_node)

    builder.add_conditional_edges(START, _start_router, {"triage": "triage", "diagnose": "diagnose"})

    builder.add_conditional_edges(
        "triage",
        _triage_router,
        {
            "faq": "faq",
            "diagnose": "diagnose",
            "recommend": "diagnose",        # first request — collect profile via diagnose
            "recommend_rerun": "recommend", # constraint change — skip diagnose
            "recommend_determined": "recommend",  # Sprint 2.1 — determined customer, skip diagnose
            "close": "close",               # product selected — confirm and handoff
            "consultoria": "consultoria",   # customer asked about / wants the Consultoria
            "handoff": "handoff",
            "smalltalk": "smalltalk",
            # Sprint 1.10 — follow-up routing
            "price_inquiry": "price_inquiry",
            "product_selection": "product_selection",
            "re_recommendation": "re_recommendation",
            "product_detail": "product_detail",
            "out_of_scope": "out_of_scope",
            # Sprint 1.14
            "scheduling_inquiry": "scheduling_inquiry",
            # Sprint 1.15
            "ambiguous_selection": "ambiguous_selection",
        },
    )

    builder.add_edge("faq", END)
    builder.add_edge("handoff", END)
    builder.add_edge("smalltalk", END)
    builder.add_edge("close", END)
    builder.add_edge("consultoria", END)
    # Sprint 1.10 edges
    builder.add_edge("price_inquiry", END)
    # Sprint 2.0 — product_selection is now a HANDOFF event (purchase_closing).
    # The node emits its own message + persists the dossier; no need to route
    # through close anymore.
    builder.add_edge("product_selection", END)
    builder.add_edge("re_recommendation", END)
    builder.add_edge("product_detail", END)
    builder.add_edge("out_of_scope", END)
    # Sprint 1.14
    builder.add_edge("scheduling_inquiry", END)
    # Sprint 1.15 — ambiguous_selection asks the customer to clarify, ends turn
    builder.add_edge("ambiguous_selection", END)

    builder.add_conditional_edges(
        "diagnose",
        _diagnose_router,
        {"recommend": "recommend", END: END},
    )

    builder.add_conditional_edges(
        "recommend",
        lambda s: "handoff" if s.get("needs_handoff") else END,
        {"handoff": "handoff", END: END},
    )

    if checkpointer is None:
        checkpointer = get_checkpointer()
    return builder.compile(checkpointer=checkpointer)
