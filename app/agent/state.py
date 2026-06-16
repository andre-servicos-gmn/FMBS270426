from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    phone_hash: str
    # faq | diagnose | recommend | close | consultoria | handoff | smalltalk
    intent: str | None
    # Slots tracked in player_profile. Strategy (Sprint 1.5): the WhatsApp agent
    # is intentionally a shallow diagnostic so it does NOT cannibalize the paid
    # in-store Consultoria Base Sports (R$350 with on-court testing). Only the
    # ESSENTIAL slots are asked; PROTECTED slots are captured *only* when the
    # customer mentions them spontaneously.
    #
    # Essential — actively asked by diagnose:
    #   nivel_jogo            — "iniciante" | "intermediário" | "avançado"
    #   lesoes                — "nenhuma" or free text
    #   regiao_lesao          — "cotovelo" | "ombro" | "punho" | "antebraco"
    #                            | "braco_inteiro" | "mais_de_uma" | "nenhuma"
    #   esporte_raquete_previo — "nenhum" | "tênis" | "padel" | "squash" | "tênis de mesa" |
    #                            free text | "nao_aplicavel" (auto-set when nivel_jogo is
    #                            intermediário/avançado — Sprint 1.6 Rule H)
    #   modelo_desejado       — free text or "nenhum"
    #
    # Defaulted/confirmed only:
    #   esporte_praticado     — DEFAULT "beach tennis"; agent confirms ONCE only
    #                           if customer hints at padel.
    #
    # PROTECTED — never asked, only captured if customer volunteers them:
    #   orcamento             — NUNCA perguntar (preservar valor da Consultoria)
    #   frequencia_pratica    — NUNCA perguntar
    #   tempo_pratica         — NUNCA perguntar
    #   estilo_jogo           — NUNCA perguntar
    #   equipamento_atual     — NUNCA perguntar
    #   marca_restrita        — captured only if customer declares sponsorship
    player_profile: dict
    recommended_products: list[dict]
    needs_handoff: bool
    handoff_reason: str | None
    # True once the customer has been pitched the Consultoria Base Sports
    # and we want to remember the interest signal for follow-up.
    consultoria_interest: bool
    # Sprint 1.6: optional list of message blocks for WhatsApp delivery. When
    # set by a node (recommend / pitch_consultoria), the webhook sends each
    # block as a separate WhatsApp message with a small humanizing delay
    # between blocks. Absent or empty → webhook sends a single message built
    # from the AIMessage content (with splitter fallback).
    response_blocks: NotRequired[list[str]]
    # Sprint 1.10: post-recommendation tracking. last_recommendation_at is set
    # to UTC ISO-8601 string when recommend_node finishes successfully with
    # >=1 product. Combined with non-empty recommended_products, signals
    # "post_recommendation" state to the triage router. selected_product
    # captures which product the customer chose (when product_selection fires)
    # and is read by close_node to confirm the right one.
    last_recommendation_at: NotRequired[str | None]
    selected_product: NotRequired[dict | None]
    # Sprint 1.14 — anti-rerun bookkeeping. last_node_executed names the most
    # recent node that produced a customer-facing reply (e.g. "recommend",
    # "pitch_consultoria"). Combined with last_node_executed_at (ISO-8601),
    # the anti-rerun helper decides whether to block a second invocation of
    # the same node within a short window — preventing the "rerun cego"
    # where the agent repeats an identical answer.
    last_node_executed: NotRequired[str | None]
    last_node_executed_at: NotRequired[str | None]
    # Sprint 2.0 — customer name. Captured on the first interaction by
    # _smalltalk_node and reused by every customer-facing prompt for a more
    # personal tone (used sparingly, 1-2x per conversation). ``name_asked``
    # tracks whether the agent has already prompted the customer for their
    # name so we don't ask repeatedly.
    customer_name: NotRequired[str | None]
    name_asked: NotRequired[bool]
    # Sprint 2.0 — pivot to qualifier. When the customer references a specific
    # product (REFERENCE mode), we keep the search term here so dossier
    # rendering can show "Pesquisou: X" even after diagnose finishes.
    produto_pesquisado: NotRequired[str | None]
    # Sprint 2.1 — counter that caps the subtle Consultoria pitch at 1
    # mention per conversation (otherwise it becomes spammy for customers
    # who ask several follow-ups).
    consultoria_mentioned_count: NotRequired[int]
    # Sprint 2.3 — counts how many technical questions the customer asked.
    # Used by ``maybe_add_subtle_consultoria_offer`` to apply the
    # DELAYED-pitch timing rule (wait for the 2nd question for non-IMMEDIATE
    # question types). Sprint 2.6: kept (still drives timing) even though
    # ``customer_intent_path`` is gone — every customer who reaches the
    # follow-up nodes effectively determined now.
    determined_question_count: NotRequired[int]
    # Sprint 2.4 / 2.6 — when the customer declines an offer, triage routes
    # to smalltalk with this flag set so the node emits a canned
    # "tudo bem, qualquer coisa me chama" goodbye instead of going to LLM.
    goodbye_pending: NotRequired[bool]
    # Sprint 2.6.2 — when recommend emits "Você quis dizer X?" (fuzzy_low),
    # the candidate product is stashed here so triage on the next turn can
    # resolve "sim"/"não" without re-running the fuzzy match.
    awaiting_match_confirmation: NotRequired[dict | None]
    # Sprint 2.6.2 — set by triage when the customer declines a fuzzy
    # suggestion; smalltalk reads it to emit a canned "ok, qual então?" reply.
    match_decline_pending: NotRequired[bool]
    # Sprint 2.6.4 — when recommend returns ambiguous (multiple products tied
    # at the same score) OR shows a top-3 fallback, we stash the candidates
    # so the next turn can resolve references like "as duas" / "ambas" /
    # "o segundo" without losing track of the list.
    last_product_candidates: NotRequired[list[dict] | None]
    # Sprint 2.6.6 — anti-spam list of "<product_id>:<attribute>" pairs the
    # agent already alerted Andre about ("não tenho esse dado, vou
    # confirmar"). Prevents 3 perguntas seguidas sobre o mesmo atributo
    # ausente em 3 alertas idênticos. Storing as list (not set) so the
    # checkpointer can serialize cleanly.
    alerted_missing_attrs: NotRequired[list[str] | None]
    # Sprint 2.6.8 — set to True the first time help_request_node emits the
    # Consultoria/loja offer. On every subsequent call, the node detects
    # this flag and emits a DIFFERENT "refusal/redirect" message instead
    # of repeating the same pitch (the Felipe loop bug). NEVER auto-cleared:
    # if the customer eventually names a product, recommend handles them
    # normally and the flag is irrelevant; if they ask for help again
    # later, the refusal text is still the correct response.
    help_request_already_offered: NotRequired[bool]
    # Sprint 2.6.10 — set to True by recommend_node whenever it emits the
    # confirmation message ("Posso te passar mais detalhes, ou prefere ver
    # pessoalmente na loja?"). The next turn's triage uses this flag as a
    # SHORT-CIRCUIT: if the customer accepts ("detalhes", "sim", "manda"),
    # route directly to attribute_inquiry without the LLM (which historically
    # misclassified "detalhes" as help_request or out_of_scope). If the
    # customer asks price instead, force price_inquiry. Any other input
    # clears the flag and goes through normal triage.
    #
    # CRITICAL: this field MUST exist in the TypedDict — LangGraph's
    # checkpointer silently discards keys absent from the schema, which
    # is exactly the failure mode the Felipe production logs showed.
    awaiting_detail_choice: NotRequired[bool]
    # Sprint 2.7.1 — set to True by recommend_node whenever it emits the
    # ambiguous-list message ("Qual você procura? • A • B • C"). The next
    # turn's triage uses it together with ``last_product_candidates`` to
    # short-circuit positional ("primeira"), year-token ("2026") and
    # partial-name ("Hugo Russo") selections BEFORE the LLM gets a chance
    # to misclassify them as smalltalk (the production "amnesia" bug —
    # Felipe, 2026-06).
    #
    # CRITICAL: declared in the TypedDict so the checkpointer persists it.
    # See the awaiting_detail_choice comment above for the same lesson.
    awaiting_candidate_choice: NotRequired[bool]
    # Sprint 2.7.3 — flipped True by the triage short-circuit when the
    # customer mentions a budget ("até 2k", "no máximo 1500", "uns 2 mil
    # reais"). help_request_node reads it to acknowledge the budget in
    # the Consultoria pitch ("a Consultoria considera perfil, jogo E
    # faixa de valor"). NEVER lists products — the business rule forbids
    # price-range vitrine. Cleared by help_request after consumption.
    #
    # CRITICAL: declared in the TypedDict (lesson 2.6.10) so the
    # checkpointer doesn't silently drop it between turns.
    price_range_mentioned: NotRequired[bool]
