from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    phone_hash: str
    # faq | diagnose | recommend | close | consultoria | handoff | smalltalk
    intent: str | None
    # Slots tracked in player_profile. Strategy (Sprint 1.5): the WhatsApp agent
    # is intentionally a shallow diagnostic so it does NOT cannibalize the paid
    # in-store Consultoria Base Esportes (R$350 with on-court testing). Only the
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
    # True once the customer has been pitched the Consultoria Base Esportes
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
    # Sprint 2.1 — customer journey tracking.
    #   "determined" — customer named a specific racket OR clearly wants to
    #                  close (skip the diagnose questions).
    #   "exploring"  — customer asked for indication / doesn't know which one
    #                  (run diagnose, end in consultoria_offer).
    #   "unknown"    — no clear signal yet.
    # Triage assigns this; recommend/price_inquiry/product_detail tailor their
    # response shape and decide whether to emit the subtle Consultoria pitch.
    customer_intent_path: NotRequired[Literal["determined", "exploring", "unknown"] | None]
    # Sprint 2.1 — counter that caps the subtle Consultoria pitch at 1
    # mention per conversation (otherwise it becomes spammy for determined
    # customers who ask several follow-ups).
    consultoria_mentioned_count: NotRequired[int]
    # Sprint 2.3 — counts how many technical questions the determined
    # customer asked. Used by ``maybe_add_subtle_consultoria_offer`` to
    # apply the DELAYED-pitch timing rule (wait for the 2nd question for
    # non-IMMEDIATE question types).
    determined_question_count: NotRequired[int]
    # Sprint 2.4 — set to True when REFERENCE-NÃO determined asks the
    # customer "posso te ajudar a ver outras opções?". Triage reads this on
    # the next turn to interpret short "sim"/"não" replies as a transition
    # to exploring (yes) or a graceful goodbye (no).
    awaiting_alternatives_decision: NotRequired[bool]
    # Sprint 2.4 — when the customer declines the alternatives offer, triage
    # routes to smalltalk with this flag set so the node emits a canned
    # "tudo bem, qualquer coisa me chama" goodbye instead of going to LLM.
    goodbye_pending: NotRequired[bool]
