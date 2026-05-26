"""Sprint 1.14 — scheduling_inquiry: customer wants to BOOK the Consultoria.

After the pitch (or even before, if the customer arrived already knowing the
product), they may ask "como agendo?", "qual horário?", "que dias tem?".
We don't have a real booking calendar wired in, so this is a human handoff
with a contextual canned response — distinct from the generic
``out_of_scope_handoff`` because the reason and tone differ.

Sprint 2.2: persistence + WhatsApp delivery flow through
``handoff_dossier_pipeline``.
"""
import logging

from langchain_core.messages import AIMessage

from app.agent.dossier import handoff_dossier_pipeline
from app.agent.state import AgentState

logger = logging.getLogger(__name__)

_SCHEDULING_MESSAGE = (
    "Opa! Pra agendar sua consultoria, vou te encaminhar pro atendimento "
    "humano da loja — eles têm a agenda completa e conseguem te ajudar a "
    "escolher o melhor horário. Em breve alguém da equipe entra em contato!"
)


async def scheduling_inquiry_node(state: AgentState) -> dict:
    state_for_dossier = dict(state)
    state_for_dossier["handoff_reason"] = "scheduling"
    state_for_dossier["consultoria_interest"] = True
    await handoff_dossier_pipeline(state_for_dossier)

    return {
        "messages": [AIMessage(content=_SCHEDULING_MESSAGE)],
        "response_blocks": [_SCHEDULING_MESSAGE],
        "needs_handoff": True,
        "handoff_reason": "scheduling",
        "consultoria_interest": True,
    }
