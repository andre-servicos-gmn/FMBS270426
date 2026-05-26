"""Sprint 1.10 — handoff for questions outside the agent's scope.

Reason ``out_of_scope`` (vs. generic ``user_requested``) so the admin
dashboard can distinguish operational questions (delivery, payment methods,
hours) from explicit "fala com humano" requests. Sprint 2.2 pipes the
dossier through ``handoff_dossier_pipeline`` (summary → build → persist →
send to the configured WhatsApp recipient).
"""
import logging

from langchain_core.messages import AIMessage

from app.agent.dossier import handoff_dossier_pipeline
from app.agent.state import AgentState

logger = logging.getLogger(__name__)

_OUT_OF_SCOPE_MESSAGE = (
    "Pra essa informação específica, vou te encaminhar pro atendimento humano "
    "da loja. Eles conseguem te ajudar melhor com isso. Em breve alguém da "
    "equipe entra em contato!"
)


async def out_of_scope_handoff_node(state: AgentState) -> dict:
    state_for_dossier = dict(state)
    state_for_dossier["handoff_reason"] = "out_of_scope"
    await handoff_dossier_pipeline(state_for_dossier)

    return {
        "messages": [AIMessage(content=_OUT_OF_SCOPE_MESSAGE)],
        "response_blocks": [_OUT_OF_SCOPE_MESSAGE],
        "needs_handoff": True,
        "handoff_reason": "out_of_scope",
    }
