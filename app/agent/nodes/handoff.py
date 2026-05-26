"""Generic human-handoff node (customer asked for an attendant)."""
import logging

from langchain_core.messages import AIMessage

from app.agent.dossier import handoff_dossier_pipeline
from app.agent.state import AgentState

logger = logging.getLogger(__name__)

_HANDOFF_MESSAGE = (
    "Vou te conectar com um especialista, em breve alguém te chama por aqui."
)


async def handoff_node(state: AgentState) -> dict:
    state_for_dossier = dict(state)
    state_for_dossier["handoff_reason"] = "user_requested"
    await handoff_dossier_pipeline(state_for_dossier)

    return {
        "messages": [AIMessage(content=_HANDOFF_MESSAGE)],
        "response_blocks": [_HANDOFF_MESSAGE],
        "needs_handoff": True,
        "handoff_reason": "user_requested",
    }
