"""Sprint 2.0 — re_recommendation pivots to the Consultoria.

Pre-2.0 this node fetched a second pass of rackets biased by the customer's
request ("mais barata", "top de linha", "outras opções"). The strategic pivot
makes that behaviour off-limits: the agent never picks another racket. Every
re-recommendation request is now an opportunity to offer the Consultoria.

The delegation to ``consultoria_offer_node`` produces a personalized
invitation that uses the customer's profile + name + last message. After the
pivot we clear ``recommended_products`` so the next turn leaves
post-recommendation mode (no follow-up intents apply anymore).
"""
import logging

from app.agent.nodes.consultoria_offer import consultoria_offer_node
from app.agent.state import AgentState

logger = logging.getLogger(__name__)


async def re_recommendation_node(state: AgentState) -> dict:
    logger.info("re_recommendation delegating_to=consultoria_offer (Sprint 2.0 pivot)")
    offer_result = await consultoria_offer_node(state)
    # Clear the active shortlist so the next turn isn't routed as
    # post-recommendation follow-up.
    offer_result["recommended_products"] = []
    offer_result["last_recommendation_at"] = None
    return offer_result
