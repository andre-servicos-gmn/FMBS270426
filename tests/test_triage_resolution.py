"""Sprint 1.15 — integration tests for the resolution pipeline in _triage_router.

These verify the deterministic override that routes recommend/diagnose intents
to product_selection (high/low match, positional, pronominal-single) or to
ambiguous_selection (pronominal-multi).
"""
from datetime import datetime, timezone

import pytest
from langchain_core.messages import HumanMessage

from app.agent.graph import _triage_router
from app.agent.state import AgentState

pytestmark = pytest.mark.skip(
    reason="diagnose deprecated in Sprint 2.6 — post-rec router overrides removed"
)


def _post_rec_state(*, message: str, products: list[dict], llm_intent: str = "recommend") -> AgentState:
    """Build a state that looks like post-recommendation, with the LLM having
    classified the customer's message as ``llm_intent``."""
    return AgentState(  # type: ignore[typeddict-item]
        messages=[HumanMessage(content=message)],
        phone_hash="resolution" * 6 + "xx",
        intent=llm_intent,
        player_profile={},
        recommended_products=products,
        needs_handoff=False,
        handoff_reason=None,
        consultoria_interest=False,
        last_recommendation_at=datetime.now(timezone.utc).isoformat(),
    )


_BEACH = {"name": "Raquete BeachPro Carbon X5", "price_cents": 89900, "external_id": "X5"}
_FOAM = {"name": "Raquete BeachPro Foam Series 300", "price_cents": 29900, "external_id": "F1"}


def test_triage_routes_to_product_selection_when_match_high():
    """Customer types full product name → product_selection."""
    state = _post_rec_state(
        message="Pode reservar a Raquete BeachPro Carbon X5",
        products=[_BEACH, _FOAM],
    )
    assert _triage_router(state) == "product_selection"


def test_triage_routes_to_product_selection_when_spaces_collapsed():
    """The exact bug from the WhatsApp report — must route to product_selection."""
    state = _post_rec_state(
        message="Pode reservar essa beach pro foam series 300",
        products=[_BEACH, _FOAM],
    )
    assert _triage_router(state) == "product_selection"


def test_triage_routes_to_product_selection_when_positional():
    state = _post_rec_state(
        message="vou de segunda",
        products=[_BEACH, _FOAM],
    )
    assert _triage_router(state) == "product_selection"


def test_triage_routes_to_product_selection_when_pronominal_single_option():
    """Single option on the table + pronominal → resolve to that single option."""
    state = _post_rec_state(
        message="gostei dessa",
        products=[_BEACH],
    )
    assert _triage_router(state) == "product_selection"


def test_triage_routes_to_ambiguous_when_pronominal_multiple_options():
    """Pronominal with 2+ options → ambiguous_selection (ask which)."""
    state = _post_rec_state(
        message="gostei dessa",
        products=[_BEACH, _FOAM],
    )
    assert _triage_router(state) == "ambiguous_selection"


def test_triage_preserves_re_recommendation_keyword():
    """'tem mais barata?' must NOT be misclassified as product selection."""
    state = _post_rec_state(
        message="tem mais barata?",
        products=[_BEACH, _FOAM],
        llm_intent="re_recommendation",
    )
    # Router accepts the LLM's re_recommendation intent directly.
    assert _triage_router(state) == "re_recommendation"


def test_triage_keeps_recommend_rerun_when_no_signal():
    """If nothing resolves, fall back to legacy recommend_rerun path."""
    state = _post_rec_state(
        message="hm, deixa eu pensar",
        products=[_BEACH, _FOAM],
    )
    # No name match, no positional, no pronominal → recommend_rerun (legacy).
    assert _triage_router(state) == "recommend_rerun"
