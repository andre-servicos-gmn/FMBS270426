"""Sprint 2.6.2 — static guards against phantom phrases.

Two real bugs in production sparked this guard:

1. Agent revealed its own internal mechanism: ``"É só me dizer 'me ajuda' que
   eu te oriento"`` — leaks that 'me ajuda' is a keyword the LLM triggers on.

2. Agent lied about having shown options: ``"Te mostrei algumas opções
   acima"`` — left over from a deleted multi-product flow but the canned
   text survived in anti_rerun fallbacks.

Both rules below grep the entire ``app/agent/`` tree and fail if either
phrase appears as USER-FACING TEXT. (The phrases ARE allowed in docstrings
that reference the bug — we filter by stripping docstrings/comments
heuristically: we look at the raw file content for these specific phrases.)
"""
from pathlib import Path

import pytest


_AGENT_DIR = Path(__file__).resolve().parent.parent / "app" / "agent"


def _user_facing_files() -> list[Path]:
    """Files inside ``app/agent/`` that could ever emit text to the customer.

    We exclude ``diagnose.py`` (deprecated, disconnected from the graph)
    and ``ambiguous_selection.py`` (legacy from Sprint 1.15, also
    disconnected in Sprint 2.6). Tests cover the live nodes.
    """
    excluded = {"diagnose.py", "ambiguous_selection.py"}
    return [
        p for p in _AGENT_DIR.rglob("*.py")
        if p.name not in excluded and "__pycache__" not in p.parts
    ]


# ── Strict phrases (literal substring) that must NOT appear anywhere ────────

_FORBIDDEN_LITERAL: dict[str, str] = {
    "me dizer 'me ajuda'":
        "leaks the 'me ajuda' keyword to the customer (Sprint 2.6.2)",
    "Te mostrei algumas opções":
        "false claim about having shown options (Sprint 2.6.2)",
    "mostrei opções acima":
        "false claim about having shown options (Sprint 2.6.2)",
    "Quer escolher uma, comparar":
        "leftover from removed multi-product flow (Sprint 2.6.2)",
    "prefere que eu busque outras":
        "leftover from removed multi-product flow (Sprint 2.6.2)",
    "Base Esportes":
        "brand-name regression — the correct name is 'Base Sports' (Sprint 2.6.9). "
        "If you need to reference the consultoria, use 'Consultoria Base Sports'.",
}


def _all_agent_text() -> str:
    chunks = []
    for path in _user_facing_files():
        try:
            chunks.append(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    return "\n".join(chunks)


@pytest.mark.parametrize("phrase,reason", list(_FORBIDDEN_LITERAL.items()))
def test_phantom_phrase_absent_from_live_agent(phrase, reason):
    text = _all_agent_text()
    assert phrase not in text, (
        f"\nForbidden phrase found: {phrase!r}\n"
        f"Reason: {reason}\n"
        f"Remove from live agent code (docstrings referencing the bug are fine,\n"
        f"but the literal phrase mustn't be emitted to users)."
    )


def test_no_recipe_for_help_keyword():
    """The 'me ajuda' helper trigger must not be revealed to users.

    The phrase exists in the triage prompt as a CLASSIFICATION HINT (the
    LLM categorizes 'me ajuda' as help_request), but it must not appear
    in any reply template / canned message / outgoing text.
    """
    forbidden_recipe_phrases = (
        "É só me dizer 'me ajuda'",
        "É só me dizer me ajuda",
        "me chama 'me ajuda'",
    )
    text = _all_agent_text()
    for phrase in forbidden_recipe_phrases:
        assert phrase not in text, f"Mechanism leak: {phrase!r}"


def test_no_pretend_to_have_shown_options():
    """The agent must never claim to have shown options in a single-product flow."""
    forbidden = (
        "Te mostrei algumas opções",
        "mostrei opções acima",
        "Te apresentei algumas",
    )
    text = _all_agent_text()
    for phrase in forbidden:
        assert phrase not in text, f"False history claim: {phrase!r}"
