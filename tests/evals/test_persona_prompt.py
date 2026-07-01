"""T4 / T5 / T6 — persona pinned in the LIVE V2 system prompt.

Target is ``build_system_prompt()`` (app/agent/supervisor.py), the prompt the
V2 agent actually sends. Deterministic content checks — no LLM. RED until
persona edits land; that is the intended TDD gate behavior.

Note: we do NOT assert the prompt is free of em-dashes — SYSTEM_SUPERVISOR_TEMPLATE
uses em-dashes in its own instructional prose, which never reaches the customer.
The travessão guarantee is enforced on OUTPUT (see test_humanization_sanitizer.py).
"""
import pytest

from tests.evals._helpers import live_prompt

pytestmark = pytest.mark.deterministic


# ── T4: welcome signature (via prompt, no accent) ────────────────────────────

def test_welcome_signature_in_prompt():
    """The new greeting's signature phrase must be wired into the system prompt.
    (Via prompt, not canned — so we pin the signature, not byte-exact text.)"""
    p = live_prompt()
    assert "Sou o assistente Base" in p, "welcome signature 'Sou o assistente Base' missing from system prompt"


# ── T5: no AI self-declaration in the persona ────────────────────────────────

def test_prompt_no_ai_self_declaration():
    """The agent must read as a store attendant, not a bot: the persona must not
    POSITIVELY identify itself as an AI. We target the self-identification
    phrasings only — the prompt legitimately NAMES 'assistente virtual' inside a
    prohibition ("nunca se declare como ... assistente virtual"), which is the
    fix, not the bug. The bug is an identity line like "Você é o assistente
    virtual da loja" (RED until persona swaps it for the 'assistente Base'
    identity)."""
    p = live_prompt().lower()
    for bad in (
        "você é o assistente virtual",
        "voce e o assistente virtual",
        "sou o assistente virtual",
        "sou uma ia",
        "sou um robô",
        "sou um bot",
        "sou um modelo de linguagem",
    ):
        assert bad not in p, f"prompt positively self-identifies as AI: {bad!r}"


# ── T6: material cheat-sheet + translate-spec-into-impact instruction ─────────

def test_prompt_has_material_cheatsheet():
    """HÍBRIDO decision: a short cheat-sheet in the prompt naming the three
    dimensions AND instructing the agent to translate spec into practical
    impact (not dump jargon)."""
    p = live_prompt().lower()
    assert "carbono" in p, "cheat-sheet must name Carbono"
    assert "eva" in p, "cheat-sheet must name EVA"
    assert "fura" in p, "cheat-sheet must name Furação"
    assert any(k in p for k in ("na prática", "o que isso significa", "traduz", "impacto")), \
        "prompt must instruct translating technical spec into practical impact"


def test_prompt_leads_by_three_dimensions():
    """The cheat-sheet should tell the agent to LEAD a racket description by the
    three dimensions, not bury them."""
    p = live_prompt().lower()
    assert all(d in p for d in ("carbono", "eva", "fura")), \
        "all three dimensions (Carbono/EVA/Furação) must be present in the prompt"
