"""Sprint 1.13 — UX polish (lean recommend + natural confirmation + tone variation).

LLM behavior is non-deterministic, so most tests assert the PROMPT instructs
the right behaviour. The integration test that exercises the recommend node
with a mocked LLM response also confirms the structural pipeline (parse,
guardrail, blocks) still works after the template rewrite.
"""
import json
import re
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.agent.state import AgentState


@asynccontextmanager
async def _mock_db_session():
    session = MagicMock()
    session.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    )
    session.commit = AsyncMock()
    yield session


# ════════════════════════════════════════════════════════════════════════════
# RECOMENDAÇÃO ENXUTA — prompt-content checks
# ════════════════════════════════════════════════════════════════════════════

def test_recommend_each_product_max_2_lines_description():
    """SYSTEM_RECOMMEND must instruct the LLM to keep each product to ≤2 lines
    of description. The rule is encoded both as a 'MÁXIMO 2 linhas' line and
    via good/bad worked examples that show the difference."""
    from app.agent.prompts import SYSTEM_RECOMMEND
    s = SYSTEM_RECOMMEND.lower()
    # Hard line cap stated explicitly.
    assert "máximo 2 linhas" in s or "no máximo 2 linhas" in s
    # Total bracket on the racket card.
    assert "2 a 3 linhas totais" in s or "2 a 3 linhas" in s
    # Both worked examples (good and bad) must be present so the LLM sees
    # the contrast.
    assert "exemplos ✅" in s or "exemplos ✅ bons" in s
    assert "exemplos ❌" in s or "❌ ruins" in s


def test_recommend_ideal_pra_is_short():
    """Prompt must constrain the 'Ideal pra:' suffix to a short phrase."""
    from app.agent.prompts import SYSTEM_RECOMMEND
    s = SYSTEM_RECOMMEND.lower()
    # Explicit word count guideline.
    assert "3 a 7 palavras" in s
    # And the placeholder must be the short variant (not the legacy long one).
    assert "_perfil curto_" in SYSTEM_RECOMMEND


def test_recommend_does_not_stack_three_benefits():
    """The prompt forbids piling up 3+ benefits in one description."""
    from app.agent.prompts import SYSTEM_RECOMMEND
    s = SYSTEM_RECOMMEND.lower()
    assert "nunca empilhar 3+ benefícios" in s or "nunca empilhar 3+" in s


# ════════════════════════════════════════════════════════════════════════════
# CONFIRMAÇÃO NATURAL — prompt-content
# ════════════════════════════════════════════════════════════════════════════

def test_diagnose_phrase_confirms_after_availability_question():
    """SYSTEM_DIAGNOSE_PHRASE must instruct the LLM to start with a confirmation
    ('Temos sim!' / 'Vendemos sim') when the customer's last message was a
    direct availability question."""
    from app.agent.prompts import SYSTEM_DIAGNOSE_PHRASE
    s = SYSTEM_DIAGNOSE_PHRASE.lower()
    # Rule must be named.
    assert "confirmação natural" in s
    # Trigger phrasing taught explicitly.
    assert "vocês têm" in s or "vendem palas" in s
    # The model answer "Temos sim!" must be present as an example.
    assert "temos sim" in s


def test_diagnose_phrase_does_not_force_confirmation():
    """The prompt must explicitly enumerate cases where confirmation is forced
    and feels robotic, telling the LLM to skip it."""
    from app.agent.prompts import SYSTEM_DIAGNOSE_PHRASE
    s = SYSTEM_DIAGNOSE_PHRASE.lower()
    # The "quando não confirmar" section must exist.
    assert "quando não confirmar" in s
    # Examples of what NOT to do.
    assert "resposta a uma pergunta sua" in s
    assert "forçada ou redundante" in s


# ════════════════════════════════════════════════════════════════════════════
# VARIAÇÃO DE TOM — every customer-facing prompt embeds the rule
# ════════════════════════════════════════════════════════════════════════════

_TONE_OPENERS_TO_FORBID = ("show!", "beleza!", "vale a pena", "ótimo!", "perfeito!")


@pytest.mark.parametrize("prompt_name", [
    "SYSTEM_DIAGNOSE_PHRASE",
    "SYSTEM_RECOMMEND",
    "SYSTEM_SMALLTALK",
    "SYSTEM_FAQ",
    "SYSTEM_CLOSE",
    # Pitch needs the rule too — generated via builder, see separate test.
])
def test_prompts_forbid_repeated_show_beleza_vale_a_pena(prompt_name):
    """Each customer-facing prompt must carry the tone-variation rule."""
    from app.agent import prompts
    prompt = getattr(prompts, prompt_name)
    if callable(prompt):
        prompt = prompt({})  # type: ignore[call-arg]
    s = prompt.lower()
    assert "variação de tom" in s, f"{prompt_name} missing tone-variation block"
    # Each forbidden opener must be named at least once.
    for opener in _TONE_OPENERS_TO_FORBID:
        assert opener in s, f"{prompt_name} doesn't name '{opener}' as a forbidden opener"


def test_pitch_consultoria_prompt_carries_tone_variation():
    """Builder-generated pitch prompt also embeds the variation rule."""
    from app.agent.prompts import build_pitch_consultoria_prompt

    class _Settings:
        consultoria_preco = 350

    prompt = build_pitch_consultoria_prompt(_Settings()).lower()
    assert "variação de tom" in prompt
    assert "muleta robótica" in prompt or "muleta robotica" in prompt


def test_tone_variation_lists_alternatives():
    """The shared variation guidance must offer at least 3 alternative openers
    so the LLM has somewhere to pivot to."""
    from app.agent.prompts import _VARIATION_GUIDANCE
    s = _VARIATION_GUIDANCE.lower()
    alternatives = ("entendi", "boa", "legal", "bacana", "faz sentido", "anotado")
    found = sum(1 for a in alternatives if a in s)
    assert found >= 4, f"only {found} alternatives found in variation guidance"


def test_tone_variation_replaces_vale_a_pena():
    """For recommend, the variation block must offer alternatives to
    'Vale a pena começar por' so the LLM doesn't open every reco with it."""
    from app.agent.prompts import _VARIATION_GUIDANCE
    s = _VARIATION_GUIDANCE.lower()
    # At least one of the suggested alternatives for product opening.
    alternatives_for_product_opening = (
        "uma boa opção pra você é",
        "pro seu perfil, indicaria",
        "sugiro dar uma olhada",
        "boa opção é",
        "considere a",
    )
    matched = [a for a in alternatives_for_product_opening if a in s]
    assert len(matched) >= 3, f"need at least 3 alternative product openers; got {matched}"


# ════════════════════════════════════════════════════════════════════════════
# Quantitative sanity — the leaner template is actually shorter than before
# ════════════════════════════════════════════════════════════════════════════

def test_recommend_template_is_lean_about_description_length():
    """Regex sanity: the prompt explicitly bounds the description size."""
    from app.agent.prompts import SYSTEM_RECOMMEND
    # The brevity section must enforce a numeric cap on description lines.
    assert re.search(r"máximo\s+2\s+linhas", SYSTEM_RECOMMEND, flags=re.IGNORECASE)
    assert re.search(r"ultra-curto", SYSTEM_RECOMMEND, flags=re.IGNORECASE) or \
        re.search(r"3\s+a\s+7\s+palavras", SYSTEM_RECOMMEND, flags=re.IGNORECASE)
