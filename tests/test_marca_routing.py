"""Sprint 2.7.6 — fix triage misclassification of "qual a marca?" → faq.

Production bug: with an active product in the conversation (Mormaii Sunset
Plus), the customer asks "qual a marca?" and the LLM triages it as ``faq``
(treating it as "which brands do you carry"), routing to handoff instead of
``attribute_inquiry``. The attribute_inquiry node already supports the
``marca`` slug (Sprint 2.7.5), but the triage diverts before it can run.

Fix: SYSTEM_TRIAGE updated with explicit marca/modelo examples in the
attribute_inquiry list and a new disambiguation rule (d) teaching the
LLM the singular+active-product (attribute) vs plural+catalog (faq)
distinction.

These tests assert PROMPT CONTENT — we don't run the LLM (non-deterministic).
The prompt is the contract; if the LLM still misclassifies in production
with this prompt, the next iteration is reinforcement, not removal of
the rule.
"""
import re

from app.agent.prompts import SYSTEM_TRIAGE


# F-string line continuations in SYSTEM_TRIAGE preserve the indentation
# spaces after the trailing ``\``, so substrings like "qual a espessura?"
# can land as "qual a                         espessura?" in the rendered
# string. Tests use this collapsed form so the actual semantic content
# is what's asserted, not the prompt's physical layout.
_FLAT = re.sub(r"\s+", " ", SYSTEM_TRIAGE)


# ════════════════════════════════════════════════════════════════════════════
# PROMPT CONTENT — required signals for the LLM to disambiguate correctly
# ════════════════════════════════════════════════════════════════════════════

def test_prompt_lists_marca_question_as_attribute_inquiry_example():
    """The attribute_inquiry description must explicitly include the
    "qual a marca?" pattern so the LLM doesn't fall back to faq."""
    assert "qual a marca?" in _FLAT
    assert "qual a marca dela?" in _FLAT


def test_prompt_lists_modelo_question_as_attribute_inquiry_example():
    assert "qual o modelo?" in _FLAT


def test_prompt_lists_de_que_marca_e_pattern():
    """Variant that customers actually use: 'de que marca é?'."""
    assert "de que marca é?" in _FLAT


def test_prompt_lists_fabricante_synonym():
    """Customers also say 'fabricante' meaning the brand."""
    assert "qual o fabricante?" in _FLAT


def test_prompt_has_explicit_disambiguation_rule():
    """The new rule (d) MUST explicitly contrast singular+active-product
    (attribute_inquiry) vs plural+catalog (faq)."""
    # Header mention
    assert "DESAMBIGUAÇÃO MARCA/MODELO" in _FLAT
    # Plural catalog example explicitly mapped to faq
    assert "quais marcas vocês trabalham?" in _FLAT
    assert "trabalham com quais marcas?" in _FLAT
    # Explicit attribute_inquiry directive
    assert "atributo do produto ATIVO" in _FLAT
    # Explicit faq directive for plural
    assert "pergunta de CATÁLOGO geral" in _FLAT


def test_prompt_warns_against_faq_with_active_product():
    """The prompt must explicitly tell the LLM: NEVER faq when there's
    a product recently confirmed."""
    assert "NUNCA classifique como faq quando" in _FLAT


# ════════════════════════════════════════════════════════════════════════════
# REGRESSION — existing attribute_inquiry signals intact
# ════════════════════════════════════════════════════════════════════════════

def test_prompt_still_lists_peso_attribute_inquiry():
    """Sprint 2.6.6 examples must remain — modelo/material/espessura
    routing was already working in production."""
    assert "qual o peso?" in _FLAT


def test_prompt_still_lists_material_composicao_espessura():
    assert "de que material é?" in _FLAT
    assert "qual a composição?" in _FLAT
    assert "qual a espessura?" in _FLAT


def test_prompt_still_has_broad_detail_paragraph():
    """Sprint 2.6.10 'detalhes' broad-detail routing must remain."""
    assert "quero detalhes" in _FLAT
    assert "detalhes por favor" in _FLAT


def test_prompt_still_has_context_section_for_short_replies():
    """Sprint 2.7.1 history-context section must remain."""
    assert "USO DO CONTEXTO" in _FLAT
    # Positional selection from candidate list
    assert "primeira" in _FLAT
    assert "Qual você procura?" in _FLAT


def test_prompt_faq_category_still_lists_horario_localizacao():
    """Sprint 2.6 faq category must remain — store info routing intact."""
    assert "horário da loja" in _FLAT
    assert "localização" in _FLAT
    assert "garantia" in _FLAT


def test_prompt_has_9_intents_listed():
    """Smoke test on the prompt structure — all 9 Sprint 2.6 intents
    are still part of the category list."""
    for intent in (
        "smalltalk", "product_inquiry", "attribute_inquiry",
        "price_inquiry", "purchase_intent", "scheduling_inquiry",
        "out_of_scope", "faq", "help_request", "close",
    ):
        assert intent in _FLAT, f"missing intent: {intent}"
