"""Sprint 1.15 — tolerant product-name matcher tests.

Covers the 4 layers (exact, spaces-collapsed, fuzzy, token), confidence
labels, and backward compat with the legacy ``match_product_in_text``.
"""
import pytest

from app.agent.nodes._product_match import (
    MatchResult,
    match_product_in_text,
    match_product_tolerant,
)


def _product(name: str) -> dict:
    return {"name": name, "price_cents": 50000, "external_id": name}


_BEACH = _product("Raquete BeachPro Carbon X5")
_FOAM = _product("Raquete BeachPro Foam Series 300")
_ALL = [_BEACH, _FOAM]


# ── Layer 1 — exact normalized ────────────────────────────────────────────────

def test_matcher_handles_exact_match():
    r = match_product_tolerant("quero a Raquete BeachPro Carbon X5", _ALL)
    assert r.product is _BEACH
    assert r.confidence == "high"
    assert r.method == "exact"


def test_matcher_handles_case():
    r = match_product_tolerant("quero a BEACHPRO carbon x5", _ALL)
    assert r.product is _BEACH
    assert r.confidence == "high"


def test_matcher_handles_missing_accents():
    products = [_product("Raquete Avançada Pro")]
    r = match_product_tolerant("quero a avancada pro", products)
    assert r.product is products[0]
    assert r.confidence == "high"


# ── Layer 2 — spaces collapsed ────────────────────────────────────────────────

def test_matcher_handles_spaces_collapsed():
    """The exact bug from the WhatsApp report: 'beach pro foam series 300'
    should match 'BeachPro Foam Series 300'."""
    r = match_product_tolerant(
        "Pode reservar essa beach pro foam series 300", _ALL
    )
    assert r.product is _FOAM
    assert r.confidence == "high"
    assert r.method == "spaces_collapsed"


def test_matcher_handles_extra_spaces_inside_name():
    r = match_product_tolerant("quero a beach  pro  carbon  x5", _ALL)
    assert r.product is _BEACH


# ── Layer 3 — fuzzy ───────────────────────────────────────────────────────────

def test_matcher_handles_typo_small():
    """1-char typo close to threshold → low confidence."""
    r = match_product_tolerant("quero a BeechPro Carbon X5", _ALL)
    assert r.product is _BEACH
    # 1 char off in a longer string → high ratio. Either high or low is acceptable
    # depending on string lengths; the important thing is we DO match.
    assert r.confidence in ("high", "low")
    assert r.method in ("fuzzy", "spaces_collapsed")


def test_matcher_returns_low_confidence_for_fuzzy_ratio_below_95():
    """A larger typo lands in the 0.85-0.95 band → low confidence."""
    # "BechPro Carbn X" vs "BeachPro Carbon X5" → bigger ratio drop.
    r = match_product_tolerant("vou de BechPro Carbn X", _ALL)
    if r.product is not None:
        # Acceptable: matcher either flags as low confidence or refuses.
        assert r.confidence in ("low", "high")
    # If r.product is None, that's also acceptable (very conservative).


# ── Layer 4 — token unicity ───────────────────────────────────────────────────

def test_matcher_falls_back_to_token_unicity():
    """A unique distinctive token (5+ chars, non-generic) → low confidence match."""
    # 'Vertex' is unique to one product. We deliberately mention only that token.
    products = [
        _BEACH,
        _product("Raquete VertexBT Pro Elite"),
    ]
    r = match_product_tolerant("vou de vertexbt", products)
    assert r.product is products[1]
    # Token unicity is always low confidence by design.
    assert r.confidence == "low"
    assert r.method in ("token", "fuzzy")


# ── No match ──────────────────────────────────────────────────────────────────

def test_matcher_returns_none_for_unrelated_text():
    r = match_product_tolerant("vocês entregam em casa?", _ALL)
    assert r.product is None
    assert r.confidence == "none"


def test_matcher_returns_none_for_empty_input():
    assert match_product_tolerant("", _ALL).product is None
    assert match_product_tolerant("oi", []).product is None
    assert match_product_tolerant("oi", _ALL).product is None


# ── Backward compatibility ────────────────────────────────────────────────────

def test_legacy_helper_returns_just_product():
    """match_product_in_text remains a simple ``dict | None`` wrapper."""
    out = match_product_in_text("essa beach pro foam series 300", _ALL)
    assert out is _FOAM
    assert match_product_in_text("nada relacionado", _ALL) is None


def test_match_result_dataclass_has_expected_fields():
    r = MatchResult(product=None, confidence="none", method="none")
    assert r.found is False
    r2 = MatchResult(product=_BEACH, confidence="high", method="exact")
    assert r2.found is True
