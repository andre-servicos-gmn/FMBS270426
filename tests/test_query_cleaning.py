"""Sprint 2.6.8 — query cleaning + matcher behavior on noisy inputs.

The matcher used to score the WHOLE raw query against catalog names. So
"Olá! Você tem a nova Kronos?" (real production case from customer
Felipe) had a Levenshtein distance of 15 against the actual Kronos
product, and SHORTER unrelated products (Mochila Raqueteira) ranked
higher in top-3 by sliding-window distance. Result: no_result, even
though the Kronos was right there.

This module covers:
  - ``_clean_product_query`` strips greetings, courtesy verbs, generic
    adjectives, articles, punctuation.
  - It PRESERVES brand/model tokens (no allowlist required) and product
    specs like "12k" / "18k".
  - Single isolated "k" is stripped; "12k" is kept.
  - Empty result after cleaning → matcher returns ``status="none"``
    without crashing.
  - End-to-end matcher cases that DIDN'T work before now work.
  - Regressions: existing working cases (mormaii, manguito, drop shot)
    still pass; the Sprint 2.6.6 attribute-trigger strip still works.
"""
import pytest

from app.agent.nodes._product_match import (
    _clean_product_query,
    _distinctive_tokens,
    match_product_tolerant,
)


# ── _clean_product_query — unit tests ───────────────────────────────────


def test_clean_removes_greeting_and_courtesy():
    cleaned = _clean_product_query("Olá! Você tem a nova Kronos?")
    # All noise gone; only the product token remains.
    assert "kronos" in cleaned
    for noise in ("ola", "voce", "tem", "nova"):
        assert noise not in cleaned.split()


def test_clean_preserves_brand_and_model():
    cleaned = _clean_product_query("procuro uma proteo da ama")
    tokens = cleaned.split()
    assert "proteo" in tokens
    assert "ama" in tokens
    # Verbs/articles dropped.
    assert "procuro" not in tokens
    assert "uma" not in tokens
    assert "da" not in tokens


def test_clean_preserves_spec_tokens_like_12k():
    """Specs that combine digits + letter (12k, 18k) must survive."""
    cleaned = _clean_product_query("tem raquete drop shot 12k aí?")
    tokens = cleaned.split()
    assert "12k" in tokens
    assert "drop" in tokens
    assert "shot" in tokens
    # ``tem`` (STOP_WORDS) and ``ai`` (after accent strip, also in STOP_WORDS)
    # are gone. ``raquete`` stays — generic catalog tokens are filtered
    # only inside _distinctive_tokens (token-score layer), NOT here.
    assert "tem" not in tokens
    assert "ai" not in tokens


def test_clean_removes_isolated_k_keeps_attached_k():
    """`quantos k` → strip both 'quantos' and 'k'. `12k` → keep."""
    no_k = _clean_product_query("quantos k tem a kronos")
    assert "k" not in no_k.split()
    assert "kronos" in no_k

    with_k = _clean_product_query("12k carbono")
    assert "12k" in with_k.split()


def test_clean_removes_multi_word_phrases():
    """Multi-word greetings ('bom dia', 'e ai') need to be stripped as a
    phrase BEFORE tokenization or the tokens leak through."""
    assert "bom" not in _clean_product_query("bom dia, tem kronos?")
    assert "dia" not in _clean_product_query("bom dia, tem kronos?")
    assert "kronos" in _clean_product_query("bom dia, tem kronos?")


def test_clean_strips_punctuation():
    cleaned = _clean_product_query("tem ?  proteo!!! ama???")
    tokens = cleaned.split()
    assert "proteo" in tokens
    assert "ama" in tokens
    # No punctuation residue.
    for tok in tokens:
        assert tok.isalnum()


def test_clean_empty_query_returns_empty_string():
    assert _clean_product_query("") == ""
    assert _clean_product_query("   ") == ""


def test_clean_only_noise_returns_empty():
    """If the customer wrote only noise ('vocês têm?'), cleaning empties
    the query. The matcher MUST treat this as no-match, not a crash."""
    cleaned = _clean_product_query("vocês têm?")
    # Either empty OR contains no useful tokens.
    assert cleaned.strip() == ""


def test_clean_only_noise_no_match_no_crash():
    """End-to-end: empty-after-clean → status=none."""
    products = [{"id": 1, "name": "Raquete Drop Shot Legacy"}]
    result = match_product_tolerant("vocês têm?", products)
    assert result.status == "none"
    assert result.product is None


# ── _distinctive_tokens also filters noise (defense in depth) ───────────


def test_distinctive_tokens_filters_noise():
    """If a caller bypasses _clean_product_query, _distinctive_tokens
    must STILL filter noise so the token-score layer doesn't degrade."""
    tokens = _distinctive_tokens("Olá! Você tem a nova Kronos?")
    assert tokens == {"kronos"}


# ── End-to-end matcher cases — Felipe's real bug reports ────────────────


_FAKE_CATALOG = [
    # The real catalog produces these names; we use a representative
    # sample so the tests are independent of catalog content.
    {"id": 101, "name": "Raquete De Beach Tennis Ama Sport Kronos 6th Generation 2026"},
    {"id": 102, "name": "Raquete De Beach Tennis Ama Sport Kronos 5th Generation 2025"},
    {"id": 103, "name": "Raquete Ama Sport Proteo 22mm 2026"},
    {"id": 104, "name": "Mochila Raqueteira Fobel Snow Branca"},
    {"id": 105, "name": "Mochila Raqueteira Fobel Snow Preta"},
    {"id": 106, "name": "Raquete Mormaii Sunset Plus 2026"},
    {"id": 107, "name": "Manguito Compressao Beach Tennis Unissex"},
    {"id": 108, "name": "Raquete Drop Shot Legacy 12k 2025"},
]


def test_kronos_matches_after_cleaning():
    """The exact production bug: 'Olá! Você tem a nova Kronos?' must
    resolve to a Kronos product (either single match or ambiguous over
    the 2 Kronos generations)."""
    result = match_product_tolerant("Olá! Você tem a nova Kronos?", _FAKE_CATALOG)
    # Either exact (one Kronos won the score) or ambiguous (both Kronos
    # tied). What we DO NOT accept: status=none, or top result being
    # something else (Mochila).
    assert result.status in ("exact", "fuzzy_high", "ambiguous")
    candidates = result.candidates or ([result.product] if result.product else [])
    names = [(c.get("name") or "").lower() for c in candidates]
    assert any("kronos" in n for n in names), (
        f"Expected Kronos in candidates, got: {names}"
    )


def test_proteo_strong_match_not_fuzzy_low():
    """'procuro uma proteo da ama. quais modelos ama vc tem?' should
    yield a STRONG match (exact / fuzzy_high), not fuzzy_low."""
    result = match_product_tolerant(
        "procuro uma proteo da ama. quais modelos ama vc tem?",
        _FAKE_CATALOG,
    )
    assert result.status in ("exact", "fuzzy_high"), (
        f"Expected strong match, got status={result.status} "
        f"product={result.product and result.product.get('name')!r}"
    )
    assert "proteo" in (result.product.get("name") or "").lower()


def test_proteo_query_about_k_count_still_matches():
    """'detalhes raquete proteo? quantos k ela tem?' — 'quantos k' is
    noise; 'proteo' stays. Should match Proteo."""
    result = match_product_tolerant(
        "detalhes raquete proteo? quantos k ela tem?",
        _FAKE_CATALOG,
    )
    assert result.product is not None
    assert "proteo" in (result.product.get("name") or "").lower()


# ── REGRESSION — existing working cases must keep working ───────────────


def test_regression_mormaii_sunset_plus_still_matches():
    """Sprint 2.6.4 case: 'vocês têm a mormaii sunset plus?' must
    continue to resolve to the Mormaii Sunset Plus."""
    result = match_product_tolerant(
        "vocês têm a mormaii sunset plus?",
        _FAKE_CATALOG,
    )
    assert result.product is not None
    assert "mormaii" in (result.product.get("name") or "").lower()


def test_regression_manguito_still_matches():
    """Single-word product queries must keep working."""
    result = match_product_tolerant("tem manguito?", _FAKE_CATALOG)
    assert result.product is not None
    assert "manguito" in (result.product.get("name") or "").lower()


def test_regression_drop_shot_legacy_still_matches():
    """Multi-token brand+model queries must keep working."""
    result = match_product_tolerant("drop shot legacy 12k", _FAKE_CATALOG)
    assert result.product is not None
    name = (result.product.get("name") or "").lower()
    assert "drop" in name and "shot" in name and "legacy" in name


def test_regression_attribute_strip_still_works():
    """Sprint 2.6.6 case: attribute_inquiry strips 'qual o peso' BEFORE
    calling the matcher. The matcher should then handle the residual
    product name cleanly. We simulate that two-stage flow here."""
    from app.agent.nodes.attribute_inquiry import _strip_attribute_triggers_from_query

    stripped = _strip_attribute_triggers_from_query("qual o peso da Mormaii Sunset?")
    result = match_product_tolerant(stripped, _FAKE_CATALOG)
    assert result.product is not None
    assert "mormaii" in (result.product.get("name") or "").lower()
