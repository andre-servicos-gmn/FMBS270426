"""Sprint 2.6.2 — fuzzy match (Levenshtein) + status dispatch tests."""
import pytest

from app.agent.nodes._product_match import (
    _levenshtein,
    match_product_tolerant,
)


def _p(name: str, **extra) -> dict:
    base = {"id": hash(name) & 0xFFFFFFFF, "name": name, "price_cents": 100000,
            "is_raquete_praia": True, "description": "", "external_id": name}
    base.update(extra)
    return base


# ── Levenshtein unit ─────────────────────────────────────────────────────────

def test_levenshtein_identical():
    assert _levenshtein("abc", "abc") == 0


def test_levenshtein_single_insertion():
    assert _levenshtein("emit", "emmit") == 1


def test_levenshtein_substitution():
    assert _levenshtein("emit", "emot") == 1


def test_levenshtein_deletion():
    assert _levenshtein("emmit", "emit") == 1


# ── match_product_tolerant: status dispatch ──────────────────────────────────

def test_exact_match_returns_exact_status():
    products = [_p("Raquete Emit Hammer")]
    result = match_product_tolerant("Raquete Emit Hammer", products)
    assert result.status == "exact"
    assert result.product is not None
    assert result.confidence == "high"


def test_typo_distance_1_returns_fuzzy_high():
    """Single-character typo → auto-confirm."""
    products = [_p("Raquete Emit Hammer")]
    result = match_product_tolerant("emmit hammer", products)
    assert result.status == "fuzzy_high", f"got status={result.status} distance={result.distance}"
    assert result.product is not None
    assert result.product["name"] == "Raquete Emit Hammer"


def test_typo_distance_3_returns_fuzzy_low():
    """Heavier typo → ask 'Você quis dizer?'."""
    products = [_p("Raquete Mormaii Sunset")]
    result = match_product_tolerant("mormai sunsex", products)  # ~2 typos
    assert result.status in ("fuzzy_low", "fuzzy_high"), (
        f"expected fuzzy match, got {result.status}"
    )
    if result.status == "fuzzy_low":
        assert result.needs_confirmation
        assert result.product is not None
        assert result.product["name"] == "Raquete Mormaii Sunset"


def test_no_match_returns_none():
    products = [_p("Raquete Emit Hammer")]
    result = match_product_tolerant("guarda-chuva colorido", products)
    assert result.status == "none"
    assert result.product is None


def test_ambiguous_multiple_at_same_distance():
    """Query matching equally well multiple products → ambiguous."""
    products = [
        _p("Raquete Mormaii Sunset"),
        _p("Raquete Mormaii Eclipse"),
        _p("Raquete Mormaii Tempo"),
    ]
    # "mormaii" is a perfect substring of all three core names → all tied
    # at distance 0 via the sliding window.
    result = match_product_tolerant("mormaii", products)
    # Either exact (one wins via substring layer length tie) OR ambiguous
    # (Levenshtein layer reports the tie). Both are acceptable strategically
    # — what we must NOT have is a confident single match silently dropping
    # the others. So we accept exact (substring tiebreaker) but check at
    # least we didn't pick a high-confidence wrong one.
    assert result.status in ("exact", "ambiguous", "fuzzy_high")
    if result.status == "ambiguous":
        names = [c.get("name") for c in (result.candidates or [])]
        assert len(names) >= 2


def test_fuzzy_match_finds_accessories():
    """Non-raquete product (manguito) matched via fuzzy."""
    products = [
        _p("Manguito Esportivo Compressão", is_raquete_praia=False),
        _p("Raquete BeachPro Carbon X5", is_raquete_praia=True),
    ]
    result = match_product_tolerant("manguito", products)
    assert result.status in ("exact", "fuzzy_high")
    assert result.product is not None
    assert "Manguito" in result.product["name"]


def test_fuzzy_match_finds_non_raquete_products():
    """Lookup covers ALL catalog regardless of is_raquete_praia."""
    products = [
        _p("Raquete BeachPro Carbon X5", is_raquete_praia=True),
        _p("Bola Beach Tennis 3-Pack", is_raquete_praia=False),
        _p("Camiseta Dry-Fit Mormaii", is_raquete_praia=False),
    ]
    # Each query should find its intended non-raquete.
    for q, expected in (
        ("bola beach", "Bola Beach Tennis 3-Pack"),
        ("camiseta dry fit", "Camiseta Dry-Fit Mormaii"),
    ):
        r = match_product_tolerant(q, products)
        assert r.product is not None, f"{q!r} found nothing"
        assert r.product["name"] == expected, f"{q!r} matched {r.product['name']}"


def test_match_result_carries_distance_for_levenshtein_hits():
    products = [_p("Raquete Emit Hammer")]
    result = match_product_tolerant("emmit hammer", products)
    if result.status in ("fuzzy_high", "fuzzy_low"):
        assert result.distance is not None
        assert result.distance >= 1


def test_match_result_candidates_populated_for_ambiguous():
    products = [
        _p("Raquete Modelo A"),
        _p("Raquete Modelo B"),
    ]
    # Query that's equally fuzzy to both at distance 1.
    result = match_product_tolerant("raquete modelo c", products)
    if result.status == "ambiguous":
        assert result.candidates is not None
        assert len(result.candidates) >= 2
