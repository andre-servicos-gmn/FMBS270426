"""Sprint 2.6.4 — token-score match layer.

The token-based matcher was added because raw Levenshtein blew up against
long catalog names: "drop shot legacy" vs "Raquete de beach tennis DROP
SHOT LEGACY SOFT 3.0 2026 12K" has a huge edit distance even though every
distinctive query token (drop, shot, legacy) appears in the name. The new
layer scores by fraction of distinctive query tokens that land in the
product name, which is length-independent.
"""
from app.agent.nodes._product_match import (
    _distinctive_tokens,
    _STOP_WORDS,
    match_product_tolerant,
)


def _p(name: str, **extra) -> dict:
    return {
        "id": abs(hash(name)) & 0xFFFFFFFF,
        "name": name,
        "price_cents": 100000,
        "is_raquete_praia": True,
        "description": "",
        "external_id": name,
        **extra,
    }


# ── _distinctive_tokens helper ───────────────────────────────────────────────

def test_distinctive_tokens_filters_stop_words():
    out = _distinctive_tokens("vocês tem a calça da nicole nobile?")
    assert "calca" in out
    assert "nicole" in out
    assert "nobile" in out
    # Stop words / WhatsApp filler must NOT survive.
    for stop in ("voces", "tem", "da"):
        assert stop not in out


def test_distinctive_tokens_filters_generic_catalog_words():
    out = _distinctive_tokens("raquete beach tennis carbon pro")
    # "raquete", "beach", "tennis", "carbon", "pro" are all generic.
    assert not out, f"expected empty set, got {out}"


def test_distinctive_tokens_short_tokens_filtered():
    out = _distinctive_tokens("a x v 2 1 ab")
    assert not out


# ── Token-match smoke tests against the user's failure cases ─────────────────

def test_token_match_drop_shot_legacy_finds_legacy_soft():
    products = [
        _p("Raquete de beach tennis DROP SHOT LEGACY SOFT 3.0 2026 12K"),
        _p("Raquete de beach tennis DROP SHOT LEGACY SOFT 2.0 2025 12K"),
        _p("Raquete Mormaii Sunset"),
    ]
    result = match_product_tolerant("drop shot legacy", products)
    # All 3 distinctive tokens (drop, shot, legacy) match BOTH Legacy
    # products → ambiguous with both as candidates.
    assert result.status == "ambiguous"
    names = [(c.get("name") or "") for c in (result.candidates or [])]
    assert any("LEGACY SOFT 3.0" in n for n in names)
    assert any("LEGACY SOFT 2.0" in n for n in names)


def test_token_match_ops_errei_drop_shot_legacy_same_result():
    """Sprint 2.6.4 — filler tokens 'ops', 'errei', 'era' must not break match."""
    products = [
        _p("Raquete de beach tennis DROP SHOT LEGACY SOFT 3.0 2026 12K"),
    ]
    result = match_product_tolerant("ops errei era drop shot legacy", products)
    assert result.status in ("exact", "ambiguous", "fuzzy_high")
    if result.product:
        assert "LEGACY" in result.product["name"]
    else:
        # ambiguous (single candidate at score 1.0 is uncommon, but allowed)
        assert any("LEGACY" in (c.get("name") or "") for c in (result.candidates or []))


def test_token_match_nicole_nobile_finds_calca():
    products = [
        _p("Calça Legging Nicole Nobile Beach Tennis Feminina Fitness M Azul"),
        _p("Calça Legging Nicole Nobile Beach Tennis Feminina Fitness EG Azul"),
        _p("Raquete BeachPro Carbon X5"),
    ]
    result = match_product_tolerant("tem calça da nicole nobile?", products)
    # 2 calça-Nicole products tied at score 1.0 → ambiguous.
    assert result.status == "ambiguous"
    assert all("Nicole Nobile" in (c.get("name") or "")
               for c in (result.candidates or []))


def test_token_match_mormaii_sunset_returns_ambiguous():
    products = [
        _p("Raquete Beach Tennis Mormaii Sunset Plus Carbono 3k"),
        _p("Raquete Mormaii Sunset Eclipse"),
        _p("Raquete BeachPro Carbon X5"),
    ]
    result = match_product_tolerant("mormaii sunset", products)
    assert result.status == "ambiguous"
    cands = [c.get("name") for c in (result.candidates or [])]
    assert any("Sunset Plus" in n for n in cands)
    assert any("Sunset Eclipse" in n for n in cands)


def test_token_match_kit_bolinhas_finds_bola_with_72():
    """``kit`` is generic; ``bolinhas`` is the distinctive token."""
    products = [
        _p("Bola Beach Tennis Drop Shot ITF Stage 2 Pro Com 72 Bolinhas"),
        _p("Bola Beach Tennis Drop Shot ITF Stage 2 Pro Com 10 Bolinhas"),
        _p("Raquete BeachPro Carbon X5"),
    ]
    result = match_product_tolerant("kit de bolinhas", products)
    assert result.status == "ambiguous"
    cands = [c.get("name") for c in (result.candidates or [])]
    assert any("72 Bolinhas" in n for n in cands)
    assert any("10 Bolinhas" in n for n in cands)


def test_token_match_score_threshold_high():
    """Score == 1.0 single product → exact (token method)."""
    products = [
        _p("Manguito Esportivo Compressão"),
        _p("Raquete BeachPro Carbon X5"),
    ]
    result = match_product_tolerant("manguito", products)
    assert result.status == "exact"
    assert result.product is not None
    assert "Manguito" in result.product["name"]


def test_token_match_score_low_path_below_threshold():
    """Score < 0.7 → falls through (no token-layer match)."""
    products = [
        _p("Raquete BeachPro Carbon X5"),
    ]
    # Query has 3 distinctive tokens, only 1 in name → score 0.33 < 0.7.
    result = match_product_tolerant("camiseta dryfit feminina", products)
    assert result.status == "none"


def test_levenshtein_normalized_handles_long_names():
    """Normalized distance: 'mormai sunset' vs core 'mormaii sunset plus
    carbono 3k' is HIGH absolute distance but LOW normalized; the
    token-score layer catches it first anyway."""
    products = [
        _p("Raquete Beach Tennis Mormaii Sunset Plus Carbono 3k"),
    ]
    result = match_product_tolerant("mormai sunset", products)
    # Either the token layer (sunset matched, mormaii close → fuzzy)
    # OR Levenshtein normalized fires. We accept any non-none status.
    assert result.status != "none", (
        f"long-name match must not fall through to none, got {result.status}"
    )


def test_token_match_logs_score(caplog):
    """The match log line includes ``score=`` when token layer fires."""
    import logging
    products = [_p("Manguito Esportivo Compressão")]
    with caplog.at_level(logging.INFO, logger="app.agent.nodes._product_match"):
        match_product_tolerant("manguito", products)
    log_text = "\n".join(r.getMessage() for r in caplog.records)
    assert "method=token" in log_text
    assert "score=" in log_text
