"""T1 — buscar_catalogo only returns in-stock products, browse ordered by
``created_at`` DESC (newest first).

Canonical decisions (lead-confirmed):
  - stock filter is GLOBAL: explicit ``stock == 0`` excludes; a missing ``stock``
    key or ``stock is None`` KEEPS the product (fail-open).
  - ``created_at`` DESC ordering applies ONLY to the browse/default path. A
    price-range or ``preco_asc`` query keeps price order (so the existing
    spread tests in test_tools_v2_spread.py stay valid).

All deterministic — the fixture catalog is patched into the snapshot, the tool
runs its real logic, no DB / network / LLM. These are RED until dev-tools lands
the stock filter + recency ordering; that is the intended TDD gate behavior.
"""
import pytest

from tests.evals._helpers import OMIT, dt, make_racket, result_ids, run_buscar

pytestmark = pytest.mark.deterministic


def _browse_catalog() -> list[dict]:
    """4 in-stock rackets, created_at scrambled vs price so a recency sort is
    distinguishable from the legacy price-ascending sort."""
    return [
        make_racket(1, "Raquete Drop Shot Alpha", 999.0, stock=5, created_at=dt(2024, 1, 1)),
        make_racket(2, "Raquete Drop Shot Bravo", 449.0, stock=3, created_at=dt(2026, 6, 1)),
        make_racket(3, "Raquete Drop Shot Charlie", 1299.0, stock=8, created_at=dt(2025, 3, 1)),
        make_racket(4, "Raquete Drop Shot Delta", 599.0, stock=2, created_at=dt(2026, 1, 1)),
    ]


# ── stock filter (global) ────────────────────────────────────────────────────

async def test_excludes_zero_stock():
    cat = _browse_catalog() + [
        make_racket(99, "Raquete Drop Shot Zerada", 399.0, stock=0, created_at=dt(2026, 12, 1))
    ]
    ids = result_ids(await run_buscar(cat, "raquetes"))
    assert "99" not in ids, f"stock=0 product leaked into results: {ids}"


async def test_keeps_positive_stock():
    ids = set(result_ids(await run_buscar(_browse_catalog(), "raquetes")))
    assert {"1", "2", "3", "4"} <= ids, f"in-stock products dropped: {ids}"


async def test_missing_stock_field_kept():
    """Fail-open: a product with NO stock key is kept (sync gap ≠ out of stock)."""
    cat = _browse_catalog() + [
        make_racket(77, "Raquete Drop Shot SemSaldo", 399.0, stock=OMIT, created_at=dt(2026, 12, 1))
    ]
    ids = result_ids(await run_buscar(cat, "raquetes"))
    assert "77" in ids, f"missing-stock product wrongly excluded: {ids}"


async def test_none_stock_kept():
    """Fail-open: an explicit ``stock=None`` is treated like a missing field."""
    cat = _browse_catalog() + [
        make_racket(88, "Raquete Drop Shot SaldoNone", 399.0, stock=None, created_at=dt(2026, 12, 1))
    ]
    ids = result_ids(await run_buscar(cat, "raquetes"))
    assert "88" in ids, f"stock=None product wrongly excluded: {ids}"


async def test_negative_stock_excluded():
    """_has_stock excludes stock <= 0, so a negative balance is out of stock too."""
    cat = _browse_catalog() + [
        make_racket(66, "Raquete Drop Shot Negativa", 399.0, stock=-3, created_at=dt(2026, 12, 1))
    ]
    ids = result_ids(await run_buscar(cat, "raquetes"))
    assert "66" not in ids, f"negative-stock product leaked: {ids}"


async def test_stock_filter_applies_with_price_query():
    """Stock filter is GLOBAL — it also applies under a price ceiling, where the
    zero-stock item is under the cap and would otherwise surface."""
    cat = [
        make_racket(1, "Raquete Drop Shot Sem", 800.0, stock=0, created_at=dt(2025, 1, 1)),
        make_racket(2, "Raquete Drop Shot Com", 900.0, stock=5, created_at=dt(2025, 1, 1)),
    ]
    ids = result_ids(await run_buscar(cat, "raquete", preco_max=1000))
    assert "1" not in ids, f"zero-stock leaked under price cap: {ids}"
    assert "2" in ids, f"in-stock product missing under price cap: {ids}"


async def test_stock_filter_flag_off_shows_everything(monkeypatch):
    """Kill-switch: TOOLS_V2_FILTER_STOCK=false disables the filter, so even a
    stock=0 product is shown (the toggle dev-tools shipped)."""
    from app.config import get_settings

    monkeypatch.setenv("TOOLS_V2_FILTER_STOCK", "false")
    get_settings.cache_clear()
    cat = _browse_catalog() + [
        make_racket(55, "Raquete Drop Shot FlagOff", 399.0, stock=0, created_at=dt(2026, 12, 1))
    ]
    ids = result_ids(await run_buscar(cat, "raquetes"))
    assert "55" in ids, f"kill-switch off must show zero-stock too: {ids}"


async def test_empty_when_all_zero_stock():
    """Whole catalog out of stock → empty result (agent then answers honestly,
    never invents)."""
    cat = [
        make_racket(i, f"Raquete Drop Shot {i}", 500.0 + i, stock=0, created_at=dt(2025, 1, i))
        for i in range(1, 4)
    ]
    results = await run_buscar(cat, "raquetes")
    assert results == [], f"expected empty, got: {result_ids(results)}"


# ── recency ordering (browse/default only) ───────────────────────────────────

async def test_orders_by_created_at_desc():
    """A bare browse returns newest-first by created_at.
    Expected order: 2 (2026-06) > 4 (2026-01) > 3 (2025-03) > 1 (2024-01)."""
    ids = result_ids(await run_buscar(_browse_catalog(), "raquetes"))
    assert ids == ["2", "4", "3", "1"], f"browse not ordered by created_at DESC: {ids}"


async def test_recency_order_independent_of_price():
    """Prices scrambled so a created_at-DESC order is NOT the price-ascending
    order — proves the browse sorts by recency, not price."""
    cat = [
        make_racket(1, "Raquete Drop Shot A", 2000.0, stock=5, created_at=dt(2026, 6, 1)),
        make_racket(2, "Raquete Drop Shot B", 500.0, stock=5, created_at=dt(2025, 1, 1)),
        make_racket(3, "Raquete Drop Shot C", 1500.0, stock=5, created_at=dt(2024, 1, 1)),
    ]
    ids = result_ids(await run_buscar(cat, "raquetes"))
    # created_at DESC = [1, 2, 3]; price-ASC would be [2, 3, 1].
    assert ids == ["1", "2", "3"], f"order followed price, not created_at: {ids}"


async def test_stock_and_recency_combined():
    """Zero-stock excluded AND the survivors come back newest-first."""
    cat = [
        make_racket(1, "Raquete Drop Shot A", 999.0, stock=0, created_at=dt(2026, 6, 1)),  # newest, no stock
        make_racket(2, "Raquete Drop Shot B", 999.0, stock=4, created_at=dt(2025, 1, 1)),
        make_racket(3, "Raquete Drop Shot C", 999.0, stock=4, created_at=dt(2026, 1, 1)),
    ]
    ids = result_ids(await run_buscar(cat, "raquetes"))
    assert "1" not in ids, f"zero-stock newest leaked: {ids}"
    assert ids == ["3", "2"], f"survivors not newest-first: {ids}"
