"""Production fix — a price-CEILING query ("raquetes até 2 mil") used to come
back as the 8 CHEAPEST products (all R$449-469, same line), so the agent read
like a robot dumping near-identical prices. The fix samples the eligible,
price-ascending products SPREAD across the range so the LLM sees one cheap, one
mid, and one near the cap.

These tests exercise:
  - ``_spread_by_price`` (pure unit, the sampling math)
  - ``buscar_catalogo`` end-to-end with a fake beach-tennis catalog, asserting a
    ``preco_max`` query returns the SPREAD (includes a near-cap item) and that an
    explicit "as mais baratas" (ordenacao=preco_asc) still returns the cheap head.
"""
import json
from unittest.mock import AsyncMock, patch

import pytest

from app.agent.tools_v2 import _price_reais, _spread_by_price, buscar_catalogo


# ── Fake beach-tennis catalog: 8 rackets spread R$300..R$2500 ────────────────

def _racket(pid: int, name: str, price_reais: float) -> dict:
    return {
        "id": pid,
        "name": name,
        "price_cents": int(round(price_reais * 100)),
        "is_raquete_praia": True,
        "categoria_nome": "Raquetes de Praia",
        "marca": "Drop Shot",
        "modelo": name,
        "external_id": str(pid),
    }


def _fake_catalog() -> list[dict]:
    return [
        _racket(1, "Raquete Drop Shot Pentax 3.0 Iniciante", 449.00),
        _racket(2, "Raquete Drop Shot Stage Pro 1.0 BT", 459.00),
        _racket(3, "Raquete Drop Shot Nilo Red", 469.00),
        _racket(4, "Raquete Drop Shot Tiger 2.0", 699.00),
        _racket(5, "Raquete Drop Shot Conqueror 5.0", 999.00),
        _racket(6, "Raquete Drop Shot Explosion Carbon", 1299.00),
        _racket(7, "Raquete Drop Shot Power Pro 12K", 1799.00),
        _racket(8, "Raquete Drop Shot Elite 18K Competition", 2499.00),
    ]


# ════════════════════════════════════════════════════════════════════════════
# _spread_by_price — sampling math (unit)
# ════════════════════════════════════════════════════════════════════════════

def test_spread_includes_cheapest_and_most_expensive():
    products = sorted(_fake_catalog(), key=lambda p: _price_reais(p) or 0)
    picked = _spread_by_price(products, 3)
    prices = [_price_reais(p) for p in picked]
    assert prices[0] == 449.00, "spread must keep the cheapest"
    assert prices[-1] == 2499.00, "spread must keep the most expensive within the list"
    # Middle pick is genuinely in the middle, not clustered at the bottom.
    assert 600.0 <= prices[1] <= 1800.0, prices


def test_spread_returns_all_when_fewer_than_n():
    products = sorted(_fake_catalog()[:2], key=lambda p: _price_reais(p) or 0)
    picked = _spread_by_price(products, 3)
    assert picked == products


def test_spread_preserves_price_order_and_no_dupes():
    products = sorted(_fake_catalog(), key=lambda p: _price_reais(p) or 0)
    picked = _spread_by_price(products, 5)
    ids = [p["id"] for p in picked]
    assert len(ids) == len(set(ids)), "no product picked twice"
    prices = [_price_reais(p) for p in picked]
    assert prices == sorted(prices), "spread keeps ascending price order"


def test_spread_zero_n_is_empty():
    assert _spread_by_price(_fake_catalog(), 0) == []


# ════════════════════════════════════════════════════════════════════════════
# buscar_catalogo — price-ceiling query returns the SPREAD, not the cheap head
# ════════════════════════════════════════════════════════════════════════════

def _patch_snapshot(catalog):
    return patch(
        "app.sync.bling_catalog_cache.get_catalog_snapshot",
        new_callable=AsyncMock,
        return_value=catalog,
    )


@pytest.mark.asyncio
async def test_price_ceiling_returns_spread_not_cheapest_cluster():
    """The headline production bug: 'até 2 mil' returned 8 items all R$449-469.
    Now the result must SPREAD across the range — include a near-cap item."""
    with _patch_snapshot(_fake_catalog()):
        raw = await buscar_catalogo.ainvoke(
            {"preco_max": 2000, "categoria": "beach tennis"}
        )
    results = json.loads(raw)
    assert results, "must return products under the cap"
    # Parse the "R$ 1.799,00" strings back to floats.
    def _to_reais(s: str) -> float:
        return float(s.replace("R$", "").strip().replace(".", "").replace(",", "."))

    prices = [_to_reais(r["preco"]) for r in results]
    assert all(p <= 2000 for p in prices), f"nothing above the cap: {prices}"
    # The R$2499 racket is over the cap; the R$1799 one is the top under it.
    assert max(prices) >= 1299.0, (
        f"result clustered at the bottom (max={max(prices)}); the spread must "
        f"reach toward the R$2000 ceiling"
    )
    assert min(prices) <= 469.0, "spread should still include a cheap option"


@pytest.mark.asyncio
async def test_cheapest_first_intent_keeps_ascending_head():
    """Regression: an explicit 'as mais baratas' (ordenacao=preco_asc) must NOT
    spread — it should still return the genuine cheapest products in order."""
    with _patch_snapshot(_fake_catalog()):
        raw = await buscar_catalogo.ainvoke(
            {"categoria": "beach tennis", "ordenacao": "preco_asc"}
        )
    results = json.loads(raw)
    assert results
    def _to_reais(s: str) -> float:
        return float(s.replace("R$", "").strip().replace(".", "").replace(",", "."))

    prices = [_to_reais(r["preco"]) for r in results]
    assert prices == sorted(prices), "cheapest-first stays ascending"
    assert prices[0] == 449.00, "the cheapest racket must lead the list"
