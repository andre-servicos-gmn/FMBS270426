"""Sprint 1.11 — products.category filter (Sprint 2.6: SKIPPED).

The legacy ``products`` table is no longer the source of truth — Sprint
2.6 wired Bling V3 (``bling_products``) as the live catalog. Category-
filter behavior is now expressed via ``bling_products.categoria_nome`` /
``is_raquete_praia`` and tested in ``tests/test_bling_integration.py``.

This file is kept as a skipped reminder so a grep-by-name still finds
the historical coverage.
"""
import pytest

pytestmark = pytest.mark.skip(
    reason="diagnose deprecated in Sprint 2.6 + bling_products replaces products"
)


def test_placeholder():
    pass
