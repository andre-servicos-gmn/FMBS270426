"""buscar_catalogo: "bola/mochila de beach tennis" must not be filtered to rackets.

The model passes categoria="beach tennis" for "bola de beach tennis"; the
racket-only curated flag used to hide every ball/bag and the agent answered
"não temos bola".
"""
import json
from unittest.mock import patch

import pytest

from app.agent.tools_v2 import buscar_catalogo

_CATALOG = [
    {"id": 1, "name": "Raquete Beach Tennis Drop Shot Pentax 3.0", "price_cents": 44900,
     "is_raquete_praia": True, "categoria_nome": "Raquetes de Praia", "stock": 5},
    {"id": 2, "name": "Bola Beach Tennis Head Pack c/ 3", "price_cents": 8990,
     "is_raquete_praia": False, "categoria_nome": "Bolas", "stock": 30},
    {"id": 3, "name": "Raqueteira Mochila Drop Shot Tour", "price_cents": 39900,
     "is_raquete_praia": False, "categoria_nome": "RAQUETEIRAS MOCHILA", "stock": 5},
    {"id": 4, "name": "Óculos Mormaii Sunset", "price_cents": 29900,
     "is_raquete_praia": False, "categoria_nome": "Oculos", "stock": 5},
]


async def _snapshot():
    return list(_CATALOG)


async def _search(**kwargs) -> list[str]:
    with patch("app.sync.bling_catalog_cache.get_catalog_snapshot", _snapshot):
        raw = await buscar_catalogo.ainvoke(kwargs)
    data = json.loads(raw)
    return [p["nome"] for p in data] if isinstance(data, list) else []


@pytest.mark.asyncio
async def test_ball_with_beach_tennis_category_finds_ball():
    names = await _search(consulta="bola", categoria="beach tennis")
    assert names and names[0].startswith("Bola Beach Tennis")


@pytest.mark.asyncio
async def test_bag_with_beach_tennis_category_finds_bag():
    names = await _search(consulta="mochila beach tennis", categoria="beach tennis")
    assert names and "Mochila" in names[0]


@pytest.mark.asyncio
async def test_accented_accessory_word_finds_product():
    names = await _search(consulta="óculos", categoria="beach tennis")
    assert names and names[0].startswith("Óculos")


@pytest.mark.asyncio
async def test_racket_browse_with_category_still_rackets_only():
    names = await _search(consulta="raquete", categoria="beach tennis")
    assert names == ["Raquete Beach Tennis Drop Shot Pentax 3.0"]
