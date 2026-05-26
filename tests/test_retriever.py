"""Tests for app/rag/retriever.py.

All tests mock both embed_text and the SQLAlchemy session so that no real
database or OpenAI key is needed in CI.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.rag.retriever import search_products

_FAKE_EMBEDDING = [0.1] * 1536

# Minimal fake rows returned by the mocked SQL function
_BEACH_INICIANTE = {
    "id": "aaaaaaaa-0000-0000-0000-000000000001",
    "external_id": "BT001",
    "name": "Raquete BeachPro Starter",
    "sport": "beach_tennis",
    "level": "iniciante",
    "price_cents": 59900,
    "stock": 10,
    "description": "Raquete de beach tennis para iniciantes",
    "weight_g": 350,
    "balance": "médio",
    "material": "fibra de vidro",
    "url": None,
    "image_url": None,
    "updated_at": None,
    "is_active": True,
    "similarity": 0.94,
}

_BEACH_AVANCADO = {
    **_BEACH_INICIANTE,
    "id": "aaaaaaaa-0000-0000-0000-000000000002",
    "external_id": "BT002",
    "name": "Raquete BeachPro Carbon Elite",
    "level": "avançado",
    "price_cents": 189900,
    "similarity": 0.80,
}

_PADEL_INICIANTE = {
    **_BEACH_INICIANTE,
    "id": "aaaaaaaa-0000-0000-0000-000000000003",
    "external_id": "PD001",
    "name": "Raquete PadelMax Entry",
    "sport": "padel",
    "price_cents": 37900,
    "similarity": 0.72,
}

_SEM_ESTOQUE = {
    **_BEACH_INICIANTE,
    "id": "aaaaaaaa-0000-0000-0000-000000000004",
    "external_id": "BT003",
    "name": "Raquete AirBlast Promo",
    "stock": 0,
    "similarity": 0.91,
}


def _mock_session(rows: list[dict]) -> MagicMock:
    """Build an AsyncMock session whose execute() returns a fake result set."""
    result = MagicMock()
    result.mappings.return_value.all.return_value = rows
    session = AsyncMock()
    session.execute.return_value = result
    return session


# ── basic search ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_returns_list_of_dicts():
    session = _mock_session([_BEACH_INICIANTE])
    with patch("app.rag.retriever.embed_text", return_value=_FAKE_EMBEDDING):
        results = await search_products(session, "raquete beach tennis iniciante")
    assert isinstance(results, list)
    assert results[0] == _BEACH_INICIANTE


@pytest.mark.asyncio
async def test_empty_source_returns_empty_list():
    session = _mock_session([])
    with patch("app.rag.retriever.embed_text", return_value=_FAKE_EMBEDDING):
        results = await search_products(session, "raquete beach tennis")
    assert results == []


# ── SQL function is called correctly ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_sql_function_name_in_query():
    session = _mock_session([_BEACH_INICIANTE])
    with patch("app.rag.retriever.embed_text", return_value=_FAKE_EMBEDDING):
        await search_products(session, "raquete")

    call = session.execute.call_args
    sql_text = call[0][0].text
    assert "search_products" in sql_text


@pytest.mark.asyncio
async def test_embedding_formatted_as_vector_literal():
    session = _mock_session([])
    with patch("app.rag.retriever.embed_text", return_value=_FAKE_EMBEDDING):
        await search_products(session, "qualquer coisa")

    params = session.execute.call_args[0][1]
    # Must be a bracketed list of floats: '[0.1,0.1,...]'
    assert params["embedding"].startswith("[")
    assert params["embedding"].endswith("]")


@pytest.mark.asyncio
async def test_default_params_when_no_filters():
    session = _mock_session([])
    with patch("app.rag.retriever.embed_text", return_value=_FAKE_EMBEDDING):
        await search_products(session, "raquete", k=5)

    params = session.execute.call_args[0][1]
    assert params["sport"] is None
    assert params["max_price"] is None
    assert params["min_stock"] == 1
    assert params["k"] == 5


# ── sport filter ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sport_filter_passed_to_sql():
    session = _mock_session([_BEACH_INICIANTE])
    with patch("app.rag.retriever.embed_text", return_value=_FAKE_EMBEDDING):
        results = await search_products(
            session, "raquete", filters={"sport": "beach_tennis"}
        )

    params = session.execute.call_args[0][1]
    assert params["sport"] == "beach_tennis"
    # Returned rows are whatever the DB/mock returns — ordering is DB's job
    assert len(results) == 1


@pytest.mark.asyncio
async def test_sport_filter_excludes_other_sports():
    """When sport=padel, beach_tennis rows should not appear (DB enforces this;
    we verify the filter value was forwarded correctly)."""
    session = _mock_session([_PADEL_INICIANTE])  # DB already filtered
    with patch("app.rag.retriever.embed_text", return_value=_FAKE_EMBEDDING):
        results = await search_products(
            session, "raquete padel", filters={"sport": "padel"}
        )

    params = session.execute.call_args[0][1]
    assert params["sport"] == "padel"
    assert all(r["sport"] == "padel" for r in results)


# ── price filter ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_max_price_filter_passed_to_sql():
    """R$800 = 80000 cents."""
    session = _mock_session([_BEACH_INICIANTE])
    with patch("app.rag.retriever.embed_text", return_value=_FAKE_EMBEDDING):
        await search_products(
            session,
            "raquete iniciante beach tenis até 800 reais",
            filters={"max_price_cents": 80000},
        )

    params = session.execute.call_args[0][1]
    assert params["max_price"] == 80000


@pytest.mark.asyncio
async def test_query_beach_iniciante_under_800_returns_correct_products():
    """Scenario: user asks for beach tennis beginner racket under R$800.
    DB returns only the matching product — we verify it passes through."""
    session = _mock_session([_BEACH_INICIANTE])
    with patch("app.rag.retriever.embed_text", return_value=_FAKE_EMBEDDING):
        results = await search_products(
            session,
            "raquete pra iniciante de beach tenis até 800 reais",
            filters={"sport": "beach_tennis", "max_price_cents": 80000},
        )

    assert len(results) == 1
    assert results[0]["sport"] == "beach_tennis"
    assert results[0]["level"] == "iniciante"
    assert results[0]["price_cents"] <= 80000


# ── stock filter ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_min_stock_filter_passed_to_sql():
    session = _mock_session([_BEACH_INICIANTE])
    with patch("app.rag.retriever.embed_text", return_value=_FAKE_EMBEDDING):
        await search_products(
            session, "raquete", filters={"min_stock": 3}
        )

    params = session.execute.call_args[0][1]
    assert params["min_stock"] == 3


@pytest.mark.asyncio
async def test_out_of_stock_excluded_by_db():
    """DB filters stock=0 — mock returns only in-stock rows."""
    session = _mock_session([_BEACH_INICIANTE])  # sem_estoque was excluded by DB
    with patch("app.rag.retriever.embed_text", return_value=_FAKE_EMBEDDING):
        results = await search_products(
            session, "raquete", filters={"min_stock": 1}
        )

    assert all(r["stock"] >= 1 for r in results)


# ── k parameter ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_k_parameter_forwarded():
    session = _mock_session([_BEACH_INICIANTE, _BEACH_AVANCADO])
    with patch("app.rag.retriever.embed_text", return_value=_FAKE_EMBEDDING):
        await search_products(session, "raquete", k=10)

    params = session.execute.call_args[0][1]
    assert params["k"] == 10


@pytest.mark.asyncio
async def test_multiple_results_returned():
    session = _mock_session([_BEACH_INICIANTE, _BEACH_AVANCADO, _PADEL_INICIANTE])
    with patch("app.rag.retriever.embed_text", return_value=_FAKE_EMBEDDING):
        results = await search_products(session, "raquete", k=3)

    assert len(results) == 3
    assert results[0]["similarity"] >= results[1]["similarity"]  # mock preserves order


# ── combined filters ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_combined_sport_price_stock_filters():
    session = _mock_session([_BEACH_INICIANTE])
    with patch("app.rag.retriever.embed_text", return_value=_FAKE_EMBEDDING):
        results = await search_products(
            session,
            "raquete beach tennis intermediário",
            filters={
                "sport": "beach_tennis",
                "max_price_cents": 100000,
                "min_stock": 5,
            },
            k=3,
        )

    params = session.execute.call_args[0][1]
    assert params["sport"] == "beach_tennis"
    assert params["max_price"] == 100000
    assert params["min_stock"] == 5
    assert params["k"] == 3
    assert len(results) == 1
