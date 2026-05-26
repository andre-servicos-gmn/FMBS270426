import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.agent.graph import build_graph
from app.config import get_settings


@pytest.fixture(autouse=True)
def reset_settings_cache(monkeypatch) -> None:
    """Clear the lru_cache before each test so env patches take effect.

    Sprint 2.5.1 — also force ``BLING_CLIENT_ID`` to empty by default so
    a populated .env in the developer's checkout doesn't accidentally
    route legacy tests through the Bling code path. Tests that want
    Bling enabled re-set it via their own monkeypatch.
    """
    monkeypatch.setenv("BLING_CLIENT_ID", "")
    get_settings.cache_clear()
    yield  # type: ignore[misc]
    get_settings.cache_clear()


@pytest.fixture
def memory_graph():
    """Fresh agent graph compiled with an in-memory checkpointer.

    Tests use this instead of the production Redis singleton so they can run
    in isolation, without needing ``init_checkpointer()`` or a live Redis.
    Each test gets its own ``MemorySaver`` — checkpoints don't leak between
    test cases.
    """
    return build_graph(checkpointer=MemorySaver())
