"""Per-model request adaptation (app/adapters/model_params.py)."""
from app.adapters.model_params import adapt_chat_kwargs, is_reasoning_model

_TOOLS = [{"type": "function", "function": {"name": "buscar_catalogo", "parameters": {}}}]


def _req(model: str, **extra) -> dict:
    return {"model": model, "messages": [], "temperature": 0.3, "max_tokens": 1024, **extra}


def test_classic_model_untouched():
    kw = adapt_chat_kwargs(_req("gpt-4o-mini", tools=_TOOLS))
    assert kw["temperature"] == 0.3
    assert kw["max_tokens"] == 1024
    assert "max_completion_tokens" not in kw
    assert "reasoning_effort" not in kw


def test_reasoning_families_detected():
    assert is_reasoning_model("gpt-5-mini")
    assert is_reasoning_model("gpt-6-luna")
    assert not is_reasoning_model("gpt-4o-mini")
    assert not is_reasoning_model("")


def test_gpt5_renames_tokens_and_drops_temperature():
    kw = adapt_chat_kwargs(_req("gpt-5-mini", tools=_TOOLS))
    assert "temperature" not in kw
    assert "max_tokens" not in kw
    assert kw["max_completion_tokens"] >= 4000
    assert "reasoning_effort" not in kw


def test_gpt6_with_tools_pins_reasoning_none():
    kw = adapt_chat_kwargs(_req("gpt-6-luna", tools=_TOOLS))
    assert "temperature" not in kw
    assert "max_tokens" not in kw
    assert kw["max_completion_tokens"] >= 4000
    assert kw["reasoning_effort"] == "none"


def test_gpt6_without_tools_keeps_default_effort():
    kw = adapt_chat_kwargs(_req("gpt-6-luna", response_format={"type": "json_object"}))
    assert "reasoning_effort" not in kw
    assert kw["max_completion_tokens"] >= 4000
