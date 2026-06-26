"""Per-model parameter adaptation for the OpenAI chat completions API.

Different model families accept different request parameters. The classic
``gpt-4o*`` family takes ``max_tokens`` and a custom ``temperature``. The
``gpt-5*`` reasoning family instead:

  - uses ``max_completion_tokens`` (``max_tokens`` is rejected), and
  - accepts ONLY the default ``temperature`` (1) — passing a custom value
    (e.g. 0.3) returns a 400.

Both the supervisor loop and ``OpenAIClient`` build request kwargs with the
classic names. ``adapt_chat_kwargs`` rewrites those kwargs in place for the
target model so a single ``OPENAI_MODEL`` switch (e.g. to ``gpt-5-mini``)
works without touching every call site. Unknown models are left untouched.

The reasoning-model token budget is bumped to a floor because gpt-5 spends
part of the completion budget on hidden reasoning tokens before emitting the
visible answer; a 1024 cap can leave the visible reply empty.
"""
from __future__ import annotations

from typing import Any

# Floor for gpt-5* completion budget: reasoning tokens are billed against the
# same budget as the visible answer, so a low cap can truncate the reply to "".
_REASONING_TOKEN_FLOOR = 4000


def is_reasoning_model(model: str) -> bool:
    """True for model families that use the reasoning-style request shape."""
    return (model or "").startswith("gpt-5")


def adapt_chat_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Rewrite chat.completions kwargs in place for the model in ``kwargs['model']``.

    For gpt-5*: drop custom ``temperature`` (only default allowed) and rename
    ``max_tokens`` → ``max_completion_tokens`` (raised to a reasoning floor).
    For other models: unchanged.
    """
    model = kwargs.get("model", "")
    if not is_reasoning_model(model):
        return kwargs

    # gpt-5* rejects a custom temperature — drop it so the API uses the default.
    kwargs.pop("temperature", None)

    if "max_tokens" in kwargs:
        budget = kwargs.pop("max_tokens")
        kwargs["max_completion_tokens"] = max(int(budget or 0), _REASONING_TOKEN_FLOOR)

    return kwargs
