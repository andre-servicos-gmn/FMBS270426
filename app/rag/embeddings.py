"""OpenAI text-embedding-3-small embeddings (1536 dims)."""
import logging

from openai import AsyncOpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "text-embedding-3-small"
_BATCH_SIZE = 100

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=get_settings().openai_api_key)
    return _client


async def embed_text(text: str) -> list[float]:
    """Embed a single string. Returns a 1536-dim vector."""
    results = await embed_batch([text])
    return results[0]


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings, batching at most _BATCH_SIZE per API call.

    Returns embeddings in the same order as the input.
    """
    if not texts:
        return []

    client = _get_client()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        response = await client.embeddings.create(model=_EMBEDDING_MODEL, input=batch)
        # API may return items out of order — sort by index to be safe
        ordered = sorted(response.data, key=lambda e: e.index)
        all_embeddings.extend(e.embedding for e in ordered)
        logger.info(
            "embedded batch offset=%d size=%d model=%s",
            i,
            len(batch),
            _EMBEDDING_MODEL,
        )

    return all_embeddings
