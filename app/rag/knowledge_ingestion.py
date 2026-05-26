"""Knowledge base ingestion: upsert documents with embeddings into knowledge_base table."""
import logging
import uuid
from typing import Any

from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.rag.embeddings import embed_batch
from app.storage.db import get_session
from app.storage.models import KnowledgeBase

logger = logging.getLogger(__name__)


def _embedding_text(doc: dict[str, Any]) -> str:
    """Concatenate title + content for embedding — gives better semantic coverage."""
    return f"{doc['title']}\n\n{doc['content']}"


async def upsert_documents(documents: list[dict[str, Any]]) -> dict[str, int]:
    """Embed and upsert a list of knowledge base documents.

    Each document dict must have 'title', 'content', and 'category'.
    Optional keys: 'source', 'metadata', 'is_active'.

    Upsert key is (title, category) — matching the DB unique constraint.
    Returns {'upserted': N, 'total': N}.
    """
    if not documents:
        logger.warning("knowledge_ingestion: empty document list, nothing to do")
        return {"upserted": 0, "total": 0}

    texts = [_embedding_text(d) for d in documents]
    embeddings = await embed_batch(texts)

    async with get_session() as session:
        for doc, embedding in zip(documents, embeddings):
            stmt = (
                pg_insert(KnowledgeBase)
                .values(
                    id=uuid.uuid4(),
                    title=doc["title"],
                    content=doc["content"],
                    category=doc.get("category", "general"),
                    source=doc.get("source", "manual"),
                    embedding=embedding,
                    is_active=doc.get("is_active", True),
                    metadata_=doc.get("metadata"),
                )
                .on_conflict_do_update(
                    constraint="kb_title_category_unique",
                    set_={
                        "content": doc["content"],
                        "source": doc.get("source", "manual"),
                        "embedding": embedding,
                        "is_active": doc.get("is_active", True),
                        "metadata_": doc.get("metadata"),
                    },
                )
            )
            await session.execute(stmt)
        await session.commit()

    logger.info("knowledge_ingestion upserted=%d total=%d", len(documents), len(documents))
    return {"upserted": len(documents), "total": len(documents)}


async def deactivate_document(title: str, category: str) -> bool:
    """Soft-deactivate a document by title + category. Returns True if found."""
    from sqlalchemy import update

    async with get_session() as session:
        result = await session.execute(
            update(KnowledgeBase)
            .where(KnowledgeBase.title == title, KnowledgeBase.category == category)
            .values(is_active=False)
        )
        await session.commit()
        return (result.rowcount or 0) > 0
