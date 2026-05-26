import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        settings = get_settings()
        # statement_cache_size=0 is required for Supabase Transaction Pooler (PgBouncer
        # in transaction mode). Prepared statements are not supported across pooled connections.
        _engine = create_async_engine(
            settings.database_url,
            echo=False,
            pool_size=5,
            max_overflow=5,
            pool_pre_ping=True,
            connect_args={"statement_cache_size": 0},
        )
    return _engine


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(_get_engine(), expire_on_commit=False)
    return _session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields an async session, commits on clean exit."""
    async with _get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager for internal use (audit_log, jobs, scripts)."""
    async with _get_session_factory()() as session:
        yield session


async def check_supabase_connection() -> bool:
    """Return True if the database is reachable, False otherwise."""
    try:
        async with _get_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("supabase_connection ok")
        return True
    except Exception as exc:
        logger.error("supabase_connection failed: %s", exc)
        return False


async def init_db() -> None:
    """Verify the database connection on startup.

    Schema is managed by SQL migrations in supabase/migrations/ — no DDL here.
    """
    ok = await check_supabase_connection()
    if ok:
        logger.info("database ready")
    else:
        logger.warning("database not reachable — continuing startup without DB")
