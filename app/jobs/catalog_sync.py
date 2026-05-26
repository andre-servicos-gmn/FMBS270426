"""APScheduler job — periodic catalog sync."""
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import get_settings

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None


async def _run_sync() -> None:
    """Wrapper called by APScheduler — logs result or error."""
    from app.rag.ingestion import sync_catalog

    logger.info("catalog_sync starting")
    try:
        stats = await sync_catalog()
        logger.info(
            "catalog_sync finished inserted=%d updated=%d deactivated=%d total_source=%d",
            stats.get("inserted", 0),
            stats.get("updated", 0),
            stats.get("deactivated", 0),
            stats.get("total_source", 0),
        )
    except Exception as exc:
        logger.error("catalog_sync failed: %s", exc)


async def _run_bling_sync() -> None:
    """Sprint 2.5 — daily Bling full sync (UTC ``BLING_SYNC_HOUR``)."""
    from app.adapters.bling import BlingNotAuthorizedError
    from app.sync.bling_sync import BlingSync

    logger.info("bling_daily_sync starting")
    try:
        stats = await BlingSync().full_sync(only_active=True)
        logger.info("bling_daily_sync finished %s", stats)
    except BlingNotAuthorizedError:
        logger.warning("bling_daily_sync skipped — not authorized")
    except Exception as exc:
        logger.error("bling_daily_sync failed: %s", exc)


def start_scheduler() -> AsyncIOScheduler:
    """Create, configure and start the scheduler. Returns the instance.

    The legacy catalog_sync job only registers when ``CATALOG_API_URL`` is set.
    The Bling daily sync only registers when ``BLING_CLIENT_ID`` is set.
    """
    global _scheduler
    settings = get_settings()

    _scheduler = AsyncIOScheduler()

    if settings.catalog_api_url:
        _scheduler.add_job(
            _run_sync,
            CronTrigger.from_crontab(settings.catalog_sync_cron),
            id="catalog_sync",
            replace_existing=True,
            misfire_grace_time=300,
        )
        logger.info("legacy catalog_sync registered (cron=%s)", settings.catalog_sync_cron)
    else:
        logger.info("legacy catalog sync disabled (no CATALOG_API_URL)")

    # Sprint 2.5 — daily Bling sync at BLING_SYNC_HOUR UTC, minute 0.
    if settings.bling_client_id:
        _scheduler.add_job(
            _run_bling_sync,
            CronTrigger(hour=settings.bling_sync_hour, minute=0),
            id="bling_daily_sync",
            replace_existing=True,
            misfire_grace_time=600,
        )
        logger.info(
            "bling_daily_sync registered (hour=%s UTC)", settings.bling_sync_hour
        )
    else:
        logger.info("bling daily sync disabled (no BLING_CLIENT_ID)")

    _scheduler.start()
    return _scheduler


def stop_scheduler() -> None:
    """Gracefully shut down the scheduler on app teardown."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("scheduler stopped")
    _scheduler = None
