import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.admin import router as admin_router
from app.api.bling import router as bling_router
from app.api.lgpd import router as lgpd_router
from app.api.webhook import router as webhook_router
from app.config import configure_logging, get_settings

settings = get_settings()
configure_logging(settings)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    logger.info("Application starting up (env=%s)", settings.app_env)

    try:
        from app.storage.db import init_db
        await init_db()
    except Exception as exc:
        logger.warning("DB not reachable at startup: %s", exc)

    from app.agent.checkpointer import close_checkpointer, init_checkpointer
    await init_checkpointer()

    from app.jobs.catalog_sync import start_scheduler, stop_scheduler
    start_scheduler()

    yield

    stop_scheduler()
    await close_checkpointer()
    logger.info("Application shutting down")


app = FastAPI(
    title="beachtenis-agent",
    description="WhatsApp conversational agent for Beach Tennis / Padel franchise",
    version="0.1.0",
    lifespan=_lifespan,
)

app.include_router(webhook_router)
app.include_router(admin_router)
app.include_router(lgpd_router)
app.include_router(bling_router)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "env": settings.app_env})
