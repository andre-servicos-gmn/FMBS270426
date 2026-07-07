"""Sprint 2.5 — Bling OAuth + webhook endpoints.

OAuth flow:
- GET  /bling/oauth/authorize  → redirects Andre to Bling's authorization page.
                                  Stores ``state`` in Redis (5 min TTL).
- GET  /bling/oauth/callback   → receives the code, validates state, swaps
                                  for tokens, persists them, replies OK page.

Webhook:
- POST /bling/webhook          → Bling pushes product create/update/delete.
                                  We validate the HMAC, dedupe by event
                                  timestamp, and enqueue an async sync.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.adapters.bling import BlingClient, BlingError, BlingNotAuthorizedError
from app.config import get_settings
from app.sync.bling_catalog_cache import invalidate_catalog
from app.sync.bling_repo import record_webhook_event
from app.sync.bling_stock import invalidate_stock
from app.sync.bling_sync import BlingSync

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/bling", tags=["bling"])

_STATE_PREFIX = "bling:oauth:state:"
_STATE_TTL_S = 300  # 5 minutes


def _get_redis():
    """Local helper so tests can monkeypatch easily.

    Sprint 2.6.5 — resilient client (keepalive + retry + health_check).
    """
    from app.storage.redis_resilient import make_resilient_redis
    return make_resilient_redis(get_settings().redis_url)


# ── OAuth authorize / callback ──────────────────────────────────────────

@router.get("/oauth/authorize")
async def oauth_authorize() -> RedirectResponse:
    """Generate a random state, store it, redirect Andre to Bling's prompt."""
    settings = get_settings()
    if not settings.bling_client_id:
        raise HTTPException(status_code=500, detail="BLING_CLIENT_ID not configured")

    state = secrets.token_urlsafe(32)
    redis = _get_redis()
    try:
        await redis.setex(f"{_STATE_PREFIX}{state}", _STATE_TTL_S, "1")
    finally:
        await redis.aclose()

    url = BlingClient().get_authorize_url(state)
    logger.info("bling_oauth_authorize redirect state=%.8s", state)
    return RedirectResponse(url=url, status_code=302)


@router.get("/oauth/callback", response_class=HTMLResponse)
async def oauth_callback(
    code: str = Query(...),
    state: str = Query(...),
    error: str | None = Query(default=None),
) -> HTMLResponse:
    if error:
        logger.error("bling_oauth_callback bling_error=%s", error)
        return HTMLResponse(
            f"<h1>❌ Erro do Bling: {error}</h1>", status_code=400
        )

    redis = _get_redis()
    try:
        key = f"{_STATE_PREFIX}{state}"
        stored = await redis.get(key)
        if not stored:
            logger.warning("bling_oauth_callback invalid_or_expired_state")
            return HTMLResponse(
                "<h1>❌ State inválido ou expirado.</h1>"
                "<p>Tente de novo em /bling/oauth/authorize.</p>",
                status_code=400,
            )
        await redis.delete(key)
    finally:
        await redis.aclose()

    try:
        client = BlingClient()
        await client.exchange_code_for_token(code)
    except BlingError as exc:
        logger.error("bling_oauth_callback exchange_failed: %s", exc)
        return HTMLResponse(f"<h1>❌ Falha ao trocar token: {exc}</h1>", status_code=400)

    return HTMLResponse(
        "<h1>✅ Bling conectado!</h1>"
        "<p>Agora você pode rodar <code>scripts/bling_initial_sync.py</code>.</p>"
    )


# ── Webhook ─────────────────────────────────────────────────────────────

def _verify_hmac(body: bytes, signature: str | None, secret: str) -> bool:
    """Compare-digest the HMAC-SHA256 hex digest of the raw body."""
    if not signature:
        return False
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature.strip())


def _parse_event(payload: dict[str, Any]) -> tuple[int | None, str, datetime]:
    """Best-effort extraction of (product_id, event_kind, event_ts) from a
    Bling webhook payload. Different builds use slightly different shapes;
    we accept several.
    """
    data = payload.get("data") or payload
    pid_raw = (
        data.get("id")
        or data.get("idProduto")
        or data.get("productId")
        or (data.get("produto") or {}).get("id")
    )
    pid = int(pid_raw) if pid_raw is not None else None

    event_kind = (
        payload.get("event")
        or payload.get("eventType")
        or payload.get("tipo")
        or "product.updated"
    )

    ts_raw = (
        payload.get("timestamp")
        or payload.get("dataHora")
        or payload.get("occurredAt")
    )
    if ts_raw:
        try:
            event_ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            event_ts = datetime.now(timezone.utc)
    else:
        event_ts = datetime.now(timezone.utc)
    if event_ts.tzinfo is None:
        event_ts = event_ts.replace(tzinfo=timezone.utc)
    return pid, str(event_kind), event_ts


async def _process_webhook_event(payload: dict[str, Any]) -> None:
    """Background task: deduplicate + apply the event to the local catalog."""
    produto_id, event_kind, event_ts = _parse_event(payload)
    if produto_id is None:
        logger.warning("bling_webhook missing_product_id payload_keys=%s", sorted(payload.keys()))
        return

    is_newest = await record_webhook_event(produto_id, event_kind, event_ts, payload)
    if not is_newest:
        return

    sync = BlingSync()
    try:
        if "delete" in event_kind.lower() or event_kind.endswith("deleted"):
            await sync.delete_product(produto_id)
            logger.info("bling_webhook delete applied id=%s", produto_id)
        else:
            outcome = await sync.sync_single_product(produto_id)
            logger.info(
                "bling_webhook %s applied id=%s outcome=%s",
                event_kind, produto_id, outcome,
            )
    except BlingNotAuthorizedError:
        logger.warning("bling_webhook skipped — not authorized yet (id=%s)", produto_id)
        return
    except Exception as exc:
        logger.exception("bling_webhook_apply_failed id=%s: %s", produto_id, exc)
        return

    # The event changed the local catalog — drop the caches so the agent sees
    # it NOW instead of after the TTL: the per-product live-stock cache (up to
    # 5 min stale) and the in-memory catalog snapshot (up to 60 s stale). Both
    # cache modules promised this wiring in their docstrings; a cache blip must
    # never fail the webhook, hence the guard.
    try:
        await invalidate_stock(produto_id)
        invalidate_catalog()
    except Exception as exc:  # noqa: BLE001 — cache invalidation is best-effort
        logger.warning("bling_webhook cache_invalidation_failed id=%s: %s", produto_id, exc)


@router.post("/webhook")
async def bling_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    settings = get_settings()
    body = await request.body()

    secret = settings.bling_webhook_secret
    sig = request.headers.get("X-Bling-Signature") or request.headers.get("x-bling-signature")
    if secret:
        if not _verify_hmac(body, sig, secret):
            logger.warning("bling_webhook invalid_signature")
            raise HTTPException(status_code=401, detail="invalid signature")
    else:
        logger.warning(
            "bling_webhook_no_secret_configured — accepting without verification"
        )

    try:
        payload = await request.json()
    except Exception:
        logger.warning("bling_webhook invalid_json body_len=%d", len(body))
        raise HTTPException(status_code=400, detail="invalid JSON")

    background_tasks.add_task(_process_webhook_event, payload)
    return {"status": "accepted"}
