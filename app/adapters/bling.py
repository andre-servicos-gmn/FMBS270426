"""Sprint 2.5 — Bling ERP V3 OAuth 2.0 client.

Responsibilities:
- OAuth authorize-URL build + code-for-token exchange + refresh.
- Authenticated requests with automatic refresh on 401.
- Exponential backoff on 429 (Bling rate limits: 3 req/s, 120k req/day).
- Persistence of the singleton ``BlingCredentials`` row in Supabase.

Caller pattern::

    client = BlingClient()
    produtos = await client.listar_produtos(pagina=1, limite=100)

The client lazily loads credentials from the DB on the first
authenticated call. ``ensure_authorized()`` raises ``BlingNotAuthorizedError``
when no row exists — the caller decides whether to redirect to /oauth/authorize
or just log and skip.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import secrets
import urllib.parse
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from sqlalchemy import select

from app.config import get_settings
from app.storage.db import get_session
from app.storage.models import BlingCredentials

logger = logging.getLogger(__name__)


BASE_URL = "https://api.bling.com.br/Api/v3"
AUTH_URL = "https://www.bling.com.br/Api/v3/oauth/authorize"
TOKEN_URL = "https://api.bling.com.br/Api/v3/oauth/token"

_TOKEN_REFRESH_LEEWAY = timedelta(seconds=60)
_MAX_REQUEST_RETRIES = 4
_BACKOFF_BASE_S = 1.0


class BlingError(Exception):
    """Base class for Bling API errors."""


class BlingNotAuthorizedError(BlingError):
    """Raised when no Bling credentials row exists in the DB."""


class BlingRateLimitError(BlingError):
    """Raised after the retry budget is exhausted on 429 responses."""


def _basic_auth_header(client_id: str, client_secret: str) -> str:
    raw = f"{client_id}:{client_secret}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("ascii")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class BlingClient:
    """Async Bling V3 client with OAuth refresh + 429 backoff."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client_id = settings.bling_client_id
        self._client_secret = settings.bling_client_secret
        self._redirect_uri = settings.bling_redirect_uri

    # ── OAuth: authorize URL + code exchange + refresh ───────────────────

    def get_authorize_url(self, state: str) -> str:
        """Return the URL Andre opens to authorize the app."""
        params = {
            "response_type": "code",
            "client_id": self._client_id,
            "state": state,
            "redirect_uri": self._redirect_uri,
        }
        return f"{AUTH_URL}?{urllib.parse.urlencode(params)}"

    async def exchange_code_for_token(self, code: str) -> dict[str, Any]:
        """Trade authorization_code for access_token + refresh_token.

        Persists the result via ``_save_credentials``. Raises ``BlingError``
        on 4xx (invalid code) or transport failures.
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                TOKEN_URL,
                headers={
                    "Authorization": _basic_auth_header(
                        self._client_id, self._client_secret
                    ),
                    "Accept": "1.0",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                },
            )
        if resp.status_code >= 400:
            logger.error(
                "bling_oauth_exchange_failed status=%d body=%.200r",
                resp.status_code, resp.text,
            )
            raise BlingError(f"OAuth exchange failed: {resp.status_code} {resp.text[:200]}")
        data = resp.json()
        await self._save_credentials(data)
        logger.info(
            "bling_oauth_exchange_ok scope=%s expires_in=%s",
            data.get("scope"), data.get("expires_in"),
        )
        return data

    async def refresh_access_token(self) -> dict[str, Any]:
        """Use the stored refresh_token to renew the access_token."""
        creds = await self._load_credentials()
        if creds is None:
            raise BlingNotAuthorizedError("no credentials row to refresh")

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                TOKEN_URL,
                headers={
                    "Authorization": _basic_auth_header(
                        self._client_id, self._client_secret
                    ),
                    "Accept": "1.0",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": creds.refresh_token,
                },
            )
        if resp.status_code >= 400:
            logger.error(
                "bling_oauth_refresh_failed status=%d body=%.200r",
                resp.status_code, resp.text,
            )
            raise BlingError(f"OAuth refresh failed: {resp.status_code} {resp.text[:200]}")
        data = resp.json()
        await self._save_credentials(data)
        logger.info("bling_oauth_refresh_ok expires_in=%s", data.get("expires_in"))
        return data

    # ── Credentials persistence (singleton row) ─────────────────────────

    async def _load_credentials(self) -> BlingCredentials | None:
        async with get_session() as session:
            result = await session.execute(select(BlingCredentials).limit(1))
            return result.scalar_one_or_none()

    async def _save_credentials(self, token_response: dict[str, Any]) -> None:
        """Upsert the singleton row with the token bundle from Bling."""
        access = token_response.get("access_token") or ""
        refresh = token_response.get("refresh_token") or ""
        expires_in = int(token_response.get("expires_in") or 0)
        expires_at = _now() + timedelta(seconds=expires_in)
        scope = token_response.get("scope") or ""

        async with get_session() as session:
            existing = (await session.execute(select(BlingCredentials).limit(1))).scalar_one_or_none()
            if existing is None:
                session.add(BlingCredentials(
                    access_token=access,
                    refresh_token=refresh,
                    expires_at=expires_at,
                    scope=scope,
                ))
            else:
                existing.access_token = access
                # Some Bling refresh responses don't return a fresh refresh_token;
                # keep the existing one when missing.
                if refresh:
                    existing.refresh_token = refresh
                existing.expires_at = expires_at
                if scope:
                    existing.scope = scope
                existing.updated_at = _now()
            await session.commit()

    async def ensure_authorized(self) -> BlingCredentials:
        """Load credentials, refreshing if expired. Raise if not authorized."""
        creds = await self._load_credentials()
        if creds is None:
            raise BlingNotAuthorizedError(
                "Bling not authorized — open /bling/oauth/authorize first"
            )
        expires = creds.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if expires - _TOKEN_REFRESH_LEEWAY <= _now():
            logger.info("bling_token_expired_or_near_expiry — refreshing")
            await self.refresh_access_token()
            refreshed = await self._load_credentials()
            assert refreshed is not None
            return refreshed
        return creds

    # ── Authenticated request with auto-refresh + 429 backoff ──────────

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{BASE_URL}{path}"
        creds = await self.ensure_authorized()

        for attempt in range(_MAX_REQUEST_RETRIES):
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    headers={
                        "Authorization": f"Bearer {creds.access_token}",
                        "Accept": "application/json",
                    },
                )

            if resp.status_code == 401:
                # Token may have just expired — refresh once.
                if attempt == 0:
                    logger.info("bling_request_401 — refreshing token and retrying")
                    await self.refresh_access_token()
                    creds = await self.ensure_authorized()
                    continue
                logger.error("bling_request_401_after_refresh path=%s", path)
                raise BlingError(f"Unauthorized after refresh: {path}")

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After") or 0)
                delay = max(retry_after, _BACKOFF_BASE_S * (2 ** attempt))
                logger.warning(
                    "bling_rate_limited path=%s attempt=%d sleeping=%.1fs",
                    path, attempt + 1, delay,
                )
                await asyncio.sleep(delay)
                continue

            if 500 <= resp.status_code < 600:
                delay = _BACKOFF_BASE_S * (2 ** attempt)
                logger.warning(
                    "bling_server_error path=%s status=%d sleeping=%.1fs",
                    path, resp.status_code, delay,
                )
                await asyncio.sleep(delay)
                continue

            if resp.status_code >= 400:
                logger.error(
                    "bling_request_failed path=%s status=%d body=%.200r",
                    path, resp.status_code, resp.text,
                )
                raise BlingError(
                    f"{method} {path} → {resp.status_code} {resp.text[:200]}"
                )

            # 2xx — payload is either JSON or empty.
            if not resp.content:
                return {}
            return resp.json()

        raise BlingRateLimitError(
            f"exceeded {_MAX_REQUEST_RETRIES} retries on {method} {path}"
        )

    # ── Public API methods ─────────────────────────────────────────────

    async def listar_produtos(
        self,
        pagina: int = 1,
        limite: int = 100,
        criterio: int | None = None,
        categoria: int | None = None,
    ) -> dict[str, Any]:
        """GET /produtos — paginated product list.

        ``criterio=1`` (default in v3) lists active products only; 2 lists
        inactive, 3 lists both. ``categoria`` filters by category id.
        """
        params: dict[str, Any] = {"pagina": pagina, "limite": limite}
        if criterio is not None:
            params["criterio"] = criterio
        if categoria is not None:
            params["idCategoria"] = categoria
        return await self._request("GET", "/produtos", params=params)

    async def consultar_produto(self, produto_id: int) -> dict[str, Any]:
        """GET /produtos/{id} — full product detail (custom fields, etc)."""
        return await self._request("GET", f"/produtos/{produto_id}")

    async def consultar_estoque(
        self, produto_id: int, deposito_id: int | None = None
    ) -> dict[str, Any]:
        """GET /estoques/saldos — real-time stock balance."""
        params: dict[str, Any] = {"idsProdutos[]": produto_id}
        if deposito_id is not None:
            params["idsDepositos[]"] = deposito_id
        return await self._request("GET", "/estoques/saldos", params=params)

    async def listar_categorias(self) -> list[dict[str, Any]]:
        data = await self._request("GET", "/categorias/produtos")
        return data.get("data") or []

    async def listar_depositos(self) -> list[dict[str, Any]]:
        data = await self._request("GET", "/depositos")
        return data.get("data") or []

    async def listar_campos_customizados(self) -> list[dict[str, Any]]:
        """Sprint 2.5.2 — fetch the catalog of custom-field definitions.

        Bling V3 returns custom fields embedded in /produtos/{id} responses
        as ``[{"idCampoCustomizado": <id>, "valor": <value>}]`` — i.e. only
        the ID, not the human-readable name. To translate IDs to names we
        hit this catalog endpoint once at sync start.

        Returns ``[]`` if the endpoint is missing (some Bling builds expose
        it under different paths) so the sync can still proceed; the agent
        will just fall back to ``campo_<id>`` keys in that case.
        """
        for path in ("/produtos/campos-customizados", "/campos-customizados"):
            try:
                data = await self._request("GET", path)
                items = data.get("data")
                if isinstance(items, list):
                    return items
            except BlingError as exc:
                logger.info("bling_custom_fields_catalog_miss path=%s err=%s", path, exc)
                continue
        return []
