"""Sprint 2.5 — Bling → local catalog sync.

Three entry points:
- ``full_sync()`` — paginate /produtos, fetch detail per id, UPSERT.
- ``sync_single_product(id)`` — fetch one + UPSERT. Used by the webhook.
- ``delete_product(id)`` — flip situacao='E'. Used by the webhook.

The sync also runs a best-effort regex parse over the HTML description to
extract attributes like Perfil / Composição / Detalhamento and stores them
in ``atributos_parseados``. Custom fields take precedence whenever both
sources have the same key.

This module is LLM-free and idempotent — calling ``full_sync`` twice in a
row should produce zero "inserted" and N "updated".
"""
from __future__ import annotations

import json
import logging
import re
import traceback
import unicodedata
from datetime import datetime, timezone
from html import unescape
from typing import Any

from app.adapters.bling import BlingClient
from app.config import get_settings
from app.sync.bling_repo import (
    close_sync_log,
    mark_product_inactive,
    open_sync_log,
    upsert_product,
)

logger = logging.getLogger(__name__)


# Sprint 2.5.1 — payload size cap for DEBUG logs. The Bling /produtos/{id}
# response is usually 1–4 KB but can hit 20+ KB on products with rich
# descriptions / many custom fields; we cap it so the log file doesn't
# explode if someone forgets to turn DEBUG off.
_DEBUG_PAYLOAD_TRUNC = 2000


def _truncate(value: str, limit: int = _DEBUG_PAYLOAD_TRUNC) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"... <truncated, full_len={len(value)}>"


def _safe_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return repr(obj)


# ── Parsers ─────────────────────────────────────────────────────────────

_HTML_TAG_RE = re.compile(r"<[^>]+>")

# Sprint 2.5.2 — Bling descriptions use varied dash characters: ASCII "-",
# en-dash "–", em-dash "—", bullet "•", asterisk "*". The label captures
# accented letters + spaces; the value captures everything up to a
# terminator (newline, semicolon, or another bullet/dash at start of line).
_LIST_ATTR_RE = re.compile(
    r"(?:^|[\n\r])[\s]*[-–—•*][\s]*"
    r"([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ ]{2,40}?)"
    r"\s*:\s*"
    r"([^\n\r;]+?)"
    r"(?=[;\n\r]|$)",
    re.MULTILINE,
)


def _strip_html(text: str | None) -> str:
    if not text:
        return ""
    decoded = unescape(text)
    # Replace tags with newlines so block boundaries survive.
    no_tags = _HTML_TAG_RE.sub("\n", decoded)
    # Collapse runs of whitespace within lines; preserve newlines.
    no_tags = re.sub(r"[ \t]+", " ", no_tags)
    no_tags = re.sub(r"\n[ \t]+", "\n", no_tags)
    no_tags = re.sub(r"\n{2,}", "\n", no_tags)
    return no_tags.strip()


def _slug(label: str) -> str:
    norm = unicodedata.normalize("NFD", label.strip().lower())
    norm = "".join(c for c in norm if unicodedata.category(c) != "Mn")
    norm = re.sub(r"[^a-z0-9]+", "_", norm).strip("_")
    return norm


def parse_attributes_from_description(html: str | None) -> dict[str, str]:
    """Best-effort: find ``- Label: value`` lines inside the description.

    Sprint 2.5.2 — accepts varied dash characters (ASCII, en-, em-, bullet,
    asterisk) and is tolerant of inline HTML. Returns a dict keyed by slug
    (e.g. ``"perfil"``, ``"composicao"``); empty dict if nothing matched.
    """
    if not html:
        return {}
    plain = _strip_html(html)
    if not plain:
        return {}
    # Prepend a newline so the FIRST line of the text can match the
    # ``(?:^|[\n\r])`` anchor too.
    plain = "\n" + plain
    parsed: dict[str, str] = {}
    for label, value in _LIST_ATTR_RE.findall(plain):
        slug = _slug(label)
        clean_value = value.strip(" .;\t ")
        if slug and clean_value and slug not in parsed:
            parsed[slug] = clean_value
    return parsed


def _get_description_html(detail: dict[str, Any]) -> str:
    """Sprint 2.5.2 — try multiple Bling field names for the description."""
    if not isinstance(detail, dict):
        return ""
    for key in (
        "descricaoComplementar",
        "descricaoLonga",
        "descricao",
        "descricaoCurta",
    ):
        val = detail.get(key)
        if val:
            return str(val)
    return ""


# ── Custom-field flattening + raquete detection ─────────────────────────

# The Bling product detail endpoint returns custom fields under different
# keys depending on the build (the docs call them ``camposCustomizados``).
# We accept any of these variants gracefully.
_CUSTOM_FIELD_KEYS = ("camposCustomizados", "campos_customizados", "customFields")


def _flatten_custom_fields(
    detail: dict[str, Any],
    *,
    field_map: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Turn the Bling list of custom fields into a flat ``{name: value}``.

    Sprint 2.5.2 — Bling V3 actually returns custom fields as
    ``[{"idCampoCustomizado": <id>, "valor": <value>}]`` (NO name in the
    product payload). The caller passes ``field_map`` (id → name) so we
    can translate. When the map doesn't have the ID, we fall back to a
    synthetic ``"campo_<id>"`` key so the data isn't lost — the agent can
    still see the value, just without a human label.
    """
    if not isinstance(detail, dict):
        return {}

    raw: Any = None
    for key in _CUSTOM_FIELD_KEYS:
        candidate = detail.get(key)
        if candidate:  # truthy → non-empty list / non-empty dict
            raw = candidate
            break
    if raw is None:
        return {}

    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return {}

    flat: dict[str, Any] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        # First try the obvious name fields (some builds DO include them).
        name = (
            entry.get("nome")
            or entry.get("name")
            or entry.get("descricao")
            or ""
        )
        value = entry.get("valor")
        if value is None:
            value = entry.get("value")

        # Sprint 2.5.2 — when the name isn't inlined, look it up by ID.
        field_id_raw = (
            entry.get("idCampoCustomizado")
            or entry.get("id")
            or entry.get("campoId")
        )
        field_id: int | None = None
        if field_id_raw is not None:
            try:
                field_id = int(field_id_raw)
            except (TypeError, ValueError):
                field_id = None

        if not name and field_id is not None and field_map:
            name = field_map.get(field_id, "")

        if not name and field_id is not None:
            name = f"campo_{field_id}"

        if name:
            flat[str(name)] = value
    return flat


_RAQUETE_PRAIA_FIELD_NAMES = (
    "es raquete de praia",
    "é raquete de praia",
    "eh raquete de praia",
    "raquete de praia",
    "raquete praia",
    "beach racket",
    "beach tennis racket",
    "is_beach_racket",
)

# Sprint 2.5.2 — Bling toggles serialize as several truthy strings depending
# on locale/build. Accept all known variants.
_TRUTHY = {
    "true", "sim", "yes", "1", "on", "verdadeiro",
    "ativado", "ativo", "habilitado", "ligado",
}


def is_raquete_de_praia(
    detail: dict[str, Any],
    custom_fields: dict[str, Any],
    *,
    categoria_nome_resolved: str | None = None,
) -> bool:
    """Return True when the product is a beach-tennis racket.

    Priority order (Sprint 2.5.2):
    1. Custom field "Es raquete de praia" (or equivalent) → truthy toggle.
       Accepted truthy values: ``True``, ``"true"``, ``"sim"``, ``"yes"``,
       ``"1"``, ``"on"``, ``"verdadeiro"``, ``"ativado"``, ``"ativo"``,
       ``"habilitado"``, ``"ligado"``.
    2. Category match — accepts "Raquetes de Praia" exactly, or any name
       containing "raquete" AND ("praia" OR "beach"). Uses the resolved
       categoria_nome when provided (so ID→name map results count).

    Never raises — malformed input falls through to False.
    """
    try:
        # 1) Custom field
        if isinstance(custom_fields, dict):
            for k, v in custom_fields.items():
                slug = _slug(str(k))
                if any(slug == _slug(name) for name in _RAQUETE_PRAIA_FIELD_NAMES):
                    if isinstance(v, bool):
                        if v:
                            logger.info("is_raquete_praia matched=custom_field key=%s", k)
                        return v
                    if v is None:
                        return False
                    truthy = str(v).strip().lower() in _TRUTHY
                    if truthy:
                        logger.info("is_raquete_praia matched=custom_field key=%s value=%r", k, v)
                    return truthy
        # 2) Category fallback
        categoria_nome = categoria_nome_resolved or _extract_categoria_nome(detail) or ""
        norm = _slug(categoria_nome)
        # Exact match against the canonical Bling category name.
        if norm == _slug("Raquetes de Praia"):
            logger.info("is_raquete_praia matched=category_exact name=%s", categoria_nome)
            return True
        # Fuzzy match for unusual category names.
        if "raquete" in norm and ("praia" in norm or "beach" in norm):
            logger.info("is_raquete_praia matched=category_fuzzy name=%s", categoria_nome)
            return True
        return False
    except Exception as exc:
        logger.warning("is_raquete_de_praia_failed exc=%s", exc)
        return False


def _extract_categoria_nome(detail: dict[str, Any]) -> str | None:
    """Sprint 2.5.1 — accept ``categoria`` as dict / list / string / None."""
    if not isinstance(detail, dict):
        return None
    cat = detail.get("categoria")
    if cat is None:
        return None
    if isinstance(cat, str):
        return cat.strip() or None
    if isinstance(cat, list):
        first = next((c for c in cat if isinstance(c, dict)), None)
        if not first:
            return None
        return (first.get("descricao") or first.get("nome") or None)
    if isinstance(cat, dict):
        return cat.get("descricao") or cat.get("nome") or None
    return None


def _extract_categoria_id(detail: dict[str, Any]) -> int | None:
    if not isinstance(detail, dict):
        return None
    cat = detail.get("categoria")
    if isinstance(cat, dict):
        raw = cat.get("id")
    elif isinstance(cat, list):
        first = next((c for c in cat if isinstance(c, dict)), None)
        raw = first.get("id") if first else None
    else:
        raw = None
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _extract_marca_nome(detail: dict[str, Any]) -> str | None:
    """Accept ``marca`` as dict, string, or None."""
    if not isinstance(detail, dict):
        return None
    marca = detail.get("marca")
    if marca is None:
        return None
    if isinstance(marca, str):
        return marca.strip() or None
    if isinstance(marca, dict):
        return marca.get("nome") or marca.get("descricao") or None
    return None


def _extract_first_image_link(detail: dict[str, Any]) -> str | None:
    """Walk midia.imagens.externas defensively — never index a blind ``[0]``.

    Real Bling responses frequently have ``externas: []`` for products that
    don't have external images cadastrados, which used to raise IndexError
    in the original implementation.
    """
    if not isinstance(detail, dict):
        return None
    midia = detail.get("midia")
    if not isinstance(midia, dict):
        return None
    imagens = midia.get("imagens")
    if not isinstance(imagens, dict):
        return None
    externas = imagens.get("externas")
    if not isinstance(externas, list) or not externas:
        return None
    first = next((e for e in externas if isinstance(e, dict)), None)
    if not first:
        return None
    return first.get("link") or first.get("url") or None


# ── Field extractor — Bling detail → bling_products row dict ────────────

def _safe_num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def detail_to_row(
    detail: dict[str, Any],
    *,
    category_map: dict[int, str] | None = None,
    field_map: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Map a Bling /produtos/{id} response into our ``bling_products`` row.

    The response shape is ``{"data": {<product>}}`` — callers are expected
    to pass the inner dict.

    Sprint 2.5.2 — accepts optional ``category_map`` (id → nome) and
    ``field_map`` (id → nome) so we can resolve the ID-only stubs that
    Bling V3 actually returns. Both maps default to empty when omitted
    (preserving Sprint 2.5.1 backward-compat for tests).
    """
    if not isinstance(detail, dict):
        raise TypeError(f"detail_to_row expects dict, got {type(detail).__name__}")

    try:
        produto_id = int(detail["id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"product detail missing numeric id: {exc}") from exc

    custom = _flatten_custom_fields(detail, field_map=field_map)

    description_html = _get_description_html(detail)
    parsed = parse_attributes_from_description(description_html)

    # Custom fields override parsed when keys collide.
    for k, v in custom.items():
        slug = _slug(str(k))
        parsed.setdefault(slug, "" if v is None else str(v))

    dimensoes = detail.get("dimensoes") if isinstance(detail.get("dimensoes"), dict) else {}

    # Categoria: try inline name first; fall back to ID→name map.
    categoria_id = _extract_categoria_id(detail)
    categoria_nome = _extract_categoria_nome(detail)
    if not categoria_nome and categoria_id is not None and category_map:
        categoria_nome = category_map.get(categoria_id)

    # Marca: same pattern (inline string/dict OR custom field).
    marca = _extract_marca_nome(detail)
    if not marca:
        for key in ("Marca", "MARCA", "marca"):
            if key in custom and custom[key]:
                marca = str(custom[key])
                break

    # Modelo: Bling V3 typically doesn't have a top-level modelo field; it
    # lives as a custom field instead. Try inline first, fall back.
    modelo = (detail.get("modelo") or "").strip() or None
    if not modelo:
        for key in ("Modelo", "MODELO", "modelo"):
            if key in custom and custom[key]:
                modelo = str(custom[key])
                break

    return {
        "id": produto_id,
        "nome": (detail.get("nome") or "").strip() or f"Produto {produto_id}",
        "codigo": (detail.get("codigo") or None) or None,
        "preco": _safe_num(detail.get("preco")),
        "descricao_curta": _strip_html(detail.get("descricaoCurta")) or None,
        "descricao_complementar": _strip_html(description_html) or None,
        "marca": marca,
        "modelo": modelo,
        "categoria_id": categoria_id,
        "categoria_nome": categoria_nome,
        "peso_liquido": _safe_num(detail.get("pesoLiquido")),
        "peso_bruto": _safe_num(detail.get("pesoBruto")),
        "largura": _safe_num(dimensoes.get("largura") if dimensoes else None),
        "altura": _safe_num(dimensoes.get("altura") if dimensoes else None),
        "profundidade": _safe_num(dimensoes.get("profundidade") if dimensoes else None),
        "gtin": (detail.get("gtin") or None) or None,
        "situacao": detail.get("situacao") or "A",
        "is_raquete_praia": is_raquete_de_praia(
            detail, custom, categoria_nome_resolved=categoria_nome,
        ),
        "campos_customizados": custom,
        "atributos_parseados": parsed,
        "imagem_url": _extract_first_image_link(detail),
        "last_synced_at": datetime.now(timezone.utc),
    }


# ── Category filter ─────────────────────────────────────────────────────

def _wanted_categories() -> set[str]:
    raw = (get_settings().bling_sync_categories or "").strip()
    if not raw:
        return set()
    return {c.strip() for c in raw.split(",") if c.strip()}


# ── Sync API ────────────────────────────────────────────────────────────

class BlingSync:
    def __init__(self, client: BlingClient | None = None) -> None:
        self._client = client or BlingClient()
        # Sprint 2.5.2 — Bling V3 returns custom fields + categorias as
        # ID-only stubs. We pre-load both ID→name maps once per sync so
        # ``detail_to_row`` can translate.
        self._category_map: dict[int, str] = {}
        self._field_map: dict[int, str] = {}
        self._maps_loaded = False

    async def _ensure_maps(self) -> None:
        """Lazy-load the categoria + custom-field name maps. Idempotent."""
        if self._maps_loaded:
            return
        try:
            cats = await self._client.listar_categorias()
            for c in cats or []:
                if isinstance(c, dict) and c.get("id") is not None:
                    name = c.get("descricao") or c.get("nome") or ""
                    if name:
                        try:
                            self._category_map[int(c["id"])] = str(name)
                        except (TypeError, ValueError):
                            continue
            logger.info("bling_category_map_loaded n=%d", len(self._category_map))
        except Exception as exc:
            logger.warning("bling_category_map_load_failed: %s", exc)

        try:
            fields = await self._client.listar_campos_customizados()
            for f in fields or []:
                if not isinstance(f, dict):
                    continue
                fid = f.get("id") or f.get("idCampoCustomizado")
                fname = f.get("nome") or f.get("descricao") or f.get("name")
                if fid is None or not fname:
                    continue
                try:
                    self._field_map[int(fid)] = str(fname)
                except (TypeError, ValueError):
                    continue
            logger.info("bling_field_map_loaded n=%d", len(self._field_map))
        except Exception as exc:
            logger.warning("bling_field_map_load_failed: %s", exc)

        self._maps_loaded = True

    async def sync_single_product(self, produto_id: int) -> str:
        """Fetch one + UPSERT. Returns 'inserted' | 'updated' | 'skipped'.

        Sprint 2.5.1 — logs the raw Bling payload at DEBUG so investigations
        can re-trace shape variations without re-hitting the API. The
        product id is always included in failure logs so callers can
        cross-reference quickly.
        """
        detail_resp = await self._client.consultar_produto(produto_id)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "BLING_RAW_PAYLOAD id=%s body=%s",
                produto_id, _truncate(_safe_dumps(detail_resp)),
            )

        detail = (detail_resp or {}).get("data") if isinstance(detail_resp, dict) else None
        if not detail or not isinstance(detail, dict):
            logger.warning(
                "bling_sync_single empty_detail id=%s response_keys=%s",
                produto_id,
                sorted(detail_resp.keys()) if isinstance(detail_resp, dict) else type(detail_resp).__name__,
            )
            return "skipped"

        await self._ensure_maps()
        try:
            row = detail_to_row(
                detail,
                category_map=self._category_map,
                field_map=self._field_map,
            )
        except Exception as exc:
            logger.error(
                "bling_sync_parse_failed id=%s exc=%s payload=%s\n%s",
                produto_id, exc, _truncate(_safe_dumps(detail_resp)),
                traceback.format_exc(),
            )
            raise

        return await upsert_product(row)

    async def delete_product(self, produto_id: int) -> bool:
        return await mark_product_inactive(produto_id)

    async def full_sync(self, only_active: bool = True) -> dict[str, int]:
        """Paginate /produtos and UPSERT each one (filtered by category)."""
        wanted = _wanted_categories()
        criterio = 1 if only_active else 3

        log_id = await open_sync_log("full", {"wanted_categories": sorted(wanted)})
        stats = {"total_processed": 0, "inserted": 0, "updated": 0, "skipped": 0, "errors": 0}

        # Sprint 2.5.2 — front-load the ID→name maps so we don't pay the cost
        # per product. Errors here are non-fatal; sync proceeds with empty maps.
        await self._ensure_maps()

        try:
            pagina = 1
            while True:
                listing = await self._client.listar_produtos(
                    pagina=pagina, limite=100, criterio=criterio
                )
                items = listing.get("data") or []
                if not items:
                    break

                for item in items:
                    stats["total_processed"] += 1
                    # The list endpoint returns a shallow product; we always
                    # fetch the detail to get custom fields + full description.
                    produto_id = int(item.get("id"))
                    categoria_nome = ((item.get("categoria") or {}).get("descricao") or "")
                    if wanted and categoria_nome and categoria_nome not in wanted:
                        stats["skipped"] += 1
                        continue
                    try:
                        outcome = await self.sync_single_product(produto_id)
                        if outcome == "inserted":
                            stats["inserted"] += 1
                        elif outcome == "updated":
                            stats["updated"] += 1
                        else:
                            stats["skipped"] += 1
                    except Exception as exc:
                        stats["errors"] += 1
                        # Sprint 2.5.1 — keep the WARNING line for the
                        # short-form log everyone reads, but also dump the
                        # full traceback at DEBUG so --debug-first surfaces
                        # the failing line directly.
                        logger.warning(
                            "bling_sync_single_failed id=%s: %s", produto_id, exc
                        )
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                "bling_sync_single_traceback id=%s\n%s",
                                produto_id, traceback.format_exc(),
                            )

                pagina += 1
        except Exception as exc:
            logger.exception("bling_full_sync_aborted: %s", exc)
            await close_sync_log(log_id, error_message=str(exc)[:500], **stats)
            raise

        await close_sync_log(log_id, **stats)
        logger.info("bling_full_sync_done %s", stats)
        return stats
