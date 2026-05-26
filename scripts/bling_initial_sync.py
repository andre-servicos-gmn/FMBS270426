"""Sprint 2.5 + 2.5.1 — run the first full sync of the Bling catalog.

Usage::

    .venv/Scripts/python scripts/bling_initial_sync.py
    .venv/Scripts/python scripts/bling_initial_sync.py --debug-first

``--debug-first`` is a low-cost probe that:
- Lists ONE page of products (1 API call).
- Picks the first 3 ids, calls /produtos/{id} per id (up to 3 more calls).
- Stops at the first successful sync OR after the 3rd attempt — whichever
  comes first. Total: 1–4 API calls.
- Prints the raw JSON Bling returned + the full traceback when a parse
  fails. Goal is to capture the exact shape variation without burning the
  daily Bling cota.

Without ``--debug-first``, behaviour is unchanged from Sprint 2.5: full
paginated sync.

Prerequisites:
- ``.env`` populated with BLING_CLIENT_ID + BLING_CLIENT_SECRET.
- Andre has visited ``/bling/oauth/authorize`` (credentials row exists).
- Migrations 0006 + 0007 applied.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import traceback

from app.adapters.bling import BlingClient, BlingNotAuthorizedError
from app.config import configure_logging, get_settings
from app.sync.bling_sync import BlingSync, detail_to_row


def _print_section(title: str) -> None:
    print()
    print("─" * 72)
    print(f"  {title}")
    print("─" * 72)


def _safe_pretty(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return repr(obj)


def _level_keys(obj: object, max_levels: int = 2, prefix: str = "data") -> list[str]:
    """Return printable "level N (path.*): k1, k2, ..." lines."""
    lines: list[str] = []
    if not isinstance(obj, dict):
        return lines
    lines.append(f"  keys @ {prefix}: {', '.join(sorted(obj.keys()))}")
    if max_levels <= 1:
        return lines
    for k in sorted(obj.keys()):
        v = obj[k]
        if isinstance(v, dict):
            lines.extend(_level_keys(v, max_levels - 1, f"{prefix}.{k}"))
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            lines.extend(_level_keys(v[0], max_levels - 1, f"{prefix}.{k}[0]"))
    return lines


async def _debug_first_probe(client: BlingClient, sync: BlingSync) -> int:
    """Sprint 2.5.2 — process ALL 3 sample products and dump full JSON to
    individual files. Terminal only shows the keys layout, so even rich
    payloads stay readable. Files are written to the current directory
    (next to the script invocation) for easy attachment.
    """
    print("DEBUG MODE: 1 listing call + up to 3 detail calls (max).")
    print("Full raw JSON is saved to ./debug_raw_payload_<id>.json")
    try:
        listing = await client.listar_produtos(pagina=1, limite=100, criterio=1)
    except BlingNotAuthorizedError:
        print("❌ No Bling credentials. Open /bling/oauth/authorize first.")
        return 2
    except Exception as exc:
        print(f"❌ listar_produtos failed: {exc}")
        return 3

    items = (listing or {}).get("data") or []
    if not items:
        print("⚠️  /produtos returned an empty list.")
        print(f"Raw listing top-level keys: {sorted((listing or {}).keys())}")
        return 0

    # Pre-load the categoria + custom-field maps so the parse uses
    # the resolved names (matches the production sync_single_product path).
    await sync._ensure_maps()
    print(
        f"  ID→name maps loaded: "
        f"categorias={len(sync._category_map)} "
        f"campos_customizados={len(sync._field_map)}"
    )

    sample_ids = [int(it.get("id")) for it in items[:3]
                  if isinstance(it, dict) and it.get("id")]
    print(f"Sample IDs probed: {sample_ids}")

    successes = 0
    failures: list[tuple[int, str, str]] = []

    from app.sync.bling_repo import upsert_product
    from app.sync.bling_sync import detail_to_row

    for produto_id in sample_ids:
        _print_section(f"Probing produto_id={produto_id}")
        try:
            raw = await client.consultar_produto(produto_id)
        except Exception as exc:
            print(f"❌ consultar_produto({produto_id}) failed: {exc}")
            failures.append((produto_id, "fetch_failed", traceback.format_exc()))
            continue

        # Save the FULL raw JSON to disk (untruncated).
        out_path = f"debug_raw_payload_{produto_id}.json"
        try:
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(raw, fh, ensure_ascii=False, indent=2, default=str)
            print(f"  ✓ saved full payload to {out_path}")
        except Exception as exc:
            print(f"  ⚠️  failed to write {out_path}: {exc}")

        # Print the KEYS layout only (no values).
        print("  --- JSON key layout ---")
        for line in _level_keys(raw, max_levels=2):
            print(line)

        detail = (raw or {}).get("data") if isinstance(raw, dict) else None
        if not detail:
            print(f"  ⚠️  detail is empty/missing for id={produto_id}")
            failures.append((produto_id, "empty_detail", ""))
            continue

        try:
            row = detail_to_row(
                detail,
                category_map=sync._category_map,
                field_map=sync._field_map,
            )
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"  ❌ detail_to_row failed for id={produto_id}: {exc}")
            print(tb)
            failures.append((produto_id, str(exc), tb))
            continue

        try:
            outcome = await upsert_product(row)
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"  ❌ upsert failed for id={produto_id}: {exc}")
            print(tb)
            failures.append((produto_id, f"upsert_failed: {exc}", tb))
            continue

        print(f"  ✅ id={produto_id} parsed + persisted ({outcome}).")
        print(f"     nome:                 {row.get('nome')!r}")
        print(f"     codigo:               {row.get('codigo')!r}")
        print(f"     categoria_nome:       {row.get('categoria_nome')!r}")
        print(f"     categoria_id:         {row.get('categoria_id')!r}")
        print(f"     marca:                {row.get('marca')!r}")
        print(f"     modelo:               {row.get('modelo')!r}")
        print(f"     is_raquete_praia:    {row.get('is_raquete_praia')}")
        print(f"     custom_fields_count:  {len(row.get('campos_customizados') or {})}")
        custom_keys = sorted((row.get('campos_customizados') or {}).keys())
        print(f"     custom_field_names:   {custom_keys[:10]}{' ...' if len(custom_keys) > 10 else ''}")
        print(f"     atributos_parseados:  {sorted((row.get('atributos_parseados') or {}).keys())}")
        successes += 1
        # Sprint 2.5.2 — do NOT stop early. We want to see all 3.

    _print_section("PROBE SUMMARY")
    print(f"  successes: {successes} / {len(sample_ids)}")
    print(f"  failures:  {len(failures)}")
    if failures:
        for pid, msg, _tb in failures:
            print(f"    - id={pid}: {msg[:200]}")
    return 0 if successes else 1


async def main() -> int:
    parser = argparse.ArgumentParser(description="Bling initial sync")
    parser.add_argument(
        "--debug-first",
        action="store_true",
        help="Probe the first 3 products with DEBUG logs + raw JSON dump.",
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings)
    logger = logging.getLogger("bling_initial_sync")

    if not settings.bling_client_id:
        logger.error("BLING_CLIENT_ID is empty — fill .env first.")
        return 1

    # Sprint 2.5.1 — only this script ever flips DEBUG on (and only when
    # --debug-first is requested). uvicorn / production stays at INFO.
    if args.debug_first:
        logging.getLogger("app.sync.bling_sync").setLevel(logging.DEBUG)
        logging.getLogger("app.adapters.bling").setLevel(logging.DEBUG)

    started = time.time()
    sync = BlingSync()

    if args.debug_first:
        try:
            return await _debug_first_probe(BlingClient(), sync)
        except Exception as exc:
            logger.exception("debug_first_probe_failed: %s", exc)
            return 3

    try:
        stats = await sync.full_sync(only_active=True)
    except BlingNotAuthorizedError:
        logger.error(
            "No Bling credentials in the DB. Open /bling/oauth/authorize first."
        )
        return 2
    except Exception as exc:
        logger.exception("Sync failed: %s", exc)
        return 3

    elapsed = time.time() - started
    print()
    _print_section("FULL SYNC RESULT")
    print(f"  total_processed: {stats['total_processed']}")
    print(f"  inserted:        {stats['inserted']}")
    print(f"  updated:         {stats['updated']}")
    print(f"  skipped:         {stats['skipped']}")
    print(f"  errors:          {stats['errors']}")
    print(f"  elapsed:         {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
