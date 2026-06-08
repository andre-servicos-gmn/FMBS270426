"""Sprint 2.6.7 — re-parse the 5 target attributes against the catalog
already in Supabase, WITHOUT hitting Bling.

The Sprint 2.5.x daily sync stored ``descricao_curta`` and
``descricao_complementar`` for every product. Sprint 2.6.7 improved the
parser; this script applies the improved parser over the already-synced
descriptions so we don't have to wait 14 min for the next daily sync
just to back-fill the attributes.

Safety:
- DEFAULT is ``--dry-run`` — no writes. Prints a diff summary.
- ``--apply`` is the explicit opt-in for writes.
- Each UPDATE merges into ``atributos_parseados``: preserves every
  ``campo_XXXX`` (Bling custom-field key) AND any non-target key already
  there. ONLY the 5 target slugs are updated/added.
- Idempotent: running ``--apply`` twice produces the same result the
  second time around (no diff).

Usage::

    .venv/Scripts/python scripts/reparse_attributes.py            # dry-run (default)
    .venv/Scripts/python scripts/reparse_attributes.py --apply    # writes
    .venv/Scripts/python scripts/reparse_attributes.py --limit 50 # try on 50 first
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Any

from sqlalchemy import select, update

from app.storage.db import get_session
from app.storage.models import BlingProduct
from app.sync.bling_sync import parse_attributes_from_description

logger = logging.getLogger("reparse_attributes")

_TARGET_SLUGS = ("peso", "equilibrio", "composicao", "espessura", "comprimento")


def _combined_description(p: BlingProduct) -> str:
    """Concatenate descricao_curta + descricao_complementar, in that order."""
    parts: list[str] = []
    if p.descricao_curta:
        parts.append(p.descricao_curta)
    if p.descricao_complementar:
        parts.append(p.descricao_complementar)
    return "\n".join(parts)


def _merge_attrs(existing: dict[str, Any] | None, fresh: dict[str, str]) -> dict[str, Any]:
    """Return a new dict that preserves every NON-target key in ``existing``
    and sets the 5 target slugs to whatever ``fresh`` provides. If
    ``fresh`` is missing a target, the existing value (if any) is kept."""
    merged: dict[str, Any] = dict(existing or {})
    for slug in _TARGET_SLUGS:
        if slug in fresh:
            merged[slug] = fresh[slug]
        # else: leave existing[slug] untouched (don't downgrade good data
        # by overwriting with empty).
    return merged


def _diff_target_slugs(
    before: dict[str, Any] | None, after: dict[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """Return {slug: (before, after)} for the 5 target slugs that CHANGED."""
    before = before or {}
    out: dict[str, tuple[Any, Any]] = {}
    for slug in _TARGET_SLUGS:
        b = before.get(slug)
        a = after.get(slug)
        if b != a:
            out[slug] = (b, a)
    return out


async def _load_all() -> list[BlingProduct]:
    async with get_session() as session:
        result = await session.execute(select(BlingProduct))
        return list(result.scalars().all())


async def _apply_update(product_id: int, new_attrs: dict[str, Any]) -> None:
    async with get_session() as session:
        await session.execute(
            update(BlingProduct)
            .where(BlingProduct.id == product_id)
            .values(atributos_parseados=new_attrs)
        )
        await session.commit()


async def main(args: argparse.Namespace) -> int:
    rows = await _load_all()
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("Nenhum produto em bling_products. Rode o sync inicial primeiro.")
        return 1

    gained: dict[str, int] = {s: 0 for s in _TARGET_SLUGS}
    changed: dict[str, int] = {s: 0 for s in _TARGET_SLUGS}
    cleared: dict[str, int] = {s: 0 for s in _TARGET_SLUGS}
    unchanged = 0
    touched = 0
    samples: list[str] = []

    for p in rows:
        description = _combined_description(p)
        fresh = parse_attributes_from_description(description)
        before = p.atributos_parseados or {}
        merged = _merge_attrs(before, fresh)
        diff = _diff_target_slugs(before, merged)

        if not diff:
            unchanged += 1
            continue

        touched += 1
        for slug, (b, a) in diff.items():
            if b in (None, "") and a not in (None, ""):
                gained[slug] += 1
            elif a in (None, "") and b not in (None, ""):
                cleared[slug] += 1
            else:
                changed[slug] += 1

        if len(samples) < 10:
            short_name = (p.nome or f"id={p.id}")[:60]
            slug_changes = ", ".join(
                f"{s}: {b!r} → {a!r}"
                for s, (b, a) in list(diff.items())[:3]
            )
            samples.append(f"  • {short_name}\n      {slug_changes}")

        if args.apply:
            await _apply_update(p.id, merged)

    # ── Summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"  REPARSE SUMMARY ({'APPLIED' if args.apply else 'DRY-RUN'})")
    print("=" * 72)
    print(f"  produtos processados:  {len(rows)}")
    print(f"  inalterados:           {unchanged}")
    print(f"  tocados:               {touched}")
    print()
    print(f"  {'atributo':<14} {'+ novos':>10} {'alterados':>12} {'limpos':>10}")
    print("  " + "-" * 50)
    for slug in _TARGET_SLUGS:
        print(
            f"  {slug:<14} "
            f"{gained[slug]:>10} "
            f"{changed[slug]:>12} "
            f"{cleared[slug]:>10}"
        )
    print()

    if samples:
        print("Amostras (até 10):")
        for line in samples:
            print(line)

    if not args.apply:
        print()
        print("(dry-run — nada foi gravado. Rode com --apply pra persistir.)")
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--apply",
        action="store_true",
        help="grava as mudanças no Supabase (default é dry-run)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="processa só os primeiros N produtos (útil pra teste)",
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(asyncio.run(main(_parse_args())))
