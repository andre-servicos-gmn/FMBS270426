"""Sprint 2.7.4 — temporary debug script. Dumps the raw first page of
``/campos-customizados/modulos/{idModulo}`` so we can confirm the response
shape (key names, situacao value type, etc.).

DELETE after the fix lands. NOT meant to ship to production.

Usage::

    docker exec -it <container> python scripts/debug_custom_fields_response.py
"""
from __future__ import annotations

import asyncio
import json
import sys

from app.adapters.bling import BlingClient, BlingNotAuthorizedError


async def main() -> int:
    client = BlingClient()

    # 1. Discover the Produtos module id.
    print("=" * 72)
    print("STEP 1 — discovering Produtos module id")
    print("=" * 72)
    try:
        module_id = await client._find_produtos_module_id()
    except BlingNotAuthorizedError:
        print("ERROR: not authorized. Re-run /bling/oauth/authorize first.")
        return 2

    if module_id is None:
        print("FAILED to discover 'Produtos' module — see logs above.")
        return 3

    print(f"✓ Module 'Produtos' id = {module_id}\n")

    # 2. Fetch page 1 raw, NO filtering.
    print("=" * 72)
    print(f"STEP 2 — raw GET /campos-customizados/modulos/{module_id}?pagina=1&limite=100")
    print("=" * 72)
    try:
        raw = await client._request(
            "GET",
            f"/campos-customizados/modulos/{module_id}",
            params={"pagina": 1, "limite": 100},
        )
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 4

    items = (raw or {}).get("data") or []
    print(f"\nTop-level keys: {sorted((raw or {}).keys())}")
    print(f"data is list?:  {isinstance(items, list)}")
    print(f"data length:    {len(items)}\n")

    # 3. Print first 3 items in full, then a value-distribution summary.
    print("=" * 72)
    print("STEP 3 — first 3 items (full raw JSON)")
    print("=" * 72)
    for i, item in enumerate(items[:3]):
        print(f"\n--- item[{i}] ---")
        print(json.dumps(item, ensure_ascii=False, indent=2, default=str))

    print()
    print("=" * 72)
    print("STEP 4 — shape summary across all 100 items")
    print("=" * 72)

    # Aggregate key presence + situacao value-type distribution.
    all_keys: dict[str, int] = {}
    situacao_values: dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for k in item.keys():
            all_keys[k] = all_keys.get(k, 0) + 1
        v = item.get("situacao")
        repr_v = f"{type(v).__name__}={v!r}"
        situacao_values[repr_v] = situacao_values.get(repr_v, 0) + 1

    print(f"\nKey presence across {len(items)} items (key → count):")
    for k, c in sorted(all_keys.items(), key=lambda x: -x[1]):
        print(f"  {k:30s} {c}")

    print(f"\nsituacao value distribution:")
    for v, c in sorted(situacao_values.items(), key=lambda x: -x[1]):
        print(f"  {v:30s} {c}")

    # Distinct id+nome examples — confirms naming is intact.
    print(f"\nFirst 10 id → nome:")
    for item in items[:10]:
        if isinstance(item, dict):
            print(f"  {item.get('id'):>12} → {item.get('nome')!r}  (situacao={item.get('situacao')!r})")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
