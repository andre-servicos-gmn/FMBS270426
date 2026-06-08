"""Sprint 2.6.7 — Fase 1 diagnostic. Read-only.

Scans every row of ``bling_products`` and reports, per target attribute,
whether it's already in ``atributos_parseados``, recoverable from the
description, or genuinely missing in the source data.

Usage:
    .venv/Scripts/python scripts/diagnose_attributes.py

Output: a coverage table for the 5 target attributes + 10 raw-text
samples around the labels in the "recoverable" bucket (so we can read
the real formats and tune the parser without guessing). No writes.
"""
from __future__ import annotations

import asyncio
import re
import sys
import unicodedata
from typing import Any

from sqlalchemy import select

from app.storage.db import get_session
from app.storage.models import BlingProduct


# Target attributes + label markers we search for in the raw text. The
# markers are the SHORTEST distinctive prefix of each label — picked so a
# substring match on lowercased + accent-stripped text catches every
# inflection ("Composição", "composicao", "COMPOSIÇÃO" → "composi").
_TARGETS: dict[str, tuple[str, ...]] = {
    "peso": ("peso",),
    "equilibrio": ("equilibrio", "equilíbrio", "balance", "balanco", "balanço"),
    "composicao": ("composi", "material"),
    "espessura": ("espessura",),
    "comprimento": ("comprimento",),
}


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text or "")
        if unicodedata.category(c) != "Mn"
    )


def _norm(text: str) -> str:
    return _strip_accents((text or "")).lower()


def _description_text(p: BlingProduct) -> str:
    """Concatenate descricao_curta + descricao_complementar (HTML stripped)."""
    parts: list[str] = []
    for raw in (p.descricao_curta, p.descricao_complementar):
        if raw:
            parts.append(re.sub(r"<[^>]+>", " ", raw))
    return " \n ".join(parts)


def _has_label_in_text(text_norm: str, markers: tuple[str, ...]) -> bool:
    """Does the description contain at least one of the label markers
    immediately followed (within 5 chars) by a colon? We require the
    colon to avoid false positives on "peso" used as marketing copy."""
    for m in markers:
        m_norm = _norm(m)
        idx = 0
        while True:
            pos = text_norm.find(m_norm, idx)
            if pos == -1:
                break
            # Look ahead up to 5 chars for a colon.
            tail = text_norm[pos + len(m_norm): pos + len(m_norm) + 5]
            if ":" in tail:
                return True
            idx = pos + len(m_norm)
    return False


def _excerpt_around(text: str, markers: tuple[str, ...], width: int = 60) -> str | None:
    """Return ``±width`` chars around the first marker hit, original casing."""
    text_norm = _norm(text)
    for m in markers:
        m_norm = _norm(m)
        pos = text_norm.find(m_norm)
        if pos != -1:
            start = max(0, pos - width)
            end = min(len(text), pos + len(m_norm) + width)
            snippet = text[start:end].replace("\n", " ").replace("\r", " ")
            snippet = re.sub(r"\s+", " ", snippet).strip()
            return f"…{snippet}…"
    return None


async def main() -> int:
    async with get_session() as session:
        result = await session.execute(select(BlingProduct))
        products = result.scalars().all()

    total = len(products)
    print(f"\nProdutos em bling_products: {total}\n")
    if total == 0:
        print("Banco vazio — rode o sync inicial antes de diagnosticar.")
        return 1

    # Per-attribute counters and sample buckets.
    already_has: dict[str, int] = {a: 0 for a in _TARGETS}
    recoverable: dict[str, int] = {a: 0 for a in _TARGETS}
    irrecoverable: dict[str, int] = {a: 0 for a in _TARGETS}
    samples: dict[str, list[tuple[str, str]]] = {a: [] for a in _TARGETS}

    for p in products:
        attrs = p.atributos_parseados or {}
        desc = _description_text(p)
        desc_norm = _norm(desc)

        for attr, markers in _TARGETS.items():
            in_attrs = bool(attrs.get(attr))
            in_desc = _has_label_in_text(desc_norm, markers)

            if in_attrs:
                already_has[attr] += 1
            elif in_desc:
                recoverable[attr] += 1
                if len(samples[attr]) < 10:
                    excerpt = _excerpt_around(desc, markers)
                    if excerpt:
                        samples[attr].append((p.nome or f"id={p.id}", excerpt))
            else:
                irrecoverable[attr] += 1

    # ── Coverage table ──────────────────────────────────────────────────
    print(f"{'atributo':<14} {'já tem':>10} {'recuperável':>14} {'irrecup.':>14}")
    print("-" * 56)
    for attr in _TARGETS:
        print(
            f"{attr:<14} "
            f"{already_has[attr]:>10} "
            f"{recoverable[attr]:>14} "
            f"{irrecoverable[attr]:>14}"
        )
    print()

    # ── Samples ─────────────────────────────────────────────────────────
    print("=" * 72)
    print("Amostras de texto cru (até 10 por atributo) — formato real do catálogo:")
    print("=" * 72)
    for attr, items in samples.items():
        if not items:
            continue
        print(f"\n— [{attr}] —")
        for i, (nome, snippet) in enumerate(items, 1):
            short_nome = (nome[:50] + "…") if len(nome) > 50 else nome
            print(f"  {i:>2}. {short_nome}")
            print(f"      {snippet}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
