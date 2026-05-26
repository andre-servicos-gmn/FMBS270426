"""Sprint 2.3 — question-type classification for the contextual pitch.

The subtle Consultoria pitch (cliente determinado) now varies BOTH by
content AND by timing, depending on what the customer is asking:

- IMMEDIATE types (PRICE, STOCK, FITNESS, COMFORT) → pitch can be emitted
  on the very first question — these signal proximity to closing or a
  doubt about adequacy. Acting fast is OK.
- DELAYED types (WEIGHT, MATERIAL, BALANCE, OTHER) → pitch waits until
  the customer's second technical question, so we don't push the
  Consultoria onto every casual curiosity.

This module is intentionally LLM-free: keyword + accent-insensitive
substring matching. Determinism > variation here — the pitch text is
sensitive (price + conditions) so any classification ambiguity should
favour the safer DELAYED branch via ``OTHER``.
"""
from __future__ import annotations

import unicodedata
from enum import Enum


class QuestionType(str, Enum):
    """Categories used by ``maybe_add_subtle_consultoria_offer`` to pick
    the right preset and the right timing."""

    PRICE = "price"          # "quanto custa", "preço"
    STOCK = "stock"          # "vocês têm a X?", "tá disponível"
    FITNESS = "fitness"      # "serve pra mim?", "é boa pro meu nível?"
    COMFORT = "comfort"      # "antivibração?", "evita lesão?"
    WEIGHT = "weight"        # "peso", "leve"
    MATERIAL = "material"    # "material", "carbono"
    BALANCE = "balance"      # "balance", "ponto de impacto"
    OTHER = "other"          # default / catch-all


# Types whose pitch may fire on the FIRST determined question. Everything
# else waits for the 2nd question (delayed path).
# Sprint 2.4: STOCK moved to DELAYED — stock confirmation is now pitch-free
# by design (cliente que pede pra ver se temos a raquete X não precisa de
# pressão pra Consultoria na confirmação; só depois, se houver dúvida real).
IMMEDIATE_TYPES: frozenset[QuestionType] = frozenset({
    QuestionType.PRICE,
    QuestionType.FITNESS,
    QuestionType.COMFORT,
})

DELAYED_TYPES: frozenset[QuestionType] = frozenset({
    QuestionType.STOCK,
    QuestionType.WEIGHT,
    QuestionType.MATERIAL,
    QuestionType.BALANCE,
    QuestionType.OTHER,
})


# Order matters when keywords overlap (FITNESS sometimes contains "boa"
# which is also a generic praise — we check FITNESS before OTHER). Each
# tuple is (QuestionType, list of normalized substrings).
_CLASSIFY_ORDER: list[tuple[QuestionType, tuple[str, ...]]] = [
    (
        QuestionType.PRICE,
        (
            "quanto custa", "quanto sai", "quanto e", "qual o preco",
            "qual preco", "qual valor", "preco", "valor", "investimento",
            "quanto fica", "quanto da", "parcela",
        ),
    ),
    (
        QuestionType.STOCK,
        (
            "voces tem", "voce tem", "vcs tem", "vc tem",
            "tem em estoque", "ta disponivel", "esta disponivel",
            "tem essa", "tem a ", "tem o ", "tem ela",
            "tem disponivel", "disponivel",
        ),
    ),
    (
        QuestionType.FITNESS,
        (
            "serve pra mim", "serve mesmo", "boa pra mim", "boa pro meu",
            "ideal pra mim", "combina comigo", "combina com meu",
            "combina pra mim", "vai bem pra mim", "ela e boa pra meu",
            "essa funciona pra mim", "essa atende",
            "boa pro meu nivel", "boa para mim",
        ),
    ),
    (
        QuestionType.COMFORT,
        (
            "antivibracao", "anti vibracao", "vibracao",
            "machuca", "machuca o", "machuca a",
            "dor", "dolorida", "lesao", "tendinite", "epicondilite",
            "conforto", "confortavel", "evita lesao", "evita dor",
            "absorcao de impacto", "absorve impacto", "amortece",
        ),
    ),
    (
        QuestionType.WEIGHT,
        (
            "peso", "pesa", "pesada", "pesado", "leve", "gramas",
            "quantos gramas", "quanto pesa",
        ),
    ),
    (
        QuestionType.MATERIAL,
        (
            "material", "fibra", "carbono", "fibra de vidro",
            "fibra de carbono", "nucleo", "núcleo", "eva",
            "polipropileno", "feita de", "feito de", "composicao",
        ),
    ),
    (
        QuestionType.BALANCE,
        (
            "balance", "balanceamento", "ponto de impacto",
            "centro de impacto", "head light", "head heavy",
            "cabeca leve", "cabeca pesada", "centro",
        ),
    ),
]


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def _normalize(text: str) -> str:
    return _strip_accents((text or "").lower())


def classify_question(text: str) -> QuestionType:
    """Return the ``QuestionType`` that best matches the customer's text.

    Returns ``QuestionType.OTHER`` when no keyword class hits. Empty / None
    input also returns ``OTHER``.

    The classifier is intentionally simple: first hit by category order
    wins. Ambiguous cases (e.g. "qual o peso e o preço?") favor the
    earlier category in ``_CLASSIFY_ORDER`` — PRICE wins over WEIGHT here,
    which is fine because PRICE is the more pitch-relevant signal.
    """
    if not text:
        return QuestionType.OTHER

    norm = _normalize(text)
    for q_type, needles in _CLASSIFY_ORDER:
        if any(needle in norm for needle in needles):
            return q_type
    return QuestionType.OTHER


def is_immediate(question_type: QuestionType) -> bool:
    """True when the question type allows pitching on the FIRST determined turn."""
    return question_type in IMMEDIATE_TYPES
