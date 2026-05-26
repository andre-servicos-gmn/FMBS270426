"""Shared helpers for matching product names against free-text customer messages.

Sprint 1.15 — tolerant matcher with 4 layered strategies, from strict to
forgiving. Used by all follow-up nodes (price_inquiry, product_selection,
product_detail, re_recommendation) and by the triage router.

Layers (each one only runs if the previous returned no match):

    1. Exact normalized substring (lowercase + strip accents + strip
       'raquete '/'pala ' prefix). Confidence: high.
    2. Spaces-collapsed substring — drop ALL whitespace from both name and
       text. Resolves "beach pro foam series 300" vs "BeachPro Foam Series
       300". Confidence: high.
    3. Fuzzy match via difflib.SequenceMatcher with a sliding window the
       size of the candidate name. Ratio ≥ 0.95 → high confidence (typo).
       Ratio ≥ 0.85 → low confidence (heavier typo or partial match).
    4. Token uniqueness — a 5+ char non-generic token in the customer's
       message that belongs to exactly ONE product in the shortlist.
       Confidence: low.

Returns a ``MatchResult`` so the caller can route differently for
high vs low confidence. ``match_product_in_text`` is preserved as a
backward-compatible wrapper that just returns the product (or None).
"""
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text or "")
        if unicodedata.category(c) != "Mn"
    )


def normalize(text: str) -> str:
    return strip_accents((text or "").lower().strip())


_GENERIC_TOKENS = frozenset({
    "raquete", "pala", "carbon", "pro", "elite", "series", "edition", "edicao",
    "edição", "beach", "padel", "x", "v",
})


def _core_name(text: str) -> str:
    """Return the normalized name without the generic 'raquete '/'pala ' prefix."""
    norm = normalize(text)
    for prefix in ("raquete ", "pala "):
        if norm.startswith(prefix):
            return norm[len(prefix):].strip()
    return norm


def _strip_spaces(s: str) -> str:
    """Return ``s`` with every whitespace character removed."""
    return "".join(s.split())


# ── Public types ─────────────────────────────────────────────────────────────

Confidence = Literal["high", "low", "none"]
Method = Literal["exact", "spaces_collapsed", "fuzzy", "token", "none"]


@dataclass
class MatchResult:
    """Outcome of a tolerant product-name match."""

    product: dict | None
    confidence: Confidence
    method: Method

    @property
    def found(self) -> bool:
        return self.product is not None


_NO_MATCH = MatchResult(product=None, confidence="none", method="none")


# Thresholds for fuzzy matching.
_FUZZY_HIGH_RATIO = 0.95
_FUZZY_LOW_RATIO = 0.85
# Minimum length for a meaningful match — guards against trivial substring noise.
_MIN_FULL_VARIANT = 4


# ── Layer helpers ────────────────────────────────────────────────────────────

def _name_variants(product: dict) -> list[str]:
    """Return the candidate name forms to compare, ordered by specificity."""
    name = product.get("name") or ""
    full = normalize(name)
    core = _core_name(name)
    out: list[str] = []
    if full and len(full) >= _MIN_FULL_VARIANT:
        out.append(full)
    if core and core != full and len(core) >= _MIN_FULL_VARIANT:
        out.append(core)
    return out


def _try_exact(text_norm: str, products: list[dict]) -> dict | None:
    """Layer 1 — normalized substring match. Returns longest hit or None."""
    candidates: list[tuple[int, dict]] = []
    for p in products:
        for variant in _name_variants(p):
            if variant in text_norm:
                candidates.append((len(variant), p))
                break
    if not candidates:
        return None
    candidates.sort(key=lambda pair: -pair[0])
    return candidates[0][1]


def _try_spaces_collapsed(text_norm: str, products: list[dict]) -> dict | None:
    """Layer 2 — drop every whitespace and try again.

    Fixes the real-world bug: customer typed "beach pro foam series 300"
    but the catalog has "BeachPro Foam Series 300". After whitespace
    removal both become "beachprofoamseries300".
    """
    text_collapsed = _strip_spaces(text_norm)
    if not text_collapsed:
        return None
    candidates: list[tuple[int, dict]] = []
    for p in products:
        for variant in _name_variants(p):
            v = _strip_spaces(variant)
            if v and len(v) >= _MIN_FULL_VARIANT and v in text_collapsed:
                candidates.append((len(v), p))
                break
    if not candidates:
        return None
    candidates.sort(key=lambda pair: -pair[0])
    return candidates[0][1]


def _best_fuzzy_ratio(text_norm: str, name_norm: str) -> float:
    """Return the best SequenceMatcher ratio of any same-length window of
    ``text_norm`` against ``name_norm``.

    For short messages (most WhatsApp follow-ups) this is cheap. We slide a
    window of len(name_norm) across text_norm and pick the highest ratio.
    """
    if not text_norm or not name_norm:
        return 0.0
    name_len = len(name_norm)
    if len(text_norm) <= name_len:
        return SequenceMatcher(None, text_norm, name_norm).ratio()
    best = 0.0
    for i in range(len(text_norm) - name_len + 1):
        window = text_norm[i:i + name_len]
        ratio = SequenceMatcher(None, window, name_norm).ratio()
        if ratio > best:
            best = ratio
            if best >= 0.999:  # already perfect — stop early
                break
    return best


def _try_fuzzy(text_norm: str, products: list[dict]) -> tuple[dict | None, Confidence]:
    """Layer 3 — fuzzy match with confidence based on ratio thresholds."""
    text_collapsed = _strip_spaces(text_norm)

    best_product: dict | None = None
    best_ratio = 0.0
    for p in products:
        for variant in _name_variants(p):
            # Compare both with-spaces and collapsed forms; take the higher.
            v_collapsed = _strip_spaces(variant)
            ratio = max(
                _best_fuzzy_ratio(text_norm, variant),
                _best_fuzzy_ratio(text_collapsed, v_collapsed),
            )
            if ratio > best_ratio:
                best_ratio = ratio
                best_product = p

    if best_product is None or best_ratio < _FUZZY_LOW_RATIO:
        return None, "none"
    if best_ratio >= _FUZZY_HIGH_RATIO:
        return best_product, "high"
    return best_product, "low"


def _try_token_unicity(text_norm: str, products: list[dict]) -> dict | None:
    """Layer 4 — a 5+ char non-generic token belongs to exactly 1 product."""
    token_owners: dict[str, set[int]] = {}
    for idx, p in enumerate(products):
        name_norm = normalize(p.get("name") or "")
        for token in name_norm.split():
            if len(token) < 5 or token in _GENERIC_TOKENS:
                continue
            token_owners.setdefault(token, set()).add(idx)

    best_product: dict | None = None
    best_score = 0
    for token, owners in token_owners.items():
        if len(owners) != 1:
            continue
        if token in text_norm:
            score = len(token)
            if score > best_score:
                idx = next(iter(owners))
                best_product = products[idx]
                best_score = score
    return best_product


# ── Public API ───────────────────────────────────────────────────────────────

def match_product_tolerant(text: str, products: list[dict]) -> MatchResult:
    """Return the best match with explicit confidence + method.

    Layered evaluation: exact → spaces_collapsed → fuzzy → token.
    First high-confidence hit wins. Token unicity is always low-confidence.
    """
    if not text or not products:
        return _NO_MATCH
    text_norm = normalize(text)
    if not text_norm:
        return _NO_MATCH

    # Layer 1 — exact normalized.
    p = _try_exact(text_norm, products)
    if p is not None:
        return MatchResult(product=p, confidence="high", method="exact")

    # Layer 2 — collapse whitespace and try substring again.
    p = _try_spaces_collapsed(text_norm, products)
    if p is not None:
        return MatchResult(product=p, confidence="high", method="spaces_collapsed")

    # Layer 3 — fuzzy.
    p, conf = _try_fuzzy(text_norm, products)
    if p is not None:
        return MatchResult(product=p, confidence=conf, method="fuzzy")

    # Layer 4 — token uniqueness, conservative.
    p = _try_token_unicity(text_norm, products)
    if p is not None:
        return MatchResult(product=p, confidence="low", method="token")

    return _NO_MATCH


def match_product_in_text(text: str, products: list[dict]) -> dict | None:
    """Backward-compatible wrapper: returns just the product (or None).

    Sprint 1.15 — internally uses the new tolerant pipeline. Callers that
    don't care about confidence keep working unchanged.
    """
    return match_product_tolerant(text, products).product


# ── Other shared helpers ─────────────────────────────────────────────────────

def format_price_brl(price_cents: int | float | None) -> str:
    """Return a BRL-formatted price string like 'R$ 1.299' (no centavos)."""
    if price_cents is None:
        return "R$ -"
    try:
        reais = int(price_cents) // 100
    except (TypeError, ValueError):
        return "R$ -"
    return f"R$ {reais:,}".replace(",", ".")
