"""Sprint 1.15 — detect pronominal references like 'gostei dessa' / 'essa aí'.

Returns True/False. The caller (triage router) decides what to do with it:
    - If True and only 1 product on the table → product_selection
    - If True and multiple products → ambiguous_selection (ask which one)

Conservative: we only match when the demonstrative pronoun is near a
choice/approval verb. Bare "essa" without context is too easy to misread.
"""
import re
import unicodedata
from typing import Final


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text or "")
        if unicodedata.category(c) != "Mn"
    )


def _normalize(text: str) -> str:
    return _strip_accents((text or "").lower().strip())


# Phrases that pair a demonstrative pronoun with a choice/approval signal.
# Order doesn't matter — we just check substring presence in normalized text.
_PRONOMINAL_PATTERNS: Final[tuple[str, ...]] = (
    # "gostei dessa/dela/desse/dele"
    "gostei dessa", "gostei dela", "gostei desse", "gostei dele",
    "gostei muito dessa", "gostei muito dela",
    # "essa/essa mesmo/essa aí/essa serve"
    "essa ai", "essa mesmo", "essa serve", "essa pra mim",
    "esse mesmo", "esse serve", "esse aí", "esse ai",
    # "vou de essa/fico com essa/pode reservar essa"
    "vou de essa", "vou nessa", "fico com essa", "fico nessa",
    "pode reservar essa", "pode reservar essa mesma",
    "pode reservar essa ai", "reserva essa", "reserva esta",
    # "quero essa", "vou levar essa"
    "quero essa", "vou levar essa", "vou ficar com essa",
    "leva essa", "manda essa",
    # "pode ser essa", "essa aí mesmo"
    "pode ser essa", "essa ai mesmo", "essa aí mesmo",
)

# Sometimes "essa" alone is a clear approval signal — only when it's the
# entire message or near a polarity word. We test as exact short messages.
_BARE_APPROVAL_SHORT_MESSAGES: Final[frozenset[str]] = frozenset({
    "essa", "essa.", "essa!", "esse", "esse.", "esse!",
    "essa mesmo", "esse mesmo",
    "essa ai", "essa aí",
    "perfeita", "perfeita.",
})


def detect_pronominal_reference(text: str) -> bool:
    """Return True when the customer is pointing at "this/that one" with
    approval intent — without naming the product explicitly."""
    if not text:
        return False
    norm = _normalize(text)
    if not norm:
        return False

    # Exact short-message approvals.
    if norm in _BARE_APPROVAL_SHORT_MESSAGES:
        return True

    # Phrasal patterns.
    if any(p in norm for p in _PRONOMINAL_PATTERNS):
        return True

    # "essa" / "esse" near "reservar" / "comprar" / "marca" within a small window.
    nearby = re.search(
        r"\b(essa|esse)\b[\s\w]{0,15}\b(reserv|compr|leva|fech|marca|pega)",
        norm,
    )
    if nearby:
        return True

    return False
