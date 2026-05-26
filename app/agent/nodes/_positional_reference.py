"""Sprint 1.15 — detect positional references like 'a primeira' / '2ª' in
customer messages.

Conservative on purpose: a customer saying "primeira vez que jogo" is NOT
selecting a product. We only return an index when the positional word
appears near a choice-intent verb (quero, prefiro, gostei, vou de, fico
com, pega, leva, escolho, reserva). And only when the resolved index is a
valid position in the current options list.

Returns ``None`` when no clear positional intent is detected.
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


# Verbs / phrases that indicate the customer is making a CHOICE about one of
# the options shown. Reference must appear close to one of these.
_CHOICE_VERBS: Final[tuple[str, ...]] = (
    "quero", "prefiro", "gostei", "vou de", "vou com", "vou na", "vou pela",
    "fico com", "fico na", "pega", "leva", "escolho", "escolhe", "reserva",
    "manda", "me ve", "me ver", "vai", "to com", "tô com",
)

# Ordinal mapping. Includes "ultima"/"último" as -1 sentinel.
_POSITIONAL_TOKENS: Final[dict[str, int]] = {
    # 1st
    "primeira": 0, "primeiro": 0,
    "1a": 0, "1ª": 0, "1o": 0, "1º": 0, "1": 0, "primeira opcao": 0,
    "uma": 0,
    # 2nd
    "segunda": 1, "segundo": 1,
    "2a": 1, "2ª": 1, "2o": 1, "2º": 1, "2": 1, "segunda opcao": 1,
    # 3rd
    "terceira": 2, "terceiro": 2,
    "3a": 2, "3ª": 2, "3o": 2, "3º": 2, "3": 2,
    # last (-1 sentinel — resolved against num_options)
    "ultima": -1, "ultimo": -1, "ultima opcao": -1,
}

# Patterns that strongly bind a positional word to a choice — used to relax
# the requirement of a choice verb nearby (e.g. customer just types "a primeira").
_STRONG_PATTERNS: Final[tuple[str, ...]] = (
    "a primeira", "a segunda", "a terceira", "a ultima",
    "o primeiro", "o segundo", "o terceiro", "o ultimo",
    "1ª", "2ª", "3ª", "1a", "2a", "3a",
)

# Words that disqualify a positional reference even if it shows up next to a
# verb. "primeira vez", "primeiro contato", "primeiro dia" — common in
# pre-diagnose chatter.
_DISQUALIFY_NEXT_WORDS: Final[tuple[str, ...]] = (
    "vez", "contato", "dia", "encontro", "tentativa", "experiencia",
)


def _has_choice_context(norm_text: str) -> bool:
    """True if a choice verb appears anywhere in the (normalized) text."""
    return any(verb in norm_text for verb in _CHOICE_VERBS)


def _has_strong_pattern(norm_text: str) -> bool:
    return any(p in norm_text for p in _STRONG_PATTERNS)


def _word_after(norm_text: str, position_word: str) -> str | None:
    """Return the word immediately after the first occurrence of position_word."""
    match = re.search(rf"\b{re.escape(position_word)}\s+(\w+)", norm_text)
    return match.group(1) if match else None


def detect_positional_reference(text: str, num_options: int) -> int | None:
    """Return the 0-based index of the option the customer references, or None.

    Examples (assuming num_options=2):
        "quero a primeira"          → 0
        "vou de segunda"            → 1
        "2ª"                        → 1
        "fico com a última"         → 1
        "primeira vez que jogo"     → None   (no choice context)
        "minha primeira raquete"    → None   (followed by 'raquete', disqualified)
        "quero a quarta"            → None   (index out of range)
    """
    if not text or num_options <= 0:
        return None
    norm = _normalize(text)

    has_choice = _has_choice_context(norm)
    has_strong = _has_strong_pattern(norm)

    # If neither a choice verb NOR a strong "a primeira" pattern is present,
    # we don't try to disambiguate — too easy to misfire.
    if not (has_choice or has_strong):
        return None

    for token, idx in _POSITIONAL_TOKENS.items():
        # Match token as a whole word (allow boundaries on either side).
        if not re.search(rf"(?:^|\W){re.escape(token)}(?:\W|$)", norm):
            continue

        # Disqualify when followed by a common false-positive word.
        next_word = _word_after(norm, token)
        if next_word in _DISQUALIFY_NEXT_WORDS:
            continue

        resolved_idx = idx if idx >= 0 else num_options - 1
        if 0 <= resolved_idx < num_options:
            return resolved_idx

    return None
