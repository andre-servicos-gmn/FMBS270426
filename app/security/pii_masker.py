import hashlib
import hmac
import logging
import re
from typing import Final

from app.config import get_settings

logger = logging.getLogger(__name__)

# Patterns applied in order — more specific first to avoid partial overlaps.
# All 11-digit bare sequences (unformatted CPF or mobile) are caught last.
_PATTERNS: Final[list[tuple[re.Pattern[str], str]]] = [
    # Email — most distinctive (has @)
    (
        re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
        "[EMAIL]",
    ),
    # CPF formatted: 123.456.789-00
    (
        re.compile(r"(?<!\d)\d{3}\.\d{3}\.\d{3}-\d{2}(?!\d)"),
        "[CPF]",
    ),
    # Phone with area code in parentheses: (11) 99999-9999 or (11) 8888-8888
    (
        re.compile(r"\(\d{2}\)\s?9?\d{4}-?\d{4}"),
        "[FONE]",
    ),
    # Phone without parentheses but with space separator: 11 99999-9999
    # Uses [ \t] to avoid cross-line matches
    (
        re.compile(r"(?<!\d)\d{2}[ \t]9?\d{4}-\d{4}(?!\d)"),
        "[FONE]",
    ),
    # Address heuristic: rua / r. / av. / avenida + street name + number
    # [^\d\n]+? stops lazily at the first digit (the house/building number)
    (
        re.compile(r"(?i)\b(?:avenida|rua|r\.|av\.?)[ \t]+[^\d\n]+?\d+"),
        "[ENDERECO]",
    ),
    # 11-digit bare sequences — unformatted CPF or mobile without formatting
    (
        re.compile(r"(?<!\d)\d{11}(?!\d)"),
        "[CPF]",
    ),
    # CEP: 01310-100
    (
        re.compile(r"(?<!\d)\d{5}-\d{3}(?!\d)"),
        "[CEP]",
    ),
]


def mask_pii(text: str) -> str:
    """Return text with all detected PII replaced by placeholder tokens."""
    settings = get_settings()
    if not settings.pii_mask_enabled:
        logger.warning("pii_mask_enabled is False — masking skipped (dev only)")
        return text
    result = text
    for pattern, replacement in _PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def hash_phone(phone: str) -> str:
    """Return a deterministic HMAC-SHA256 hex digest of the normalized phone number.

    Normalization strips all non-digit characters so that "(11) 99999-9999"
    and "11999999999" produce the same hash.
    """
    normalized = "".join(c for c in phone if c.isdigit())
    salt = get_settings().pii_salt.encode("utf-8")
    return hmac.new(salt, normalized.encode("utf-8"), hashlib.sha256).hexdigest()


def is_clean(text: str) -> bool:
    """Return True if no PII pattern is detected in text.

    Intended for use in assertions and tests.
    """
    return mask_pii(text) == text
