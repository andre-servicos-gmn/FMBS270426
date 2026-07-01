"""T5 — travessão guarantee enforced deterministically on OUTPUT.

The canonical guard is a backstop in ``_sanitize_for_whatsapp`` (a prompt
instruction can't reliably stop gpt-4o-mini from emitting an em-dash). These
pin: em-dash (—, U+2014) and en-dash (–, U+2013) are removed from the final
answer, the list hyphen (-) is preserved, and the replacement inserts a
separator so words don't glue together.

RED until dev/persona add the dash normalization to ``_sanitize_for_whatsapp``.
"""
import pytest

from app.agent.supervisor import _sanitize_for_whatsapp

pytestmark = pytest.mark.deterministic

_EM_DASH = "—"  # —
_EN_DASH = "–"  # –


def test_sanitizer_strips_em_dash():
    out = _sanitize_for_whatsapp("A Kronos é boa — e bem leve.")
    assert _EM_DASH not in out, f"em-dash survived sanitize: {out!r}"


def test_sanitizer_strips_en_dash():
    out = _sanitize_for_whatsapp("Faixa de 449 – 1799 reais.")
    assert _EN_DASH not in out, f"en-dash survived sanitize: {out!r}"


def test_sanitizer_keeps_hyphen_in_list():
    """A store attendant lists items with a plain hyphen — that must survive."""
    out = _sanitize_for_whatsapp("Raquete Drop Shot - R$ 449,00")
    assert "-" in out, f"list hyphen was stripped: {out!r}"
    assert "R$ 449,00" in out


def test_sanitizer_em_dash_no_word_glue():
    """Replacing the dash must insert a separator, not delete it and glue words."""
    out = _sanitize_for_whatsapp("boa—leve")
    assert _EM_DASH not in out
    assert "boaleve" not in out, f"words glued after dash removal: {out!r}"
