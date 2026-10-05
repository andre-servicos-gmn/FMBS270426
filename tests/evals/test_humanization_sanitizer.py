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


def test_sanitizer_semicolon_becomes_sentence_break():
    out = _sanitize_for_whatsapp("A Kronos usa carbono 12K; a Proteo usa 18K.")
    assert out == "A Kronos usa carbono 12K. A Proteo usa 18K.", out


def test_sanitizer_semicolon_without_following_space_untouched():
    # Not a sentence pause (e.g. a pasted code/list token) — leave it alone.
    assert _sanitize_for_whatsapp("tamanhos P;M;G") == "tamanhos P;M;G"


# ── Re-apresentação fora do 1º turno (backstop no sanitize_node) ─────────────

def _run_sanitize(history_answers: list[str], final: str) -> str:
    from langchain_core.messages import AIMessage, HumanMessage

    from app.agent.supervisor import sanitize_node

    msgs = []
    for a in history_answers:
        msgs += [HumanMessage(content="oi"), AIMessage(content=a)]
    last = AIMessage(content=final)
    msgs += [HumanMessage(content="quero uma raquete"), last]
    out = sanitize_node({"messages": msgs})
    return out["messages"][0].content if out else final


def test_reintro_stripped_after_first_turn():
    out = _run_sanitize(
        ["Boa tarde! Sou o assistente Base, da Base Sports. O que você procura?"],
        "Boa tarde! Sou o assistente Base, da Base Sports. Tem alguma marca em mente?",
    )
    assert out == "Tem alguma marca em mente?", out


def test_intro_kept_on_first_turn():
    first = "Fala! Sou o assistente Base, da Base Sports. O que você procura?"
    assert _run_sanitize([], first) == first


def test_reintro_only_message_is_kept():
    # Nothing left after stripping (customer just said "oi" again) → keep it.
    text = "Oi! Sou o assistente Base, da Base Sports."
    assert _run_sanitize(["Oi! Sou o assistente Base, da Base Sports. Em que posso ajudar?"], text) == text


def test_mid_answer_mention_not_stripped():
    text = "A Kronos custa R$ 1.190. Sou o assistente Base e posso te mostrar outras."
    assert _run_sanitize(["Oi! Sou o assistente Base, da Base Sports."], text) == text
