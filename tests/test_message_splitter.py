"""Tests for app.agent.message_splitter — Sprint 1.6 message block + delay."""
import json
from unittest.mock import patch

import pytest

from app.agent.message_splitter import compute_typing_delay, parse_messages


# ── parse_messages — happy path ──────────────────────────────────────────────

def test_parse_messages_from_valid_json_list():
    """LLM JSON output {"messages": [...]} must be returned verbatim, trimmed."""
    payload = json.dumps({"messages": [
        "Recomendo a Carbon X5 pelo seu perfil intermediário.",
        "Posso reservar para você?",
        "Se quiser ter mais certeza, temos a Consultoria — você testa em quadra.",
    ]})
    blocks = parse_messages(payload)
    assert len(blocks) == 3
    assert blocks[0].startswith("Recomendo a Carbon X5")
    assert blocks[1].startswith("Posso reservar")
    assert blocks[2].startswith("Se quiser")


def test_parse_messages_accepts_dict_directly():
    """If a node already parsed the JSON to a dict, parse_messages must accept it."""
    blocks = parse_messages({"messages": ["a", "b"]})
    assert blocks == ["a", "b"]


# ── numbered product list must never break mid-item (production regression) ──

def test_numbered_product_list_never_splits_mid_item():
    """Production bug: a numbered raquete list broke into balloons mid-item
    ('...459,00\\n3.' | 'Raquete...'). A list must only break BETWEEN whole
    items — every block line is a complete item (or the intro)."""
    import re
    txt = (
        "Aqui estão algumas raquetes de Beach Tennis até R$ 2000:\n"
        "1. Raquete de Beach Tennis Drop Shot Pentax 3.0 Fibra Iniciante - R$ 449,00\n"
        "2. Raquete de Beach Tennis Drop Shot Stage Pro 1.0 BT Iniciante - R$ 459,00\n"
        "3. Raquete De Beach Tennis Drop Shot Nilo Red Iniciante - R$ 469,00\n"
        "4. Raquete de Beach Tennis DROP SHOT KEY Coco RED Iniciante - R$ 469,00\n"
        "5. Raquete De Beach Tennis Drop Shot Key Coco Blue Original - R$ 469,00\n"
        "6. Raquete de Beach Tennis Drop Shot Tiger 2.0 Iniciante - R$ 469,00"
    )
    blocks = parse_messages(txt)
    # No block may START with a bare item number missing its text, and no block
    # may END with a dangling marker like "3." — i.e. every numbered line that
    # appears must carry the product name + price on the same line.
    orphan = re.compile(r"(?m)^\s*\d{1,2}\.\s*$")
    for b in blocks:
        assert not orphan.search(b), f"orphan list marker in block: {b!r}"
        # Every numbered line in the block has a price on the same line.
        for line in b.splitlines():
            m = re.match(r"^\s*\d{1,2}\.\s", line)
            if m:
                assert "R$" in line, f"item line split from its price: {line!r}"


def test_short_list_stays_single_block():
    """A 2-3 item short list under the threshold stays in one balloon."""
    txt = (
        "Tem opção a partir de R$ 449:\n"
        "- Drop Shot Pentax 3.0 - R$ 449,00\n"
        "- Drop Shot Stage Pro - R$ 459,00"
    )
    blocks = parse_messages(txt)
    assert len(blocks) == 1


def test_parse_messages_accepts_plain_list():
    blocks = parse_messages(["um", "dois"])
    assert blocks == ["um", "dois"]


# ── parse_messages — fallback paths ──────────────────────────────────────────

def test_parse_messages_short_string_stays_single_block():
    """Strings under the short threshold come back as ONE block, not split."""
    short = "Tudo certo, posso reservar pra você?"
    blocks = parse_messages(short)
    assert blocks == [short]


def test_parse_messages_fallback_from_long_string_with_paragraphs():
    """A long string broken by blank lines must split on paragraph boundaries."""
    text = (
        "Para o seu perfil intermediário e sem lesões, recomendo a BeachPro Carbon X5. "
        "Ela combina boa potência e controle e funciona muito bem para quem já tem base.\n\n"
        "Outra opção válida é a BeachPro Foam Series 300, mais leve e acessível, "
        "indicada se você quer começar mais conservador.\n\n"
        "Posso reservar uma das duas para você passar na loja? Se quiser ter mais certeza, "
        "temos a Consultoria Base Esportes onde você testa em quadra antes de comprar."
    )
    blocks = parse_messages(text)
    assert 2 <= len(blocks) <= 4
    # Each block contains roughly one coherent idea
    assert "Carbon X5" in blocks[0]
    assert any("Foam" in b for b in blocks)
    assert any("Consultoria" in b or "reservar" in b for b in blocks)


def test_parse_messages_fallback_from_long_string_without_paragraphs():
    """Long single paragraph with no blank lines falls back to sentence split."""
    text = (
        "Para seu perfil intermediário e sem lesões essa raquete é uma boa opção. "
        "Ela combina potência e controle de forma equilibrada. "
        "Quem vem do tênis se adapta bem a esse modelo. "
        "Posso reservar para você passar na loja? "
        "Se quiser, temos a Consultoria Base Esportes para testar em quadra. "
        "Avisa qual prefere?"
    )
    assert "\n\n" not in text
    blocks = parse_messages(text)
    assert len(blocks) >= 2, f"expected sentence-split into >=2 blocks, got {blocks}"
    for b in blocks:
        assert len(b) <= 500


def test_parse_messages_handles_empty_input():
    """Empty / None inputs return an empty list — caller must handle no-reply."""
    assert parse_messages("") == []
    assert parse_messages("   \n  ") == []
    assert parse_messages(None) == []
    assert parse_messages({}) == []
    assert parse_messages([]) == []


def test_parse_messages_malformed_json_falls_back_to_text_split():
    """An LLM that opened with '{' but produced invalid JSON shouldn't crash."""
    bad = '{"messages": ["broken'
    # Won't parse as JSON — falls through to string split. Since len < 200,
    # it becomes a single block.
    blocks = parse_messages(bad)
    assert blocks == [bad]


def test_parse_messages_drops_empty_blocks_inside_list():
    blocks = parse_messages({"messages": ["primeiro", "", "  ", "segundo"]})
    assert blocks == ["primeiro", "segundo"]


def test_parse_messages_splits_oversize_block_in_list():
    """A single >500-char block inside the list is sentence-split, not truncated."""
    huge = (
        "Frase A com mais detalhe técnico. "
        "Frase B explicando características. "
        "Frase C trazendo o motivo da indicação. "
    ) * 8  # ~640 chars
    blocks = parse_messages({"messages": [huge]})
    assert len(blocks) >= 2
    assert all(len(b) <= 500 for b in blocks)


# ── compute_typing_delay ─────────────────────────────────────────────────────

def test_compute_typing_delay_short_message():
    """Short messages (<50 chars) pull from the 1.0-1.5 base bucket."""
    # Force the random values to the bucket center for determinism.
    with patch("app.agent.message_splitter.random.uniform") as ru:
        ru.side_effect = [1.25, 0.0]  # base, then jitter
        delay = compute_typing_delay("oi tudo bem?")
    assert delay == pytest.approx(1.25, abs=0.001)


def test_compute_typing_delay_long_message():
    """Long messages (>150 chars) sit in the 2.0-3.0 base bucket."""
    long_text = "x" * 200
    with patch("app.agent.message_splitter.random.uniform") as ru:
        ru.side_effect = [2.5, 0.0]
        delay = compute_typing_delay(long_text)
    assert delay == pytest.approx(2.5, abs=0.001)


def test_compute_typing_delay_within_1_to_3_seconds_range():
    """Across random draws, the delay must always be clamped to [1.0, 3.0]."""
    for length in (10, 75, 200, 400):
        text = "x" * length
        for _ in range(50):
            d = compute_typing_delay(text)
            assert 1.0 <= d <= 3.0
