"""Sprint 1.15 — positional reference detection tests."""
import pytest

from app.agent.nodes._positional_reference import detect_positional_reference


# Helper: a "2-option" list so the function has a valid range to resolve against.
NUM = 2


@pytest.mark.parametrize(
    "text,expected_idx",
    [
        ("quero a primeira", 0),
        ("Quero a Primeira", 0),
        ("vou de primeiro", 0),
        ("vou de segunda", 1),
        ("prefiro a segunda", 1),
        ("fico com a primeira", 0),
        ("a primeira mesmo", 0),
        ("pega a 1ª", 0),
        ("pega a 2ª", 1),
        ("vou na 1a", 0),
        ("vou na 2a", 1),
        ("fico com a ultima", 1),  # last of 2 → idx 1
        ("vou de última opção", 1),
    ],
)
def test_detects_positional(text, expected_idx):
    assert detect_positional_reference(text, NUM) == expected_idx


def test_detects_terceira_only_when_in_range():
    """terceira returns idx 2 only when there are 3+ options."""
    assert detect_positional_reference("quero a terceira", 3) == 2
    # With only 2 options, terceira is out of range.
    assert detect_positional_reference("quero a terceira", 2) is None


def test_returns_none_when_no_choice_context():
    """'primeira vez', 'primeiro dia' etc are NOT product references."""
    assert detect_positional_reference("primeira vez que jogo beach tennis", NUM) is None
    assert detect_positional_reference("é meu primeiro dia jogando", NUM) is None
    assert detect_positional_reference("a primeira experiência foi ótima", NUM) is None


def test_returns_none_when_index_out_of_range():
    assert detect_positional_reference("quero a quarta", 2) is None
    assert detect_positional_reference("vou de quinta", 3) is None


def test_returns_none_for_empty_or_no_options():
    assert detect_positional_reference("quero a primeira", 0) is None
    assert detect_positional_reference("", 3) is None
    assert detect_positional_reference("", 0) is None


def test_strong_pattern_works_without_explicit_verb():
    """'a primeira' alone is strong enough — no choice verb required."""
    assert detect_positional_reference("a primeira", NUM) == 0
    assert detect_positional_reference("a segunda", NUM) == 1
