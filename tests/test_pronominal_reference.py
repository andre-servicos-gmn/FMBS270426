"""Sprint 1.15 — pronominal reference detection tests."""
import pytest

from app.agent.nodes._pronominal_reference import detect_pronominal_reference


@pytest.mark.parametrize(
    "text",
    [
        "gostei dessa",
        "gostei muito dessa",
        "gostei dela",
        "essa aí",
        "essa aí mesmo",
        "essa mesmo",
        "essa serve",
        "vou de essa",
        "fico com essa",
        "pode reservar essa",
        "reserva essa",
        "quero essa",
        "vou levar essa",
        "leva essa",
        "manda essa",
        "pode ser essa",
        # Bare short-message approvals
        "essa",
        "essa.",
        "essa!",
        "essa mesmo",
    ],
)
def test_detects_pronominal_choice(text):
    assert detect_pronominal_reference(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "",
        "oi tudo bem?",
        "essa raquete tem peso quanto?",  # "essa" + "raquete" — not a selection
        "quanto custa?",
        "tem outra opção?",
        "vocês entregam em casa?",
        "quero uma raquete",  # generic, not a choice on shown options
    ],
)
def test_returns_false_for_neutral_text(text):
    assert detect_pronominal_reference(text) is False
