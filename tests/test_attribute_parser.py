"""Sprint 2.6.7 — attribute parser tests.

Three areas:
1. REGRESSION — Padrão A (Mormaii Sunset Plus, leading-dash format) must
   keep parsing all 5 target attributes correctly.
2. NEW — Padrão B (Furia Attack, bare-label format) must start parsing.
3. ANTI-LIXO — marketing blocks ("conforto", "potência", "tecnologia",
   "design") must NEVER appear in atributos_parseados.

The fixtures live in ``tests/fixtures/desc_*.txt`` and reproduce real
descriptions from the user's catalog (paraphrased; safe to commit).
"""
from pathlib import Path

import pytest

from app.sync.bling_sync import parse_attributes_from_description

_FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> str:
    return (_FIXTURE_DIR / name).read_text(encoding="utf-8")


# ── REGRESSION — Padrão A (Mormaii Sunset Plus) ────────────────────────────

def test_mormaii_sunset_parses_peso():
    parsed = parse_attributes_from_description(_fixture("desc_mormaii_sunset.txt"))
    assert parsed.get("peso") == "320g (+/- 10g)"


def test_mormaii_sunset_parses_equilibrio():
    parsed = parse_attributes_from_description(_fixture("desc_mormaii_sunset.txt"))
    assert parsed.get("equilibrio", "").startswith("Aproximadamente")


def test_mormaii_sunset_parses_composicao():
    parsed = parse_attributes_from_description(_fixture("desc_mormaii_sunset.txt"))
    composicao = parsed.get("composicao", "")
    assert "Carbono" in composicao


def test_mormaii_sunset_parses_espessura():
    parsed = parse_attributes_from_description(_fixture("desc_mormaii_sunset.txt"))
    assert parsed.get("espessura") == "22mm"


def test_mormaii_sunset_parses_comprimento():
    parsed = parse_attributes_from_description(_fixture("desc_mormaii_sunset.txt"))
    assert parsed.get("comprimento") == "50cm"


def test_mormaii_sunset_all_five_present():
    parsed = parse_attributes_from_description(_fixture("desc_mormaii_sunset.txt"))
    for slug in ("peso", "equilibrio", "composicao", "espessura", "comprimento"):
        assert slug in parsed, f"missing {slug}: got {sorted(parsed)}"


# ── NEW — Padrão B (Furia Attack) ──────────────────────────────────────────

def test_furia_attack_parses_peso():
    parsed = parse_attributes_from_description(_fixture("desc_furia_attack.txt"))
    assert parsed.get("peso") == "220-240g"


def test_furia_attack_parses_composicao():
    parsed = parse_attributes_from_description(_fixture("desc_furia_attack.txt"))
    assert parsed.get("composicao") == "Carbono 3K"


def test_furia_attack_parses_espessura():
    parsed = parse_attributes_from_description(_fixture("desc_furia_attack.txt"))
    assert parsed.get("espessura") == "16mm"


def test_furia_attack_parses_comprimento():
    parsed = parse_attributes_from_description(_fixture("desc_furia_attack.txt"))
    assert parsed.get("comprimento") == "41,5cm"


# ── ANTI-LIXO — marketing must NOT appear as target attributes ────────────

def test_marketing_blocks_not_captured_as_specs():
    """conforto / potencia / precisao / durabilidade / tecnologia / design
    must NEVER show up among the 5 target keys, on either pattern."""
    for fixture in ("desc_mormaii_sunset.txt", "desc_furia_attack.txt"):
        parsed = parse_attributes_from_description(_fixture(fixture))
        # The dict only ever uses the 5 canonical slugs.
        forbidden = {
            "conforto", "potencia", "potência", "precisao", "precisão",
            "durabilidade", "tecnologia", "design",
            # Sprint 2.5.2 also captured these — restricted now.
            "perfil", "detalhamento",
        }
        leaked = forbidden & set(parsed.keys())
        assert not leaked, f"leaked marketing keys in {fixture}: {leaked}"


def test_only_5_target_slugs_can_appear():
    """The parser must NEVER emit any slug besides the 5 canonical targets."""
    allowed = {"peso", "equilibrio", "composicao", "espessura", "comprimento"}
    for fixture in ("desc_mormaii_sunset.txt", "desc_furia_attack.txt"):
        parsed = parse_attributes_from_description(_fixture(fixture))
        extra = set(parsed.keys()) - allowed
        assert not extra, f"unexpected slug in {fixture}: {extra}"


# ── FORMAT variants ───────────────────────────────────────────────────────

def test_label_with_and_without_dash():
    """Both "- Peso: X" AND "Peso: X" (bare) must work."""
    with_dash = "- Peso: 300g"
    without_dash = "Peso: 300g"
    assert parse_attributes_from_description(with_dash).get("peso") == "300g"
    assert parse_attributes_from_description(without_dash).get("peso") == "300g"


def test_value_range_format():
    parsed = parse_attributes_from_description("Peso: 220-240g")
    assert parsed.get("peso") == "220-240g"


def test_value_tolerance_format():
    parsed = parse_attributes_from_description("- Peso: 320g (+/- 10g)")
    assert parsed.get("peso") == "320g (+/- 10g)"


def test_accented_label_normalized_to_unaccented_slug():
    """Composição (with cedilla) maps to slug "composicao" (no accent)."""
    parsed = parse_attributes_from_description("Composição: Carbono 3K")
    assert "composicao" in parsed
    assert "composição" not in parsed


def test_synonym_balance_maps_to_equilibrio():
    parsed = parse_attributes_from_description("Balance: 27,5cm")
    assert parsed.get("equilibrio") == "27,5cm"
    parsed2 = parse_attributes_from_description("Balanço: 28cm")
    assert parsed2.get("equilibrio") == "28cm"


def test_synonym_material_maps_to_composicao():
    parsed = parse_attributes_from_description("Material: Fibra de carbono")
    assert parsed.get("composicao") == "Fibra de carbono"


def test_synonym_tamanho_maps_to_comprimento():
    parsed = parse_attributes_from_description("Tamanho: 50cm")
    assert parsed.get("comprimento") == "50cm"


def test_idempotent_on_repeated_parse():
    """Sprint 2.6.7 — running the parser twice on the same input must give
    bit-identical results. The reparse script depends on this."""
    text = _fixture("desc_furia_attack.txt")
    first = parse_attributes_from_description(text)
    second = parse_attributes_from_description(text)
    assert first == second


# ── Empty / degenerate inputs ─────────────────────────────────────────────

@pytest.mark.parametrize("text", ["", "   ", "<p></p>", "Description without colon"])
def test_empty_or_unstructured_returns_empty_dict(text):
    assert parse_attributes_from_description(text) == {}


def test_html_stripped_before_parsing():
    """<p>Peso: 300g</p> must work just like the bare text."""
    parsed = parse_attributes_from_description("<p>Peso: 300g</p>")
    assert parsed.get("peso") == "300g"
