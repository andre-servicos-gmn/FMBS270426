import pytest

from app.security.pii_masker import hash_phone, is_clean, mask_pii


# ---------------------------------------------------------------------------
# CPF
# ---------------------------------------------------------------------------

class TestMaskCPF:
    def test_formatted_with_dots_and_dash(self) -> None:
        assert mask_pii("Meu CPF é 123.456.789-00") == "Meu CPF é [CPF]"

    def test_formatted_mid_sentence(self) -> None:
        assert mask_pii("CPF 000.000.001-91 para verificar") == "CPF [CPF] para verificar"

    def test_unformatted_11_digits(self) -> None:
        result = mask_pii("CPF: 12345678900")
        assert result == "CPF: [CPF]"

    def test_cpf_not_matched_inside_longer_number(self) -> None:
        # 13 consecutive digits must NOT be masked as a CPF
        result = mask_pii("1234567890012")
        assert result == "1234567890012"


# ---------------------------------------------------------------------------
# Phone
# ---------------------------------------------------------------------------

class TestMaskPhone:
    def test_ddd_parens_with_nine(self) -> None:
        assert mask_pii("Ligue para (11) 99999-9999") == "Ligue para [FONE]"

    def test_ddd_parens_without_nine(self) -> None:
        assert mask_pii("Tel: (11) 8888-8888") == "Tel: [FONE]"

    def test_ddd_parens_no_space(self) -> None:
        assert mask_pii("(21)98765-4321") == "[FONE]"

    def test_ddd_space_no_parens_with_nine(self) -> None:
        assert mask_pii("Whats 11 99999-9999 ok") == "Whats [FONE] ok"

    def test_ddd_space_no_parens_without_nine(self) -> None:
        assert mask_pii("Fone 11 8888-8888 aqui") == "Fone [FONE] aqui"

    def test_unformatted_11_digits_masked(self) -> None:
        # 11 bare digits are treated as PII regardless of phone vs CPF ambiguity
        result = mask_pii("contato 11999998888 ok")
        assert "[CPF]" in result or "[FONE]" in result


# ---------------------------------------------------------------------------
# Address
# ---------------------------------------------------------------------------

class TestMaskAddress:
    def test_rua_with_number(self) -> None:
        assert mask_pii("Moro na Rua das Flores 123") == "Moro na [ENDERECO]"

    def test_av_without_dot(self) -> None:
        assert mask_pii("Trabalho na Av Paulista, 1000") == "Trabalho na [ENDERECO]"

    def test_avenida_full_word(self) -> None:
        assert mask_pii("Fica na Avenida Brasil 500") == "Fica na [ENDERECO]"

    def test_r_dot_abbreviation(self) -> None:
        assert mask_pii("Endereço: r. XV de Novembro, 200") == "Endereço: [ENDERECO]"

    def test_address_case_insensitive(self) -> None:
        assert mask_pii("RUA AUGUSTA 900") == "[ENDERECO]"


# ---------------------------------------------------------------------------
# Clean text
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_plain_message_unchanged(self) -> None:
        text = "Quero agendar uma aula de beach tennis amanhã"
        assert mask_pii(text) == text

    def test_is_clean_true(self) -> None:
        assert is_clean("Quero jogar padel") is True

    def test_is_clean_false_cpf(self) -> None:
        assert is_clean("CPF: 123.456.789-00") is False

    def test_is_clean_false_phone(self) -> None:
        assert is_clean("Fone: (11) 99999-9999") is False

    def test_is_clean_false_email(self) -> None:
        assert is_clean("Email: user@example.com") is False

    def test_is_clean_false_address(self) -> None:
        assert is_clean("Moro na Rua das Flores 123") is False

    def test_number_in_context_not_masked(self) -> None:
        # Short numbers that are not PII must pass through
        assert mask_pii("Tenho 5 raquetes e 2 bolas") == "Tenho 5 raquetes e 2 bolas"


# ---------------------------------------------------------------------------
# hash_phone
# ---------------------------------------------------------------------------

class TestHashPhone:
    def test_deterministic(self) -> None:
        assert hash_phone("11999999999") == hash_phone("11999999999")

    def test_different_phones_differ(self) -> None:
        assert hash_phone("11999999999") != hash_phone("11888888888")

    def test_returns_sha256_hex(self) -> None:
        result = hash_phone("11999999999")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_normalizes_formatting(self) -> None:
        # Formatted and bare versions of the same number must hash identically
        assert hash_phone("11999999999") == hash_phone("(11) 99999-9999")

    def test_different_salts_differ(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from app.config import get_settings

        first = hash_phone("11999999999")

        monkeypatch.setenv("PII_SALT", "outro-salt-qualquer")
        get_settings.cache_clear()

        second = hash_phone("11999999999")
        assert first != second
