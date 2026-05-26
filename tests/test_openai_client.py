"""Tests for OpenAIClient — PII masking, safe logging, dev-mode assertion."""
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.adapters.openai_client import OpenAIClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_response(content: str = '{"intent": "faq"}') -> MagicMock:
    """Build a minimal fake ChatCompletion response."""
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5

    choice = MagicMock()
    choice.message.content = content

    resp = MagicMock()
    resp.choices = [choice]
    resp.model = "gpt-4o-mini"
    resp.usage = usage
    return resp


def _make_raw_client(content: str = '{"intent": "faq"}') -> AsyncMock:
    raw = AsyncMock()
    raw.chat.completions.create.return_value = _make_mock_response(content)
    return raw


def _sent_messages(raw: AsyncMock) -> list[dict[str, str]]:
    """Return the messages list from the last create() call."""
    return raw.chat.completions.create.call_args.kwargs["messages"]


@pytest.fixture
def raw_client() -> AsyncMock:
    return _make_raw_client()


@pytest.fixture
def client(raw_client: AsyncMock) -> OpenAIClient:
    return OpenAIClient(client=raw_client)


# ---------------------------------------------------------------------------
# PII masking — message content
# ---------------------------------------------------------------------------


class TestPIIMaskingMessages:
    async def test_cpf_formatted_is_masked(
        self, client: OpenAIClient, raw_client: AsyncMock
    ) -> None:
        await client.chat(
            messages=[{"role": "user", "content": "Meu CPF é 123.456.789-00"}],
            system="sistema",
        )
        for msg in _sent_messages(raw_client):
            assert "123.456.789-00" not in msg["content"]

    async def test_cpf_replaced_with_token(
        self, client: OpenAIClient, raw_client: AsyncMock
    ) -> None:
        await client.chat(
            messages=[{"role": "user", "content": "CPF: 123.456.789-00"}],
            system="sistema",
        )
        user_msg = next(m for m in _sent_messages(raw_client) if m["role"] == "user")
        assert "[CPF]" in user_msg["content"]

    async def test_phone_in_message_is_masked(
        self, client: OpenAIClient, raw_client: AsyncMock
    ) -> None:
        await client.chat(
            messages=[{"role": "user", "content": "Me liga no (11) 99999-9999"}],
            system="sistema",
        )
        for msg in _sent_messages(raw_client):
            assert "(11) 99999-9999" not in msg["content"]

    async def test_email_in_message_is_masked(
        self, client: OpenAIClient, raw_client: AsyncMock
    ) -> None:
        await client.chat(
            messages=[{"role": "user", "content": "email: cliente@email.com"}],
            system="sistema",
        )
        for msg in _sent_messages(raw_client):
            assert "cliente@email.com" not in msg["content"]

    async def test_clean_message_passes_unchanged(
        self, client: OpenAIClient, raw_client: AsyncMock
    ) -> None:
        await client.chat(
            messages=[{"role": "user", "content": "Quero jogar padel amanhã"}],
            system="sistema",
        )
        user_msg = next(m for m in _sent_messages(raw_client) if m["role"] == "user")
        assert user_msg["content"] == "Quero jogar padel amanhã"

    async def test_multiple_messages_all_masked(
        self, client: OpenAIClient, raw_client: AsyncMock
    ) -> None:
        await client.chat(
            messages=[
                {"role": "user", "content": "Meu CPF é 111.222.333-44"},
                {"role": "assistant", "content": "Entendido."},
                {"role": "user", "content": "Meu telefone é (21) 98765-4321"},
            ],
            system="sistema",
        )
        sent = _sent_messages(raw_client)
        full_text = " ".join(m["content"] for m in sent)
        assert "111.222.333-44" not in full_text
        assert "(21) 98765-4321" not in full_text


# ---------------------------------------------------------------------------
# PII masking — system prompt
# ---------------------------------------------------------------------------


class TestPIIMaskingSystem:
    async def test_cpf_in_system_is_masked(
        self, client: OpenAIClient, raw_client: AsyncMock
    ) -> None:
        await client.chat(
            messages=[],
            system="Contexto admin: CPF 000.000.001-91",
        )
        system_msg = next(m for m in _sent_messages(raw_client) if m["role"] == "system")
        assert "000.000.001-91" not in system_msg["content"]
        assert "[CPF]" in system_msg["content"]

    async def test_clean_system_passes_unchanged(
        self, client: OpenAIClient, raw_client: AsyncMock
    ) -> None:
        await client.chat(messages=[], system="Você é um assistente de padel.")
        system_msg = next(m for m in _sent_messages(raw_client) if m["role"] == "system")
        assert system_msg["content"] == "Você é um assistente de padel."

    async def test_system_is_first_message(
        self, client: OpenAIClient, raw_client: AsyncMock
    ) -> None:
        await client.chat(
            messages=[{"role": "user", "content": "oi"}],
            system="meu sistema",
        )
        assert _sent_messages(raw_client)[0]["role"] == "system"


# ---------------------------------------------------------------------------
# Logging — content must never appear in logs
# ---------------------------------------------------------------------------


class TestLogging:
    async def test_user_content_not_logged(
        self, client: OpenAIClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        secret = "frase-secreta-que-nao-vai-pro-log-xkcd"
        with caplog.at_level(logging.DEBUG, logger="app.adapters.openai_client"):
            await client.chat(
                messages=[{"role": "user", "content": secret}],
                system="sistema",
            )
        assert secret not in caplog.text

    async def test_system_content_not_logged(
        self, client: OpenAIClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        secret_system = "prompt-secreto-que-nao-vai-pro-log-42"
        with caplog.at_level(logging.DEBUG, logger="app.adapters.openai_client"):
            await client.chat(messages=[], system=secret_system)
        assert secret_system not in caplog.text

    async def test_model_logged(
        self, client: OpenAIClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="app.adapters.openai_client"):
            await client.chat(messages=[], system="sistema")
        assert "gpt-4o-mini" in caplog.text

    async def test_token_counts_logged(
        self, client: OpenAIClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="app.adapters.openai_client"):
            await client.chat(messages=[], system="sistema")
        assert "tokens_in" in caplog.text
        assert "tokens_out" in caplog.text

    async def test_latency_logged(
        self, client: OpenAIClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="app.adapters.openai_client"):
            await client.chat(messages=[], system="sistema")
        assert "latency_ms" in caplog.text


# ---------------------------------------------------------------------------
# Dev-mode assertion (defense in depth)
# ---------------------------------------------------------------------------


class TestDevModeAssertion:
    async def test_raises_on_pii_leak_in_message(
        self,
        raw_client: AsyncMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Simulate a masking gap: is_clean always reports PII detected
        monkeypatch.setattr("app.adapters.openai_client.is_clean", lambda _: False)
        c = OpenAIClient(client=raw_client)
        with pytest.raises(ValueError, match="PII leak"):
            await c.chat(
                messages=[{"role": "user", "content": "olá"}],
                system="sistema",
            )

    async def test_raises_on_pii_leak_in_system(
        self,
        raw_client: AsyncMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        call_count = 0

        def selective_is_clean(text: str) -> bool:
            nonlocal call_count
            call_count += 1
            # First call is the system prompt — simulate leak there
            return call_count > 1

        monkeypatch.setattr("app.adapters.openai_client.is_clean", selective_is_clean)
        c = OpenAIClient(client=raw_client)
        with pytest.raises(ValueError, match="system"):
            await c.chat(messages=[], system="sistema")

    async def test_no_assertion_in_production(
        self,
        raw_client: AsyncMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # is_clean always False, but prod mode must skip the assertion
        monkeypatch.setattr("app.adapters.openai_client.is_clean", lambda _: False)
        prod = MagicMock()
        prod.app_env = "production"
        prod.openai_model = "gpt-4o-mini"
        prod.pii_mask_enabled = True
        monkeypatch.setattr("app.adapters.openai_client.get_settings", lambda: prod)

        c = OpenAIClient(client=raw_client)
        result = await c.chat(messages=[], system="sistema")
        assert isinstance(result, str)

    async def test_no_assertion_when_masking_disabled(
        self,
        raw_client: AsyncMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("app.adapters.openai_client.is_clean", lambda _: False)
        dev_no_mask = MagicMock()
        dev_no_mask.app_env = "development"
        dev_no_mask.openai_model = "gpt-4o-mini"
        dev_no_mask.pii_mask_enabled = False
        monkeypatch.setattr("app.adapters.openai_client.get_settings", lambda: dev_no_mask)

        c = OpenAIClient(client=raw_client)
        # pii_mask_enabled=False disables the assertion too
        result = await c.chat(messages=[], system="sistema")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# json_mode
# ---------------------------------------------------------------------------


class TestJSONMode:
    async def test_json_mode_sets_response_format(
        self, client: OpenAIClient, raw_client: AsyncMock
    ) -> None:
        await client.chat(messages=[], system="retorne json", json_mode=True)
        kwargs = raw_client.chat.completions.create.call_args.kwargs
        assert kwargs.get("response_format") == {"type": "json_object"}

    async def test_default_mode_no_response_format(
        self, client: OpenAIClient, raw_client: AsyncMock
    ) -> None:
        await client.chat(messages=[], system="sistema")
        kwargs = raw_client.chat.completions.create.call_args.kwargs
        assert "response_format" not in kwargs

    async def test_max_tokens_and_temperature_forwarded(
        self, client: OpenAIClient, raw_client: AsyncMock
    ) -> None:
        await client.chat(messages=[], system="sistema", max_tokens=512, temperature=0.1)
        kwargs = raw_client.chat.completions.create.call_args.kwargs
        assert kwargs["max_tokens"] == 512
        assert kwargs["temperature"] == 0.1


# ---------------------------------------------------------------------------
# Prompts — smoke test for guardrail presence
# ---------------------------------------------------------------------------


class TestPrompts:
    def test_all_prompts_contain_guardrail(self) -> None:
        from app.agent import prompts

        # Sprint 1.8 split SYSTEM_DIAGNOSE into three specialised prompts.
        prompt_names = (
            "SYSTEM_TRIAGE",
            "SYSTEM_DIAGNOSE_EXTRACT",
            "SYSTEM_DIAGNOSE_PHRASE",
            "SYSTEM_DIAGNOSE_META",
            "SYSTEM_RECOMMEND",
            "SYSTEM_FAQ",
        )
        for name in prompt_names:
            prompt = getattr(prompts, name)
            assert "CPF" in prompt, f"{name} missing CPF guardrail"
            assert "endereço" in prompt, f"{name} missing address guardrail"

    def test_triage_prompt_contains_json_word(self) -> None:
        from app.agent.prompts import SYSTEM_TRIAGE

        assert "json" in SYSTEM_TRIAGE.lower()

    def test_diagnose_extract_prompt_contains_json_word(self) -> None:
        """Sprint 1.8: the extraction step uses json_mode."""
        from app.agent.prompts import SYSTEM_DIAGNOSE_EXTRACT

        assert "json" in SYSTEM_DIAGNOSE_EXTRACT.lower()

    def test_faq_prompt_contains_handoff_marker(self) -> None:
        from app.agent.prompts import SYSTEM_FAQ

        assert "[HANDOFF]" in SYSTEM_FAQ

    def test_recommend_prompt_requests_json_blocks(self) -> None:
        """Sprint 1.6: recommend switched to json_mode with messages array."""
        from app.agent.prompts import SYSTEM_RECOMMEND

        s = SYSTEM_RECOMMEND.lower()
        # OpenAI requires the word "json" to appear in the system prompt when
        # response_format=json_object is set.
        assert "json" in s
        # The blocks contract must be advertised explicitly.
        assert '"messages"' in SYSTEM_RECOMMEND or "messages" in s
        assert "blocos" in s
