import logging
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI
    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="gpt-4o-mini")

    # Evolution API
    evolution_api_url: str = Field(default="")
    evolution_api_key: str = Field(default="")
    evolution_instance: str = Field(default="")
    evolution_webhook_token: str = Field(default="")

    # Admin
    admin_api_key: str = Field(default="")

    # Sprint 2.7 — RESET_ALLOWED_PHONES: lista separada por vírgula de números
    # autorizados a usar `/reset` no WhatsApp (formato internacional sem o +,
    # ex: "5511987654321,5511912345678"). Vazio → /reset DESATIVADO pra todo
    # mundo. Use em dev/staging; em produção mantenha vazio salvo necessidade.
    reset_allowed_phones: str = Field(default="")

    # Sprint 2.7.2 — debounce de mensagens rápidas do mesmo cliente.
    # Mensagens de TEXTO recebidas em ``message_debounce_ms`` umas das outras
    # são agrupadas em um único processamento (input concatenado por ". ").
    # Resolve o bug do Felipe: "Quero a Proteo" + "Vc tem?" 200ms depois →
    # 1 resposta coerente em vez de 2 (uma certa, uma genérica).
    # ``cap`` e ``hard_ttl`` são circuit-breakers defensivos: se o cliente
    # mandar muito ou ficar resetando o timer indefinidamente, força flush.
    message_debounce_ms: int = Field(default=1500)
    message_debounce_cap: int = Field(default=10)
    message_debounce_hard_ttl_ms: int = Field(default=8000)

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0")
    session_ttl_seconds: int = Field(default=86400)
    session_hard_cap_seconds: int = Field(default=604800)

    # Supabase
    supabase_url: str = Field(default="")
    supabase_service_role_key: str = Field(default="")

    # Postgres
    database_url: str = Field(default="postgresql+asyncpg://beachtenis:beachtenis@localhost:5432/beachtenis")

    # Catalog
    catalog_source: Literal["api", "file"] = Field(default="api")
    catalog_api_url: str = Field(default="")
    catalog_api_key: str = Field(default="")
    catalog_file_path: str = Field(default="")
    catalog_sync_cron: str = Field(default="0 */6 * * *")

    # Embeddings
    embedding_provider: Literal["voyage", "openai"] = Field(default="voyage")
    embedding_api_key: str = Field(default="")

    # Compliance
    pii_mask_enabled: bool = Field(default=True)
    pii_salt: str = Field(default="change-me-in-production")
    lead_retention_days: int = Field(default=365)

    # Sprint 2.2 — número que recebe o dossiê em handoff via WhatsApp (Evolution).
    # Formato: internacional sem o + (ex: "5511987654321"). Vazio → o dossiê só
    # é gravado no banco (fallback gracioso, sem envio externo). Em piloto o
    # destino é o WhatsApp do dono; em produção troca-se para o gerente da loja.
    dossier_recipient_phone: str = Field(default="")

    # Sprint 2.5 — Bling ERP OAuth 2.0 + sync. Vazio → integração desativada
    # (agent volta a usar o catálogo local seedado em ``products``).
    bling_client_id: str = Field(default="")
    bling_client_secret: str = Field(default="")
    bling_redirect_uri: str = Field(default="")
    # Lista de categorias relevantes (nomes EXATOS conforme cadastrados no
    # painel do Bling). Vazia → sync importa TUDO (não recomendado).
    bling_sync_categories: str = Field(default="")
    bling_sync_hour: int = Field(default=4)
    bling_stock_cache_ttl: int = Field(default=300)
    # Sprint 2.6.3 — in-memory catalog snapshot TTL. Default 60s so a fresh
    # webhook update propagates to the match layer within a minute without
    # forcing a Supabase round-trip per inbound WhatsApp message.
    bling_catalog_cache_ttl: int = Field(default=60)
    # Segredo HMAC do webhook. Vazio em dev → validação desligada (LOG WARNING).
    bling_webhook_secret: str = Field(default="")

    # Loja física — injetada no SYSTEM_CLOSE legado (build_close_prompt) E no
    # bloco de identidade da loja do supervisor v2. ENDEREÇO/HORÁRIO/NOME são
    # fatos fixos e únicos da Base Sports.
    #
    # Defaults VAZIOS de propósito (regra de segurança): um default fixture
    # alimentaria o legado e o v2 com um endereço FALSO sempre que o .env de
    # produção não preenchesse store_address — e mandar o cliente a um endereço
    # inventado é o dano que tratamos como bloqueio. Com vazio, o legado cai no
    # fallback genérico que já tinha, e o v2 (build_system_prompt) NÃO cita
    # endereço, pedindo ao cliente que confirme. O André preenche os valores
    # reais no .env / painel antes do deploy.
    store_name: str = Field(default="")
    store_address: str = Field(default="")
    store_hours: str = Field(default="")
    store_maps_url: str = Field(default="")
    store_phone: str = Field(default="")

    # E-commerce online da Base Sports (venda online com PIX). Default VAZIO
    # pela mesma regra de segurança do endereço: sem link, o agente menciona o
    # e-commerce mas NÃO inventa URL — pede pro cliente confirmar o link. O
    # André preenche o link real no .env antes do deploy.
    ecommerce_url: str = Field(default="")

    # Consultoria Base Esportes — produto pago de consultoria com teste em
    # quadra que o agente NÃO pode substituir. consultoria_enabled=False
    # remove a menção do recommend e desativa o node de pitch.
    consultoria_preco: int = Field(default=350)
    consultoria_enabled: bool = Field(default=True)

    # Fase 0 — supervisor V2 (grafo de tool-calling em paralelo ao grafo atual).
    # Default False: o webhook continua usando o grafo switch legado. Ligar SOMENTE
    # em dev/staging para smoke test do loop (tools ainda são stubs). Lê de USE_V2.
    use_v2: bool = Field(default=False)

    # App
    app_env: Literal["development", "staging", "production"] = Field(default="development")
    log_level: str = Field(default="INFO")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
