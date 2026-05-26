import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, Boolean, DateTime, Integer, JSON, Numeric, String, Text, Uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Lead(Base):
    __tablename__ = "leads"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    phone_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    profile: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now)
    last_interaction_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, onupdate=_now
    )
    retention_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Product(Base):
    __tablename__ = "products"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    external_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    sport: Mapped[str | None] = mapped_column(String(100), nullable=True)
    level: Mapped[str | None] = mapped_column(String(100), nullable=True)
    weight_g: Mapped[int | None] = mapped_column(Integer, nullable=True)
    balance: Mapped[str | None] = mapped_column(String(100), nullable=True)
    material: Mapped[str | None] = mapped_column(String(255), nullable=True)
    price_cents: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    stock: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    image_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, onupdate=_now
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")


class AccessLog(Base):
    __tablename__ = "access_logs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    actor: Mapped[str] = mapped_column(String(255), nullable=False)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    target_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now)
    ip: Mapped[str | None] = mapped_column(String(45), nullable=True)
    # Column name "metadata" reserved in SQLAlchemy Base; accessed as metadata_ in Python.
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON, nullable=True)


class ConversationLog(Base):
    __tablename__ = "conversation_logs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    phone_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # user | assistant | system
    message_role: Mapped[str] = mapped_column(String(20), nullable=False)
    # Content stored only after PII masking — never raw user input
    content_masked: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now)


class BlingCredentials(Base):
    """Sprint 2.5 — singleton row holding the Bling OAuth tokens."""
    __tablename__ = "bling_credentials"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    access_token: Mapped[str] = mapped_column(Text, nullable=False)
    refresh_token: Mapped[str] = mapped_column(Text, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    scope: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, onupdate=_now
    )


class BlingProduct(Base):
    """Sprint 2.5 — mirrored Bling product catalog."""
    __tablename__ = "bling_products"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    nome: Mapped[str] = mapped_column(Text, nullable=False)
    codigo: Mapped[str | None] = mapped_column(Text, nullable=True)
    preco: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    descricao_curta: Mapped[str | None] = mapped_column(Text, nullable=True)
    descricao_complementar: Mapped[str | None] = mapped_column(Text, nullable=True)
    marca: Mapped[str | None] = mapped_column(Text, nullable=True)
    modelo: Mapped[str | None] = mapped_column(Text, nullable=True)
    categoria_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    categoria_nome: Mapped[str | None] = mapped_column(Text, nullable=True)
    peso_liquido: Mapped[Decimal | None] = mapped_column(Numeric(10, 3), nullable=True)
    peso_bruto: Mapped[Decimal | None] = mapped_column(Numeric(10, 3), nullable=True)
    largura: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    altura: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    profundidade: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    gtin: Mapped[str | None] = mapped_column(Text, nullable=True)
    situacao: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_raquete_praia: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    campos_customizados: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    atributos_parseados: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    imagem_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_synced_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, onupdate=_now
    )


class BlingSyncLog(Base):
    __tablename__ = "bling_sync_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    total_processed: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    inserted: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    updated: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    skipped: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    errors: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON, nullable=True)


class BlingWebhookEvent(Base):
    __tablename__ = "bling_webhook_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    event_kind: Mapped[str] = mapped_column(Text, nullable=False)
    event_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # faq | shipping | exchange | warranty | payment | store | general
    category: Mapped[str] = mapped_column(String(100), nullable=False, server_default="general")
    # manual | api_sync | upload
    source: Mapped[str] = mapped_column(String(100), nullable=False, server_default="manual")
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, onupdate=_now
    )
