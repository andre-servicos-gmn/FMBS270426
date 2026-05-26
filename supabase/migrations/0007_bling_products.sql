-- Sprint 2.5 — Bling product catalog mirrored locally.
--
-- We mirror the subset of fields the agent actually consults. The Bling ID
-- is the primary key (BIGINT) so subsequent syncs are simple UPSERTs.
-- ``atributos_parseados`` carries best-effort regex-parsed values from the
-- HTML description; ``campos_customizados`` carries the raw structured
-- custom fields. Both are JSONB so future fields don't require migrations.

CREATE TABLE IF NOT EXISTS bling_products (
    id                       BIGINT       PRIMARY KEY,
    nome                     TEXT         NOT NULL,
    codigo                   TEXT,
    preco                    NUMERIC(10,2),
    descricao_curta          TEXT,
    descricao_complementar   TEXT,
    marca                    TEXT,
    modelo                   TEXT,
    categoria_id             BIGINT,
    categoria_nome           TEXT,
    peso_liquido             NUMERIC(10,3),
    peso_bruto               NUMERIC(10,3),
    largura                  NUMERIC(10,2),
    altura                   NUMERIC(10,2),
    profundidade             NUMERIC(10,2),
    gtin                     TEXT,
    situacao                 TEXT,
    is_raquete_praia         BOOLEAN      DEFAULT FALSE,
    campos_customizados      JSONB,
    atributos_parseados      JSONB,
    imagem_url               TEXT,
    last_synced_at           TIMESTAMPTZ  DEFAULT NOW(),
    created_at               TIMESTAMPTZ  DEFAULT NOW(),
    updated_at               TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_bling_products_categoria
    ON bling_products (categoria_nome);

CREATE INDEX IF NOT EXISTS idx_bling_products_raquete_praia
    ON bling_products (is_raquete_praia) WHERE is_raquete_praia = TRUE;

CREATE INDEX IF NOT EXISTS idx_bling_products_situacao
    ON bling_products (situacao);

CREATE INDEX IF NOT EXISTS idx_bling_products_marca_modelo
    ON bling_products (marca, modelo);

-- Webhook idempotency tracking: skip events whose ``event_timestamp``
-- is older than what we last applied for that product.
CREATE TABLE IF NOT EXISTS bling_webhook_events (
    id             SERIAL      PRIMARY KEY,
    product_id     BIGINT      NOT NULL,
    event_kind     TEXT        NOT NULL,
    event_timestamp TIMESTAMPTZ NOT NULL,
    received_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    payload        JSONB
);
CREATE INDEX IF NOT EXISTS idx_bling_webhook_events_product
    ON bling_webhook_events (product_id, event_timestamp DESC);
