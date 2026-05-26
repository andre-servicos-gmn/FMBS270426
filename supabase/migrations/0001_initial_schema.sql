-- 0001_initial_schema.sql
-- Apply via: supabase db push
-- Or paste directly in Supabase Dashboard → SQL Editor

-- pgvector (already available in Supabase, but idempotent)
CREATE EXTENSION IF NOT EXISTS vector;

-- ── leads ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS leads (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    phone_hash          VARCHAR(64) NOT NULL UNIQUE,
    profile             JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_interaction_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    retention_until     TIMESTAMPTZ,
    deleted_at          TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_leads_phone_hash ON leads (phone_hash);

-- ── products ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS products (
    id          UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id VARCHAR(255) NOT NULL UNIQUE,
    name        VARCHAR(500) NOT NULL,
    sport       VARCHAR(100),
    level       VARCHAR(100),
    weight_g    INTEGER,
    balance     VARCHAR(100),
    material    VARCHAR(255),
    price_cents INTEGER      NOT NULL DEFAULT 0,
    stock       INTEGER      NOT NULL DEFAULT 0,
    description TEXT,
    url         VARCHAR(2048),
    image_url   VARCHAR(2048),
    embedding   vector(1536),
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    is_active   BOOLEAN      NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_products_external_id ON products (external_id);

-- IVFFlat approximate nearest-neighbor (cosine); useful once table has ≥100 rows
CREATE INDEX IF NOT EXISTS idx_products_embedding ON products
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ── access_logs ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS access_logs (
    id          UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    actor       VARCHAR(255) NOT NULL,
    action      VARCHAR(100) NOT NULL,
    target_hash VARCHAR(64)  NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    ip          VARCHAR(45),
    metadata    JSONB
);

CREATE INDEX IF NOT EXISTS idx_access_logs_target_hash ON access_logs (target_hash);

-- ── conversation_logs ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversation_logs (
    id             UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    phone_hash     VARCHAR(64) NOT NULL,
    message_role   VARCHAR(20) NOT NULL,
    content_masked TEXT        NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_conversation_logs_phone_hash ON conversation_logs (phone_hash);

-- ── Row Level Security ────────────────────────────────────────────────────────
-- service_role bypasses RLS automatically in Supabase.
-- These policies are explicit safeguards; anon/authenticated roles are denied by default.

ALTER TABLE leads             ENABLE ROW LEVEL SECURITY;
ALTER TABLE products          ENABLE ROW LEVEL SECURITY;
ALTER TABLE access_logs       ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversation_logs ENABLE ROW LEVEL SECURITY;

CREATE POLICY service_role_leads ON leads
    FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY service_role_products ON products
    FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY service_role_access_logs ON access_logs
    FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY service_role_conversation_logs ON conversation_logs
    FOR ALL TO service_role USING (true) WITH CHECK (true);
