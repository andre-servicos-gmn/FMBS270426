-- Sprint 2.5 — Bling OAuth 2.0 credentials (singleton row).
--
-- The Nouvaris virtual attendant uses a SINGLE Bling app authorization
-- shared by the entire deployment (the agent is the OAuth client). We
-- enforce singleton with a unique index on a constant boolean expression
-- so a second insert fails at the DB layer instead of relying on app code.

CREATE TABLE IF NOT EXISTS bling_credentials (
    id            SERIAL      PRIMARY KEY,
    access_token  TEXT        NOT NULL,
    refresh_token TEXT        NOT NULL,
    expires_at    TIMESTAMPTZ NOT NULL,
    scope         TEXT,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_bling_creds_singleton
    ON bling_credentials ((TRUE));

-- Audit log of sync runs — one row per full_sync (or per webhook batch).
CREATE TABLE IF NOT EXISTS bling_sync_logs (
    id              SERIAL      PRIMARY KEY,
    kind            TEXT        NOT NULL CHECK (kind IN ('full', 'webhook', 'manual')),
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at     TIMESTAMPTZ,
    total_processed INTEGER     DEFAULT 0,
    inserted        INTEGER     DEFAULT 0,
    updated         INTEGER     DEFAULT 0,
    skipped         INTEGER     DEFAULT 0,
    errors          INTEGER     DEFAULT 0,
    error_message   TEXT,
    metadata        JSONB
);
CREATE INDEX IF NOT EXISTS idx_bling_sync_logs_started ON bling_sync_logs (started_at DESC);
