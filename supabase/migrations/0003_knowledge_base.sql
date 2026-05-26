-- 0003_knowledge_base.sql
-- Base de conhecimento vetorial para RAG do nó FAQ do agente.
-- Apply via: supabase db push  OR  paste in Supabase Dashboard → SQL Editor

-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE: knowledge_base
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS knowledge_base (
    id          UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    title       VARCHAR(500) NOT NULL,
    content     TEXT         NOT NULL,
    category    VARCHAR(100) NOT NULL DEFAULT 'general',
    source      VARCHAR(100) NOT NULL DEFAULT 'manual',
    embedding   vector(1536),
    is_active   BOOLEAN      NOT NULL DEFAULT TRUE,
    metadata    JSONB,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),

    CONSTRAINT kb_title_category_unique UNIQUE (title, category)
);

CREATE INDEX IF NOT EXISTS idx_kb_embedding_hnsw
    ON knowledge_base
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_kb_category_active
    ON knowledge_base (category, is_active)
    WHERE is_active = TRUE;

CREATE INDEX IF NOT EXISTS idx_kb_category
    ON knowledge_base (category);

CREATE TRIGGER kb_moddatetime
    BEFORE UPDATE ON knowledge_base
    FOR EACH ROW
    EXECUTE PROCEDURE moddatetime(updated_at);

-- ─────────────────────────────────────────────────────────────────────────────
-- RLS
-- ─────────────────────────────────────────────────────────────────────────────
ALTER TABLE knowledge_base ENABLE ROW LEVEL SECURITY;

CREATE POLICY "service_role_kb_all"
    ON knowledge_base FOR ALL TO service_role
    USING (true) WITH CHECK (true);

-- ─────────────────────────────────────────────────────────────────────────────
-- FUNCTION: search_knowledge_base
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION search_knowledge_base(
    query_embedding  vector(1536),
    p_category       VARCHAR  DEFAULT NULL,
    p_limit          INTEGER  DEFAULT 5
)
RETURNS TABLE (
    id          UUID,
    title       VARCHAR(500),
    content     TEXT,
    category    VARCHAR(100),
    source      VARCHAR(100),
    metadata    JSONB,
    updated_at  TIMESTAMPTZ,
    similarity  FLOAT
)
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    RETURN QUERY
    SELECT
        kb.id,
        kb.title,
        kb.content,
        kb.category,
        kb.source,
        kb.metadata,
        kb.updated_at,
        (1 - (kb.embedding <=> query_embedding))::FLOAT AS similarity
    FROM knowledge_base kb
    WHERE
        kb.is_active   = TRUE
        AND kb.embedding IS NOT NULL
        AND (p_category IS NULL OR kb.category = p_category)
    ORDER BY kb.embedding <=> query_embedding
    LIMIT p_limit;
END;
$$;

GRANT EXECUTE ON FUNCTION search_knowledge_base TO service_role;
REVOKE EXECUTE ON FUNCTION search_knowledge_base FROM anon, authenticated;
