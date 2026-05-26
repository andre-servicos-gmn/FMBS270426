-- 0002_search_function.sql
-- Cosine-similarity product search with optional filters.

CREATE OR REPLACE FUNCTION search_products(
    query_embedding vector(1536),
    p_sport         VARCHAR  DEFAULT NULL,
    p_max_price     INTEGER  DEFAULT NULL,
    p_min_stock     INTEGER  DEFAULT 1,
    p_limit         INTEGER  DEFAULT 10
)
RETURNS TABLE (
    id          UUID,
    external_id VARCHAR(255),
    name        VARCHAR(500),
    sport       VARCHAR(100),
    level       VARCHAR(100),
    weight_g    INTEGER,
    balance     VARCHAR(100),
    material    VARCHAR(255),
    price_cents INTEGER,
    stock       INTEGER,
    description TEXT,
    url         VARCHAR(2048),
    image_url   VARCHAR(2048),
    updated_at  TIMESTAMPTZ,
    is_active   BOOLEAN,
    similarity  FLOAT
)
LANGUAGE plpgsql
STABLE
AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.id,
        p.external_id,
        p.name,
        p.sport,
        p.level,
        p.weight_g,
        p.balance,
        p.material,
        p.price_cents,
        p.stock,
        p.description,
        p.url,
        p.image_url,
        p.updated_at,
        p.is_active,
        (1 - (p.embedding <=> query_embedding))::FLOAT AS similarity
    FROM products p
    WHERE
        p.is_active = TRUE
        AND p.stock >= p_min_stock
        AND (p_sport IS NULL OR p.sport = p_sport)
        AND (p_max_price IS NULL OR p.price_cents <= p_max_price)
        AND p.embedding IS NOT NULL
    ORDER BY p.embedding <=> query_embedding
    LIMIT p_limit;
END;
$$;
