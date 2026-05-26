-- 0004_add_category_to_products.sql
-- Sprint 1.11 — adiantamento parcial da Sprint 2 do ESCOPO.md (catálogo flexível).
-- Adiciona coluna `category` com CHECK constraint, índice, e backfill dos
-- produtos já seedados. NÃO inclui o campo `attributes JSONB` previsto na
-- Sprint 2 completa — esse fica pendente.
--
-- Reverso: ver 0004_add_category_to_products_down.sql

-- 1. ALTER TABLE: nova coluna com default seguro para qualquer linha existente.
ALTER TABLE products
    ADD COLUMN IF NOT EXISTS category TEXT NOT NULL DEFAULT 'outros'
    CHECK (category IN (
        'raquete',      -- raquetes de beach tennis
        'pala',         -- palas de padel
        'bola',         -- bolas, kits de bolas
        'acessorio',    -- overgrip, fita, protetor
        'vestuario',    -- roupas
        'calcado',      -- tênis
        'bolsa',        -- mochilas, raqueteiras
        'outros'        -- fallback
    ));

-- 2. Índice para filtros frequentes (recommend / re_recommendation).
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);

-- 3. Backfill dos 20 produtos seedados pelo scripts/seed_via_rest.py.
--    Mapeamento por external_id (estável) para evitar drift com renomeações.
UPDATE products SET category = 'raquete' WHERE external_id IN (
    'BT001', 'BT002', 'BT003', 'BT004', 'BT005', 'BT006', 'BT007', 'BT008'
);
UPDATE products SET category = 'bolsa' WHERE external_id IN (
    'BT009', 'PD010'
);
UPDATE products SET category = 'bola' WHERE external_id IN (
    'BT010'
);
UPDATE products SET category = 'pala' WHERE external_id IN (
    'PD001', 'PD002', 'PD003', 'PD004', 'PD005',
    'PD006', 'PD007', 'PD008', 'PD009'
);
