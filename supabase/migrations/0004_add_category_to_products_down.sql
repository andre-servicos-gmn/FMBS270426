-- 0004_add_category_to_products_down.sql
-- Reverte 0004_add_category_to_products.sql.
-- Drop é seguro porque a coluna foi adicionada com IF NOT EXISTS no UP e o
-- DROP usa IF EXISTS aqui — pode ser executado várias vezes sem erro.

DROP INDEX IF EXISTS idx_products_category;
ALTER TABLE products DROP COLUMN IF EXISTS category;
