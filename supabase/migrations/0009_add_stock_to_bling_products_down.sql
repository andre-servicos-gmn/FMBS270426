-- 0009_add_stock_to_bling_products_down.sql
-- Reverte 0009_add_stock_to_bling_products.sql.
-- DROP com IF EXISTS — idempotente, pode rodar mais de uma vez sem erro.

ALTER TABLE bling_products DROP COLUMN IF EXISTS stock;
