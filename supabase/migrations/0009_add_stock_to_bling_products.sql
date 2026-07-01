-- 0009_add_stock_to_bling_products.sql
-- Sprint 3.9 — espelha o saldo de estoque dentro de bling_products para que o
-- buscar_catalogo (tools_v2) consiga filtrar "estoque > 0" de forma barata, em
-- cima do snapshot em memória, sem chamar a API de estoque do Bling por produto.
--
-- A coluna é NULLABLE de propósito: NULL = "estoque desconhecido" (o sync ainda
-- não populou, ou a leitura de estoque falhou). O filtro trata NULL como
-- "manter" (nunca esconde produto por falta de dado); só esconde quando o
-- estoque é positivamente 0. O sync popula/atualiza o valor a cada full_sync.
--
-- Reverso: ver 0009_add_stock_to_bling_products_down.sql

ALTER TABLE bling_products
    ADD COLUMN IF NOT EXISTS stock INTEGER;
