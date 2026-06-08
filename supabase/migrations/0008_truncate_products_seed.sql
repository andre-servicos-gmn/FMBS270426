-- Sprint 2.6 — empty the legacy ``products`` seed catalog.
--
-- Pre-Bling builds used ``products`` (seeded via scripts/seed_via_rest.py)
-- as the agent's source of truth. With Bling V3 wired (Sprint 2.5+) the
-- agent reads from ``bling_products`` instead. We TRUNCATE ``products``
-- so there isn't a conflicting second catalog floating around in
-- production — and so an accidental fallback in dev doesn't leak stale
-- product names into the agent.
--
-- We don't DROP the table: developers running the agent OFFLINE (no
-- BLING_CLIENT_ID) can still seed it for local tests. ``RESTART IDENTITY``
-- resets any sequences associated with the table. ``CASCADE`` cleans up
-- the embedding-related references safely.

TRUNCATE TABLE products RESTART IDENTITY CASCADE;
