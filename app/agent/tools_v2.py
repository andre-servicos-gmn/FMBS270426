"""V2 supervisor tools — Phase 1 REAL implementations (behind ``use_v2``, OFF).

Each tool reuses the data layer the legacy flow already owns; none of them
reimplements that layer:

    buscar_catalogo     → app.sync.bling_catalog_cache.get_catalog_snapshot
                          (+ app.sync.bling_repo.fetch_product_by_name fallback)
    detalhes_produto    → app.sync.bling_repo.fetch_product_by_id
    consultar_estoque   → app.sync.bling_stock.get_stock
    buscar_conhecimento → app.rag.retriever.search_knowledge_base (pgvector)
    escalar_humano      → app.agent.dossier.handoff_dossier_pipeline

IMPORTANT — this does NOT touch ``_product_match.py`` and does NOT use its
decision machinery (candidates / confirmation / gates). Disambiguation is now
the LLM's job; these tools only do a SIMPLE search over the data. No
``rapidfuzz`` in the project, so name matching is plain substring/token
overlap — no fuzzy distance.

The docstrings matter: the LLM reads them to decide when to call each tool.
"""
from __future__ import annotations

import difflib
import json
import logging
import re
import unicodedata
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from app.config import get_settings

logger = logging.getLogger(__name__)

# Phonetic-fuzzy match threshold. Calibrated against the real catalog:
#   - real typos:  "cronus"->"kronos" = 0.83, "protheu"->"proteo" = 1.00
#   - noise:       "baran"->"branco"/"branca"/"bassan" = 0.80 (transposition —
#                  difflib's ratio is generous with scrambled letters)
# 0.82 separates the real typos (>=0.83) from transposition noise (<=0.80),
# which fixes the production "Baran -> Heroes Sofia Chow (branca)" false match.
# NO rapidfuzz (not installed) and NO pg_trgm (trigram scored cronus/kronos at
# 0.077 — useless for c/k swaps); stdlib difflib over phonetically-folded
# tokens does the job. When no token clears the bar, buscar_catalogo returns an
# empty list instead of dumping an irrelevant product.
_FUZZY_THRESHOLD = 0.82


# ── helpers (simple search, NO _product_match decision machinery) ────────────

def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def _norm(s: str) -> str:
    return _strip_accents((s or "").lower()).strip()


# Generic words that carry no disambiguating signal — excluded from matching so
# "raquete"/"beach"/"tennis" don't make every product look relevant.
_STOPWORDS = {
    "raquete", "raquetes", "beach", "tennis", "padel", "praia", "the", "com",
    "para", "uma", "uns", "and", "and", "sport", "sports",
}


def _tokens(s: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", _norm(s)) if len(t) >= 3]


def _content_tokens(s: str) -> list[str]:
    return [t for t in _tokens(s) if t not in _STOPWORDS]


def _phonetic(tok: str) -> str:
    """Light PT-BR phonetic folding so common mis-spellings collapse:
    k<->c, ph->f, th->t, y->i, qu->c, collapse doubles, drop trailing vowels
    ("proteo"/"proteu" -> "prot", "kronos"/"cronus" -> "cronus"->"cronu"...).
    """
    t = tok
    t = t.replace("ph", "f").replace("th", "t").replace("qu", "c")
    t = t.replace("y", "i").replace("k", "c")
    t = re.sub(r"(.)\1+", r"\1", t)
    t = re.sub(r"[aeiou]+$", "", t)
    return t


def _price_brl(price_cents: Any) -> str:
    try:
        cents = int(price_cents or 0)
    except (TypeError, ValueError):
        return "R$ —"
    reais = cents / 100.0
    return "R$ " + f"{reais:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _score_product(q_tokens: list[str], product: dict[str, Any]) -> float:
    """Relevance score combining exact-token overlap with phonetic-fuzzy match.

    For each distinctive query token, we take the best score against any
    product token: an exact token match counts as 1.0; otherwise we fall back
    to a difflib ratio over the phonetically-folded forms (so "cronus" matches
    "kronos"). The product score is the SUM of per-query-token bests, so a
    multi-token query that matches more tokens ranks higher. A token whose best
    fuzzy score is below threshold contributes 0 (no spurious matches).
    """
    haystack = " ".join(
        str(product.get(f) or "")
        for f in ("name", "marca", "modelo", "categoria_nome")
    )
    p_tokens = _content_tokens(haystack)
    if not p_tokens:
        return 0.0
    p_set = set(p_tokens)
    p_phon = {pt: _phonetic(pt) for pt in p_tokens}

    total = 0.0
    for qt in q_tokens:
        if qt in p_set:
            # Exact token match dominates: a real "proteo"=="proteo" must
            # outrank a product that only matches "proteo" phonetically (e.g.
            # "prote" from "proteção"). Weighting exact >> fuzzy guarantees
            # the genuine product ranks above incidental phonetic collisions.
            total += 10.0
            continue
        qp = _phonetic(qt)
        best = 0.0
        for pt in p_tokens:
            r = difflib.SequenceMatcher(None, qp, p_phon[pt]).ratio()
            if r > best:
                best = r
        if best >= _FUZZY_THRESHOLD:
            total += best
    return total


# ── stock + recency (Sprint 3.9) ─────────────────────────────────────────────

def _stock_filter_enabled() -> bool:
    """Whether buscar_catalogo hides out-of-stock products. ON by default now
    that ``stock`` is mirrored into bling_products (filter over the snapshot,
    no live stock call). Toggle off with TOOLS_V2_FILTER_STOCK=false."""
    try:
        return bool(get_settings().tools_v2_filter_stock)
    except Exception:  # noqa: BLE001 — never let config break the catalog tool
        return True


def _has_stock(product: dict[str, Any]) -> bool:
    """True unless the product is POSITIVELY known to be out of stock.

    ``stock`` is the mirrored on-hand balance. None/absent means "unknown"
    (sync hasn't populated it yet, or a test fake without the field) and is
    KEPT — we never hide a product on missing data. Only a parseable stock
    that is <= 0 is dropped.
    """
    stock = product.get("stock")
    if stock is None:
        return True
    try:
        return int(stock) > 0
    except (TypeError, ValueError):
        return True


def _stock_status(product: dict[str, Any]) -> str | None:
    """Mirrored-stock status for the tool output: "disponivel" / "esgotado",
    or None when the mirror doesn't know (field absent/unparseable) — in that
    case the field is omitted so the LLM falls back to consultar_estoque."""
    stock = product.get("stock")
    if stock is None:
        return None
    try:
        return "disponivel" if int(stock) > 0 else "esgotado"
    except (TypeError, ValueError):
        return None


def _created_at_key(product: dict[str, Any]) -> str:
    """Recency sort key (newest first when sorted reverse=True). Falls back to
    last_synced_at then ''. Empty when the field is absent (e.g. unit-test
    fakes) → a stable no-op sort, never raises."""
    return str(product.get("created_at") or product.get("last_synced_at") or "")


# ── tools ────────────────────────────────────────────────────────────────────

# Top-N when filtering by a price range (more than the name-only top-5, but
# still bounded so the LLM context doesn't blow up).
_PRICE_RANGE_TOP_N = 8


def _spread_by_price(products: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Pick ``n`` products SPREAD across the price range, not the n cheapest.

    Production bug: a "raquetes até 2 mil" query returned the 8 CHEAPEST
    (R$449-469, all the same line), so the LLM only ever saw the bottom of the
    range and the answer read like a dump of near-identical prices. The customer
    asked for a ceiling, not for "the cheapest" — they want to see the SPREAD
    (one cheap, one mid, one near the ceiling) so they can place themselves.

    Given a price-ASCENDING list, sample ``n`` items at evenly spaced indices
    (``round(i*(len-1)/(n-1))``) so the result always includes the cheapest AND
    the most expensive within the cap, with mid points between. Dedupes indices
    (small lists can collapse two picks onto one index) and preserves price order.

    For the cheapest-first intent ("as mais baratas") we do NOT call this — that
    branch wants a true ascending head and is handled by the caller.
    """
    if n <= 0:
        return []
    if len(products) <= n:
        return list(products)
    last = len(products) - 1
    seen: set[int] = set()
    picked: list[dict[str, Any]] = []
    for i in range(n):
        idx = round(i * last / (n - 1))
        if idx not in seen:
            seen.add(idx)
            picked.append(products[idx])
    return picked

# Tokens that signal the customer is asking specifically for a racket, so a
# price-range query without a distinctive name still restricts to rackets.
_RACKET_HINT_TOKENS = {"raquete", "raquetes", "raqueta"}


def _price_reais(product: dict[str, Any]) -> float | None:
    """Reais price, or None when the product has no usable price.

    A price of 0 is treated as "no price" — in the synced catalog ~33 active
    products have ``preco`` null/0; surfacing them as "free" would poison a
    ``preco_asc`` ordering (they'd float to the top) and a price filter (they'd
    look cheaper than everything). So 0 → None → excluded from price queries.
    """
    cents = product.get("price_cents")
    if cents is None:
        return None
    try:
        reais = int(cents) / 100.0
    except (TypeError, ValueError):
        return None
    return reais if reais > 0 else None


# Category routing. The synced ``categoria_nome`` is free-text Bling noise (56
# distinct values, broken accents, and — critically — THREE racket categories
# that are NOT beach tennis: "Raquete de pickleball", "RAQUETE TENIS",
# "RAQUETE PADEL"). So "raquete de beach tennis" can NOT be matched by the
# word "raquete" in the category text. The reliable signal is the curated
# boolean ``is_raquete_praia`` (set in the sync from a Bling custom field +
# exact category match). Beach-tennis-racket queries route through it; other
# categories fall back to substring on the category text.
_BEACH_TENNIS_ALIASES = {
    "beach tennis", "beach", "praia", "raquete de praia", "raquete de beach",
    "raquetes de praia", "bt",
}


def _wants_beach_tennis(categoria: str | None) -> bool:
    if not categoria:
        return False
    norm = _norm(categoria)
    if norm in {_norm(a) for a in _BEACH_TENNIS_ALIASES}:
        return True
    # "raquete de beach tennis", "raquete beach", etc. — beach/praia present
    # and NOT another racket sport (padel/tenis/pickleball).
    has_beach = "beach" in norm or "praia" in norm
    other_sport = any(w in norm for w in ("padel", "tenis", "pickleball", "pickle"))
    return has_beach and not other_sport


# Name tokens that mean "this is NOT a single beach-tennis racket" even when
# the curated flag says raquete_praia. One real catalog entry — a "Frescobol
# Kit Tênis Praia 2 Raquetes" at R$169 — is mis-flagged is_raquete_praia=True
# and would otherwise lead a "mais barata" list as a R$169 racket. Frescobol is
# a different sport and a "kit" is not a single racket. This is a SYNC DATA
# GAP (the Bling custom field is wrong); guarding here keeps UX correct until
# the flag is fixed upstream.
_NOT_A_RACKET_NAME_TOKENS = {"frescobol"}


def _is_beach_tennis_racket(product: dict[str, Any]) -> bool:
    """Curated beach-tennis-racket flag, minus the known mis-flagged noise."""
    if not product.get("is_raquete_praia"):
        return False
    name_tokens = set(_tokens(str(product.get("name") or "")))
    return not (name_tokens & _NOT_A_RACKET_NAME_TOKENS)


def _is_racket(product: dict[str, Any]) -> bool:
    if _is_beach_tennis_racket(product):
        return True
    # Token-level match so "raqueteira" (bag) / "raqueteira" don't count as a
    # racket — "raquete"/"raquetes" must appear as a whole word, and the
    # category is the strongest signal.
    name_tokens = set(_tokens(str(product.get("name") or "")))
    # A "Frescobol Kit ... 2 Raquetes" is not a single racket the customer buys.
    if name_tokens & _NOT_A_RACKET_NAME_TOKENS:
        return False
    cat_tokens = set(_tokens(str(product.get("categoria_nome") or "")))
    racket_words = {"raquete", "raquetes"}
    return bool((cat_tokens | name_tokens) & racket_words)


def _matches_category(product: dict[str, Any], categoria: str) -> bool:
    """True when the product belongs to the requested category.

    Beach-tennis rackets route through the curated ``is_raquete_praia`` flag —
    NEVER the free-text category — so bags ("RAQUETEIRAS MOCHILA"), glasses,
    anti-vibrators, and tennis/pickleball/padel rackets are excluded even
    though their name or category text may contain "raquete"/"beach". Any other
    category falls back to an accent-insensitive substring on the category text.
    """
    if _wants_beach_tennis(categoria):
        return _is_beach_tennis_racket(product)
    cat_norm = _norm(str(product.get("categoria_nome") or ""))
    return _norm(categoria) in cat_norm if cat_norm else False


@tool
async def buscar_catalogo(
    consulta: str = "",
    preco_min: float | None = None,
    preco_max: float | None = None,
    categoria: str | None = None,
    ordenacao: str | None = None,
) -> str:
    """Busca produtos no catálogo da Base Sports por nome, marca, categoria ou faixa de preço.
    Use sempre que o cliente mencionar um produto, pedir comparação, ou perguntar o que existe.
    Para comparar dois produtos, chame uma vez para cada. Retorna lista com id, nome, preço e,
    quando conhecido, "estoque" ("disponivel" ou "esgotado" — espelho local, pode estar
    defasado; para confirmar disponibilidade ao vivo use consultar_estoque com o id).

    consulta: nome/marca do produto. Pode ser vazio quando a busca é só por
    categoria e/ou preço (ex: "as mais baratas de beach tennis" não tem nome).

    preco_min/preco_max (em reais): faixa de preço. Ex: "até 2k" → preco_max=2000;
    "entre 1000 e 1500" → preco_min=1000, preco_max=1500; "abaixo de mil" →
    preco_max=1000.

    categoria: filtra por tipo de produto. Use "beach tennis" para raquete de
    beach tennis, "padel" para padel, "mochila"/"raqueteira" para bolsa, etc.
    NÃO misture: "raquete de beach tennis" filtra só beach tennis, sem mochila
    nem acessório.

    ordenacao: passe "preco_asc" quando o cliente pedir "as mais baratas",
    "a mais barata" ou "mais em conta" — ordena do mais barato pro mais caro.

    Combine os três: "raquete de beach tennis até 1000" → categoria="beach tennis",
    preco_max=1000. "as mais baratas de beach tennis" → categoria="beach tennis",
    ordenacao="preco_asc" (sem nome, é só categoria + ordem).

    SEMPRE chame esta ferramenta com preco_max E categoria antes de dizer que
    não há produto numa faixa de preço."""
    q_tokens = _content_tokens(consulta)
    raw_tokens = set(_tokens(consulta))
    wants_racket = bool(raw_tokens & _RACKET_HINT_TOKENS)
    has_price_filter = preco_min is not None or preco_max is not None
    has_category = bool(categoria and categoria.strip())
    sort_by_price = (ordenacao or "").strip().lower() == "preco_asc"
    # Distinctive name tokens = the customer named a specific product (beyond the
    # generic "raquete"/"beach"/"tennis" hint words, which are stopwords).
    distinctive = [t for t in q_tokens if t not in _RACKET_HINT_TOKENS]
    # A "raquete" query with NO distinctive name and NO price/sort is a BROWSE
    # of beach tennis rackets, not a name match. Production bug: gpt-4o-mini
    # called buscar_catalogo(consulta="raquetes de beach tennis") with no
    # preco/categoria; the name score was 0 (all stopwords) → results=0 → the
    # agent said "não temos". Treat it as a beach tennis browse instead.
    racket_browse = wants_racket and not distinctive and not has_category
    # Price ordering kicks in for a price filter, an explicit preco_asc, a
    # category browse, OR a bare racket browse (reads best cheapest-first).
    price_ordered = has_price_filter or sort_by_price or has_category or racket_browse

    products: list[dict[str, Any]] = []
    # Primary: full in-memory snapshot (same path the legacy recommend uses).
    try:
        from app.sync.bling_catalog_cache import get_catalog_snapshot
        products = list(await get_catalog_snapshot())
    except Exception as exc:
        logger.warning("buscar_catalogo snapshot_failed: %s", exc)

    # Fallback: direct ILIKE on the longest query token if snapshot empty.
    if not products and q_tokens:
        try:
            from app.sync.bling_repo import fetch_product_by_name
            longest = max(q_tokens, key=len)
            products = await fetch_product_by_name(longest)
        except Exception as exc:
            logger.warning("buscar_catalogo ilike_failed: %s", exc)

    if not products:
        return json.dumps({"resultados": [], "aviso": "catalogo indisponivel"}, ensure_ascii=False)

    # Sprint 3.9 — drop out-of-stock products up front (cheap: stock is mirrored
    # into the snapshot). stock None/absent = unknown = kept; only positive 0 is
    # dropped. Done before category/price/ranking so it never leaks downstream.
    #
    # Sprint 3.12 — the dropped products are KEPT ASIDE: when the customer NAMES
    # a product (distinctive tokens, no price ask), an out-of-stock match must be
    # VISIBLE (marked "esgotado") or the agent can never say "está sem estoque" —
    # in production it substituted a sibling product of the same brand instead.
    # Browse/price/offer lists keep excluding out-of-stock entirely.
    out_of_stock: list[dict[str, Any]] = []
    if _stock_filter_enabled():
        out_of_stock = [p for p in products if not _has_stock(p)]
        products = [p for p in products if _has_stock(p)]
    named_lookup = bool(distinctive) and not has_price_filter and not sort_by_price

    # ── Category filter ───────────────────────────────────────────────────────
    # Explicit categoria param wins. Otherwise, a price/sort query defaults to
    # BEACH TENNIS rackets — this is a beach tennis/padel store, so a bare
    # "raquete até 1000" (or even a price query with no name at all) means a
    # beach tennis racket, never a tennis/pickleball racket, a bag, or an
    # anti-vibrator. We do NOT trust the LLM to pass categoria (in production it
    # often omits it, and gpt-4o-mini then let a "Raquete Tenis Tecnifibre"
    # leak into "raquete até 1k"). The default is enforced here, in code.
    if has_category:
        products = [p for p in products if _matches_category(p, categoria)]
    elif (has_price_filter or sort_by_price or racket_browse) and not distinctive:
        # Price/sort/racket-browse intent with no explicit category and no
        # distinctive name → pin to beach tennis rackets so the result is the
        # real catalog, never junk and never an empty name-match.
        products = [p for p in products if _is_beach_tennis_racket(p)]

    # ── Price filter ─────────────────────────────────────────────────────────
    if has_price_filter:
        def _in_range(p: dict[str, Any]) -> bool:
            price = _price_reais(p)
            if price is None:  # null/0 price → not a real in-range product
                return False
            if preco_min is not None and price < preco_min:
                return False
            if preco_max is not None and price > preco_max:
                return False
            return True

        products = [p for p in products if _in_range(p)]
    elif price_ordered:
        # Cheapest-first ordering only makes sense over priced products — drop
        # the null/0-price entries so they don't head the list as "free".
        products = [p for p in products if _price_reais(p) is not None]

    # ── Ranking ──────────────────────────────────────────────────────────────
    if price_ordered:
        # Price/category/racket browse → order by price ascending, bounded top-N.
        # When there ARE distinctive name tokens, a name match floats up first
        # (score desc) and price breaks ties; otherwise it's pure price order —
        # so "as mais baratas de beach tennis" / a bare "raquetes" browse work
        # WITHOUT a name to match (stopword-only tokens score 0, so we skip the
        # scoring and sort purely by price).
        if distinctive:
            scored = [(p, _score_product(q_tokens, p)) for p in products]
            scored.sort(key=lambda ps: (-ps[1], _price_reais(ps[0]) or float("inf")))
            if has_price_filter and not sort_by_price:
                # Brand/model query WITH a price ceiling (e.g. "Drop Shot até 2
                # mil"). The name score ties across a whole line — every "Drop
                # Shot" racket scores the same — so the price tie-break alone
                # returns the N CHEAPEST (R$449-469), the exact bug the no-name
                # branch already fixes with _spread_by_price. Keep only the
                # NAME-RELEVANT products (score > 0) so we don't spread into
                # unrelated brands, then SPREAD those across the price range so
                # the customer sees cheap..mid..near-ceiling, not just the floor.
                relevant = [p for p, s in scored if s > 0]
                pool = relevant or [p for p, _ in scored]
                pool.sort(key=lambda p: _price_reais(p) or float("inf"))
                ranked = _spread_by_price(pool, _PRICE_RANGE_TOP_N)
            else:
                ranked = [p for p, _ in scored[:_PRICE_RANGE_TOP_N]]
        else:
            products.sort(key=lambda p: _price_reais(p) or float("inf"))
            # A price CEILING/RANGE ("até 2 mil", "entre 1000 e 1500") without an
            # explicit cheapest-first ask → SPREAD across the range so the LLM
            # sees one cheap, one mid, one near the cap (the production fix: it
            # used to get the 8 cheapest, all R$449-469). An explicit "as mais
            # baratas" (sort_by_price) keeps the true ascending head.
            if has_price_filter and not sort_by_price:
                ranked = _spread_by_price(products, _PRICE_RANGE_TOP_N)
            elif sort_by_price:
                # Explicit "as mais baratas" → keep the true cheapest-first head.
                ranked = products[:_PRICE_RANGE_TOP_N]
            else:
                # Sprint 3.9 — bare category/racket browse (no price ask) →
                # newest first, so the customer always sees the latest arrivals.
                ranked = sorted(products, key=_created_at_key, reverse=True)[:_PRICE_RANGE_TOP_N]
    elif q_tokens:
        scored = [(p, _score_product(q_tokens, p)) for p in products]
        scored = [(p, s) for p, s in scored if s > 0]
        # Sprint 3.9 — newest first breaks ties between equally-scored matches.
        scored.sort(key=lambda ps: (ps[1], _created_at_key(ps[0])), reverse=True)
        ranked = [p for p, _ in scored[:5]]
    else:
        # Sprint 3.9 — no name, no price/category intent → newest first.
        ranked = sorted(products, key=_created_at_key, reverse=True)[:5]

    # ── Out-of-stock NAME matches (Sprint 3.12) ──────────────────────────────
    # A named lookup merges back the out-of-stock products that actually match
    # the name (score > 0), so the agent sees the product the customer asked
    # for and can answer "esgotado" honestly. Merged by score (in-stock wins
    # ties) so an exact out-of-stock match outranks an incidental brand match —
    # the production bug was answering about a DIFFERENT in-stock racket.
    if named_lookup and out_of_stock:
        pool = out_of_stock
        if has_category:
            pool = [p for p in pool if _matches_category(p, categoria)]
        oos_matches = sorted(
            ((p, _score_product(q_tokens, p)) for p in pool),
            key=lambda ps: -ps[1],
        )
        oos_matches = [(p, s) for p, s in oos_matches if s > 0][:3]
        if oos_matches:
            merged = [(p, _score_product(q_tokens, p), 1) for p in ranked]
            merged += [(p, s, 0) for p, s in oos_matches]
            merged.sort(key=lambda t: (-t[1], -t[2]))
            ranked = [p for p, _, _ in merged][: max(5, _PRICE_RANGE_TOP_N)]

    out = []
    for p in ranked:
        item = {
            "id": str(p.get("id")),
            "nome": p.get("name"),
            "preco": _price_brl(p.get("price_cents")),
        }
        # Mirrored stock status; omitted when the mirror doesn't know, so the
        # LLM falls back to consultar_estoque instead of trusting a blank.
        status = _stock_status(p)
        if status is not None:
            item["estoque"] = status
        out.append(item)
    # Price range of what we actually returned — lets us audit that a ceiling
    # query came back SPREAD (low..high) rather than clustered at the bottom.
    ranked_prices = [pr for pr in (_price_reais(p) for p in ranked) if pr is not None]
    price_lo = min(ranked_prices) if ranked_prices else None
    price_hi = max(ranked_prices) if ranked_prices else None
    logger.info(
        "buscar_catalogo q_tokens=%d price_filter=%s categoria=%s sort=%s racket=%s "
        "results=%d price_range=%s..%s",
        len(q_tokens), has_price_filter, categoria, sort_by_price, wants_racket,
        len(out), price_lo, price_hi,
    )
    return json.dumps(out, ensure_ascii=False)


# Customer-facing labels we DON'T want to surface, and values that are raw
# Bling IDs (pure digits) rather than human-readable text.
_ATTR_SKIP_SUBSTR = ("bateria", "amazon", "es raquete", "tema de varia", "sem genero")


def _is_displayable_attr(label: str, value: Any) -> bool:
    if value is None or str(value).strip() == "":
        return False
    v = str(value).strip()
    if v.isdigit():  # raw Bling reference id, not human text
        return False
    if v.lower() in ("true", "false"):
        return False
    low = _norm(label)
    return not any(skip in low for skip in _ATTR_SKIP_SUBSTR)


@tool
async def detalhes_produto(produto_id: str) -> str:
    """Retorna as especificações técnicas e características de um produto pelo id.
    Use quando o cliente quiser saber detalhes, specs ou para que tipo de jogo um produto serve."""
    try:
        pid = int(produto_id)
    except (TypeError, ValueError):
        return json.dumps({"erro": f"id invalido: {produto_id!r}"}, ensure_ascii=False)

    try:
        from app.sync.bling_repo import fetch_product_by_id
        product = await fetch_product_by_id(pid)
    except Exception as exc:
        logger.warning("detalhes_produto fetch_failed id=%s: %s", pid, exc)
        return json.dumps({"erro": "nao foi possivel consultar o produto"}, ensure_ascii=False)

    if product is None:
        return json.dumps({"erro": f"produto {produto_id} nao encontrado"}, ensure_ascii=False)

    specs: dict[str, Any] = {
        "id": str(product.get("id")),
        "nome": product.get("name"),
        "marca": product.get("marca"),
        "modelo": product.get("modelo"),
        "categoria": product.get("categoria_nome"),
        "preco": _price_brl(product.get("price_cents")),
    }
    if product.get("weight_g"):
        specs["peso"] = f"{product['weight_g']}g"
    if product.get("description"):
        specs["descricao"] = product["description"]

    # Human-readable custom fields (PT-BR labels), filtering Bling-id noise.
    extras: dict[str, str] = {}
    for label, value in (product.get("campos_customizados") or {}).items():
        if label.upper() in ("MARCA", "MODELO"):
            continue  # already surfaced above
        if _is_displayable_attr(label, value):
            extras[label] = str(value)
    if extras:
        specs["caracteristicas"] = extras

    specs = {k: v for k, v in specs.items() if v not in (None, "")}
    logger.info("detalhes_produto id=%s extras=%d", pid, len(extras))
    return json.dumps(specs, ensure_ascii=False)


@tool
async def consultar_estoque(produto_id: str) -> str:
    """Consulta AO VIVO a disponibilidade em estoque de um produto pelo id (o id vem do
    buscar_catalogo). SEMPRE chame antes de afirmar que um produto tem ou não tem estoque,
    quantas unidades restam, ou se dá pra retirar na loja — o campo "estoque" do
    buscar_catalogo é um espelho que pode estar defasado; esta ferramenta é a verdade."""
    try:
        pid = int(produto_id)
    except (TypeError, ValueError):
        return json.dumps({"erro": f"id invalido: {produto_id!r}"}, ensure_ascii=False)

    try:
        from app.sync.bling_stock import get_stock
        saldo = await get_stock(pid)
    except Exception as exc:
        logger.warning("consultar_estoque failed id=%s: %s", pid, exc)
        saldo = None

    if saldo is None:
        # Bling unauthorized / timeout / off → honest "couldn't confirm".
        return json.dumps(
            {"id": str(pid), "disponivel": None, "aviso": "nao consegui confirmar o estoque agora"},
            ensure_ascii=False,
        )
    return json.dumps(
        {"id": str(pid), "em_estoque": saldo > 0, "quantidade": saldo}, ensure_ascii=False
    )


@tool
async def buscar_conhecimento(consulta: str) -> str:
    """Busca informações da loja que NÃO são de produto específico: horário, endereço,
    garantia, formas de pagamento, como funciona a Consultoria, E TAMBÉM a referência
    técnica de raquetes (o que é carbono, EVA e furação e como cada um muda o jogo).
    Use para dúvidas institucionais e sempre que o cliente perguntar do que a raquete é
    feita ou o que um termo técnico (EVA Soft, carbono 12k, número de furos) significa."""
    try:
        from app.rag.retriever import search_knowledge_base
        from app.storage.db import get_session
        async with get_session() as session:
            docs = await search_knowledge_base(session, consulta, k=4)
    except Exception as exc:
        logger.warning("buscar_conhecimento failed: %s", exc)
        return json.dumps(
            {"resultados": [], "aviso": "nao consegui consultar a base de conhecimento agora"},
            ensure_ascii=False,
        )

    if not docs:
        # Empty KB → tell the LLM plainly so it can escalate instead of inventing.
        return json.dumps(
            {"resultados": [], "aviso": "nenhum conteudo encontrado na base de conhecimento"},
            ensure_ascii=False,
        )

    out = []
    for d in docs:
        out.append({
            "titulo": d.get("title") or d.get("titulo") or d.get("category"),
            "conteudo": d.get("content") or d.get("conteudo") or d.get("text") or "",
        })
    logger.info("buscar_conhecimento q_len=%d results=%d", len(consulta), len(out))
    return json.dumps(out, ensure_ascii=False)


@tool
async def escalar_humano(
    motivo: str, resumo: str, state: Annotated[dict, InjectedState]
) -> str:
    """Aciona um atendente humano de verdade. Use quando o cliente pedir falar com pessoa,
    quando a dúvida sair do escopo, ou para encaminhar o fechamento da Consultoria.
    'motivo' = categoria (ex: 'consultoria', 'fora_de_escopo', 'pedido_humano').
    'resumo' = resumo curto da conversa para o atendente."""
    # Identity + history come from the injected graph state, NOT from the LLM.
    # The dossier pipeline consumes ``messages`` + ``phone_hash`` and opens its
    # own DB session. ``handoff_reason`` is set to ``motivo`` so the persisted
    # and delivered dossier reflects why the escalation happened.
    from app.agent.dossier import handoff_dossier_pipeline

    state_for_dossier = dict(state or {})
    state_for_dossier["handoff_reason"] = motivo or "user_requested"
    # The supervisor's free-text ``resumo`` becomes the dossier summary seed —
    # passed via produto_pesquisado-independent path: the pipeline computes its
    # own LLM summary, but we keep the resumo in the state so future phases can
    # use it. For Phase 1 it's logged; the pipeline summarizes from messages.
    phone_hash = state_for_dossier.get("phone_hash") or ""

    try:
        await handoff_dossier_pipeline(state_for_dossier)
        logger.info(
            "escalar_humano dispatched motivo=%s phone_hash=%.8s resumo_len=%d",
            motivo, phone_hash, len(resumo or ""),
        )
    except Exception as exc:
        logger.error("escalar_humano dossier_failed motivo=%s: %s", motivo, exc)
        return "Tentei acionar um atendente mas houve um problema; um humano será avisado."

    return (
        "Atendente acionado. Encaminhei um resumo da conversa para a equipe "
        "e em breve alguém te chama por aqui."
    )


TOOLS_V2 = [buscar_catalogo, detalhes_produto, consultar_estoque, buscar_conhecimento, escalar_humano]
