"""Shared helpers for matching product names against free-text customer messages.

Sprint 1.15 — tolerant matcher with 4 layered strategies, from strict to
forgiving. Used by all follow-up nodes (price_inquiry, product_selection,
product_detail, re_recommendation) and by the triage router.

Layers (each one only runs if the previous returned no match):

    1. Exact normalized substring (lowercase + strip accents + strip
       'raquete '/'pala ' prefix). Confidence: high.
    2. Spaces-collapsed substring — drop ALL whitespace from both name and
       text. Resolves "beach pro foam series 300" vs "BeachPro Foam Series
       300". Confidence: high.
    3. Fuzzy match via difflib.SequenceMatcher with a sliding window the
       size of the candidate name. Ratio ≥ 0.95 → high confidence (typo).
       Ratio ≥ 0.85 → low confidence (heavier typo or partial match).
    4. Token uniqueness — a 5+ char non-generic token in the customer's
       message that belongs to exactly ONE product in the shortlist.
       Confidence: low.

Returns a ``MatchResult`` so the caller can route differently for
high vs low confidence. ``match_product_in_text`` is preserved as a
backward-compatible wrapper that just returns the product (or None).
"""
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text or "")
        if unicodedata.category(c) != "Mn"
    )


def normalize(text: str) -> str:
    return strip_accents((text or "").lower().strip())


_GENERIC_TOKENS = frozenset({
    "raquete", "raquetes", "pala", "carbon", "pro", "elite", "series",
    "edition", "edicao", "edição", "beach", "tennis", "padel", "x", "v",
    # Sprint 2.6.4 — appearing in 80%+ of names, useless as a match signal.
    "fibra", "fibras", "soft", "esporte", "esportivo", "esportiva",
    "produto", "modelo", "kit",
})


# Sprint 2.6.4 — Portuguese stop words + WhatsApp filler that customers
# stuff into queries. Stripped before scoring so "tem a calça da nicole
# nobile?" reduces to {calca, nicole, nobile} → all match the product.
_STOP_WORDS = frozenset({
    "a", "as", "o", "os", "um", "uma", "uns", "umas",
    "e", "ou", "mas", "pra", "para", "por", "com", "sem", "em",
    "no", "na", "nos", "nas", "ao", "aos", "à", "às",
    "de", "do", "da", "dos", "das",
    "que", "qual", "quais", "como", "onde", "quando", "porque", "porquê",
    "isso", "isto", "aquilo", "esse", "essa", "esses", "essas",
    "este", "esta", "estes", "estas", "ele", "ela", "eles", "elas",
    "voce", "voces", "vc", "vcs", "tu", "eu", "nos",
    "tem", "tens", "ter", "tenho", "tinha", "queria", "quero",
    "vou", "vai", "ja", "ai", "la", "ate", "so",
    "muito", "pouco", "mais", "menos", "tudo", "todo", "toda",
    "ser", "esta", "estou", "fica", "ficar",
    "preciso", "gostaria", "veja", "oi", "ola", "ops", "errei",
    "era", "fui", "foi",
})

# Sprint 2.6.8 — multi-word noise phrases (greetings, multi-word courtesy)
# stripped BEFORE tokenization so they never reach the matcher at all.
# Single-word noise (below) is handled token-wise after split.
_QUERY_NOISE_PHRASES: tuple[str, ...] = (
    "bom dia", "boa tarde", "boa noite",
    "e ai", "e aí",
    "me ve", "me vê",
)

# Sprint 2.6.8 — single-token conversational noise that customers wrap
# product queries with. The matcher used to see "ola voce tem a nova
# kronos" and Levenshtein scored short generic products higher than the
# real Kronos in top-3. Now we strip these tokens up front and the
# matcher only sees the product-bearing remainder.
#
# Whitelist by intent, not by blocklisting brand names: anything NOT in
# this list (or STOP_WORDS or GENERIC_TOKENS) survives, so new catalog
# brands the franchise adds tomorrow are never accidentally removed.
_QUERY_NOISE_TOKENS = frozenset({
    # Greetings.
    "ola", "olá", "oi", "opa", "eai",
    # Courtesy / possession / search verbs.
    "teria", "possui", "vende", "vendem",
    "procuro", "procurando", "busco", "buscando", "buscar",
    "manda", "mostra", "manda ai", "mostrar",
    "gostaria",
    # Generic product adjectives.
    "nova", "novo", "novos", "novas",
    "ultima", "última", "ultimo", "último",
    "lancamento", "lançamento", "lancamentos", "lançamentos",
    "versao", "versão", "versoes", "versões",
    # ``modelo`` is already in _GENERIC_TOKENS but plural ``modelos`` is
    # not — and Felipe's real query had it: "quais modelos ama vc tem?".
    "modelos",
    # Spatial/location filler.
    "aqui",
    # Single-letter "k" (as in "quantos k"). Multi-char specs like "12k"
    # survive because they tokenize as a single token "12k", not "k".
    "k",
})

_MIN_DISTINCTIVE_TOKEN_LEN = 3


def _tokenize(text: str) -> list[str]:
    """Split normalized text into alphanumeric tokens."""
    norm = normalize(text)
    return re.findall(r"[a-z0-9]+", norm)


def _distinctive_tokens(text: str) -> set[str]:
    """Return tokens that carry signal — 3+ chars, not a stop word nor a
    generic catalog token.

    Sprint 2.6.4 — used by the token-score match layer so "drop shot
    legacy" reduces to {drop, shot, legacy} (no "raquete" / "de" / "do"
    noise) and matches the long-tailed catalog name even when Levenshtein
    distance would be huge.

    Sprint 2.6.8 — also filters ``_QUERY_NOISE_TOKENS`` so any caller
    that bypasses ``_clean_product_query`` still gets the same noise
    floor.
    """
    return {
        t for t in _tokenize(text)
        if len(t) >= _MIN_DISTINCTIVE_TOKEN_LEN
        and t not in _STOP_WORDS
        and t not in _GENERIC_TOKENS
        and t not in _QUERY_NOISE_TOKENS
    }


def _clean_product_query(raw: str) -> str:
    """Strip conversational noise so the matcher sees only product-relevant
    tokens.

    Sprint 2.6.8 — fixes the "Olá! Você tem a nova Kronos?" case:
    Levenshtein at layer 5 was comparing the WHOLE raw query against
    catalog names and shorter unrelated products (Mochila Raqueteira)
    were scoring better than the real match. We pre-clean the query so
    every downstream layer sees "kronos" alone.

    Steps (order matters):
      1. ``normalize`` — lowercase + strip accents.
      2. Replace multi-word noise phrases ("bom dia", "e ai", "me ve")
         with whitespace.
      3. Strip punctuation by keeping only ``[a-z0-9 ]``.
      4. Tokenize; drop any token in STOP_WORDS / GENERIC_TOKENS /
         QUERY_NOISE_TOKENS.

    Returns the cleaned space-joined string. May be empty (caller MUST
    treat empty as no-match, not error).

    Important — does NOT filter ``_GENERIC_TOKENS``. Those (raquete,
    beach, pro, carbon, series, etc.) are real PRODUCT-NAME words even
    if they appear in 80% of names. The exact/spaces_collapsed/Levenshtein
    layers below need them to find catalog matches like "BeachPro Carbon
    X5". They're only filtered inside ``_distinctive_tokens`` (used by
    the token-score layer), where their lack of discrimination matters.
    Cleaning here removes ONLY conversational noise (greetings, courtesy
    verbs, articles, generic adjectives) — never catalog vocabulary.
    """
    if not raw:
        return ""
    norm = normalize(raw)
    for phrase in _QUERY_NOISE_PHRASES:
        if phrase in norm:
            norm = norm.replace(phrase, " ")
    # Drop punctuation while preserving token boundaries.
    norm = re.sub(r"[^a-z0-9 ]", " ", norm)
    kept = [
        t for t in norm.split()
        if t not in _STOP_WORDS
        and t not in _QUERY_NOISE_TOKENS
    ]
    return " ".join(kept)


def _core_name(text: str) -> str:
    """Return the normalized name without the generic 'raquete '/'pala ' prefix."""
    norm = normalize(text)
    for prefix in ("raquete ", "pala "):
        if norm.startswith(prefix):
            return norm[len(prefix):].strip()
    return norm


def _strip_spaces(s: str) -> str:
    """Return ``s`` with every whitespace character removed."""
    return "".join(s.split())


# ── Public types ─────────────────────────────────────────────────────────────

Confidence = Literal["high", "low", "none"]
Method = Literal["exact", "spaces_collapsed", "fuzzy", "token", "levenshtein", "none"]
# Sprint 2.6.2 — coarser status that drives the recommend node's branching.
Status = Literal["exact", "fuzzy_high", "fuzzy_low", "ambiguous", "none"]


@dataclass
class MatchResult:
    """Outcome of a tolerant product-name match.

    Sprint 2.6.2 adds ``status`` and ``candidates``:
    - ``status="exact"``       → auto-confirm (substring / spaces_collapsed /
                                  Levenshtein distance 0).
    - ``status="fuzzy_high"``  → auto-confirm with typo tolerance
                                  (Levenshtein distance ≤ 1).
    - ``status="fuzzy_low"``   → ASK ("Você quis dizer X?"). ``product`` is
                                  the single best candidate.
    - ``status="ambiguous"``   → multiple candidates within close distance;
                                  consult ``candidates`` (top 3).
    - ``status="none"``        → nothing close enough.
    """

    product: dict | None
    confidence: Confidence
    method: Method
    status: Status = "none"
    candidates: list[dict] | None = None
    distance: int | None = None  # Levenshtein distance when applicable

    @property
    def found(self) -> bool:
        return self.product is not None

    @property
    def needs_confirmation(self) -> bool:
        """True when the agent should ask 'Você quis dizer X?'."""
        return self.status == "fuzzy_low"


_NO_MATCH = MatchResult(product=None, confidence="none", method="none", status="none")


# Thresholds for fuzzy matching.
_FUZZY_HIGH_RATIO = 0.95
_FUZZY_LOW_RATIO = 0.85
# Minimum length for a meaningful match — guards against trivial substring noise.
_MIN_FULL_VARIANT = 4


# ── Layer helpers ────────────────────────────────────────────────────────────

def _name_variants(product: dict) -> list[str]:
    """Return the candidate name forms to compare, ordered by specificity."""
    name = product.get("name") or ""
    full = normalize(name)
    core = _core_name(name)
    out: list[str] = []
    if full and len(full) >= _MIN_FULL_VARIANT:
        out.append(full)
    if core and core != full and len(core) >= _MIN_FULL_VARIANT:
        out.append(core)
    return out


def _try_exact(text_norm: str, products: list[dict]) -> dict | None:
    """Layer 1 — normalized substring match. Returns longest hit or None."""
    candidates: list[tuple[int, dict]] = []
    for p in products:
        for variant in _name_variants(p):
            if variant in text_norm:
                candidates.append((len(variant), p))
                break
    if not candidates:
        return None
    candidates.sort(key=lambda pair: -pair[0])
    return candidates[0][1]


def _try_spaces_collapsed(text_norm: str, products: list[dict]) -> dict | None:
    """Layer 2 — drop every whitespace and try again.

    Fixes the real-world bug: customer typed "beach pro foam series 300"
    but the catalog has "BeachPro Foam Series 300". After whitespace
    removal both become "beachprofoamseries300".
    """
    text_collapsed = _strip_spaces(text_norm)
    if not text_collapsed:
        return None
    candidates: list[tuple[int, dict]] = []
    for p in products:
        for variant in _name_variants(p):
            v = _strip_spaces(variant)
            if v and len(v) >= _MIN_FULL_VARIANT and v in text_collapsed:
                candidates.append((len(v), p))
                break
    if not candidates:
        return None
    candidates.sort(key=lambda pair: -pair[0])
    return candidates[0][1]


def _best_fuzzy_ratio(text_norm: str, name_norm: str) -> float:
    """Return the best SequenceMatcher ratio of any same-length window of
    ``text_norm`` against ``name_norm``.

    For short messages (most WhatsApp follow-ups) this is cheap. We slide a
    window of len(name_norm) across text_norm and pick the highest ratio.
    """
    if not text_norm or not name_norm:
        return 0.0
    name_len = len(name_norm)
    if len(text_norm) <= name_len:
        return SequenceMatcher(None, text_norm, name_norm).ratio()
    best = 0.0
    for i in range(len(text_norm) - name_len + 1):
        window = text_norm[i:i + name_len]
        ratio = SequenceMatcher(None, window, name_norm).ratio()
        if ratio > best:
            best = ratio
            if best >= 0.999:  # already perfect — stop early
                break
    return best


def _try_fuzzy(text_norm: str, products: list[dict]) -> tuple[dict | None, Confidence]:
    """Layer 3 — fuzzy match with confidence based on ratio thresholds."""
    text_collapsed = _strip_spaces(text_norm)

    best_product: dict | None = None
    best_ratio = 0.0
    for p in products:
        for variant in _name_variants(p):
            # Compare both with-spaces and collapsed forms; take the higher.
            v_collapsed = _strip_spaces(variant)
            ratio = max(
                _best_fuzzy_ratio(text_norm, variant),
                _best_fuzzy_ratio(text_collapsed, v_collapsed),
            )
            if ratio > best_ratio:
                best_ratio = ratio
                best_product = p

    if best_product is None or best_ratio < _FUZZY_LOW_RATIO:
        return None, "none"
    if best_ratio >= _FUZZY_HIGH_RATIO:
        return best_product, "high"
    return best_product, "low"


# ── Levenshtein (Sprint 2.6.2) ───────────────────────────────────────────────

def _levenshtein(a: str, b: str) -> int:
    """Iterative two-row Levenshtein. O(n*m) time, O(min(n,m)) memory.

    Returns the minimum number of single-character edits (insertions,
    deletions, substitutions) to transform ``a`` into ``b``.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Force ``b`` to be the shorter to bound the row size.
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(
                curr[j - 1] + 1,         # insertion
                prev[j] + 1,             # deletion
                prev[j - 1] + (0 if ca == cb else 1),  # substitution
            ))
        prev = curr
    return prev[-1]


def _best_lev_distance(query_collapsed: str, name_collapsed: str) -> int:
    """Best Levenshtein distance between ``query_collapsed`` and any
    same-length window of ``name_collapsed`` (or the full name when shorter).

    Sliding window lets a short query like "emithammer" match the relevant
    sub-region of a longer name like "emithammercarbono3k" without paying
    the trailing-suffix cost.
    """
    if not query_collapsed or not name_collapsed:
        return max(len(query_collapsed or ""), len(name_collapsed or ""))
    q_len = len(query_collapsed)
    n_len = len(name_collapsed)
    if q_len >= n_len:
        return _levenshtein(query_collapsed, name_collapsed)
    best = q_len
    for i in range(n_len - q_len + 1):
        window = name_collapsed[i:i + q_len]
        d = _levenshtein(query_collapsed, window)
        if d < best:
            best = d
            if best == 0:
                return 0
    return best


def _try_levenshtein(
    text_norm: str, products: list[dict]
) -> tuple[dict | None, list[dict], list[dict], int]:
    """Score every product by best Levenshtein distance against name variants.

    Returns ``(best_product, tied_at_best, top_close, best_distance)``:
    - ``best_product`` is the single closest product.
    - ``tied_at_best`` is the list of products whose best distance EQUALS
      ``best_distance`` — used for ambiguity at distance 0–1.
    - ``top_close`` is the top 3 products ordered by distance — used to
      build the "Você quis dizer X?" candidate list at fuzzy_low.
    - ``best_distance`` is 999 when nothing was scored (empty inputs).
    """
    text_collapsed = _strip_spaces(text_norm)
    if not text_collapsed:
        return None, [], [], 999

    scored: list[tuple[int, dict]] = []
    for p in products:
        best_d = 999
        for variant in _name_variants(p):
            v_collapsed = _strip_spaces(variant)
            if not v_collapsed:
                continue
            d = _best_lev_distance(text_collapsed, v_collapsed)
            if d < best_d:
                best_d = d
                if best_d == 0:
                    break
        if best_d < 999:
            scored.append((best_d, p))

    if not scored:
        return None, [], [], 999

    scored.sort(key=lambda x: x[0])
    best_d, best_p = scored[0]
    tied_at_best = [p for d, p in scored if d == best_d]
    top_close = [p for _, p in scored[:3]]
    return best_p, tied_at_best, top_close, best_d


# ── Sprint 2.6.4 — token-score layer ─────────────────────────────────────────

def _try_token_score(
    query: str, products: list[dict]
) -> tuple[dict | None, list[dict], float, list[dict]]:
    """Score each product by fraction of distinctive query tokens present
    in its name (after the same normalization). Returns
    ``(best_product, tied_at_best, best_score, top5)``.

    The score is independent of name length, so a 4-token query matches
    perfectly even when the catalog name has 12 extra words of suffix
    (model, year, size, fabric, code, ...). That's the case the previous
    Levenshtein-only layer couldn't handle.
    """
    q_tokens = _distinctive_tokens(query)
    if not q_tokens:
        return None, [], 0.0, []

    scored: list[tuple[float, dict]] = []
    for p in products:
        name_tokens = set(_tokenize(p.get("name") or ""))
        if not name_tokens:
            continue
        overlap = sum(1 for t in q_tokens if t in name_tokens)
        if overlap == 0:
            continue
        score = overlap / len(q_tokens)
        scored.append((score, p))

    if not scored:
        return None, [], 0.0, []

    scored.sort(key=lambda x: -x[0])
    best_score, best_p = scored[0]
    tied_at_best = [p for s, p in scored if s == best_score]
    top5 = [p for _, p in scored[:5]]
    return best_p, tied_at_best, best_score, top5


_TOKEN_HIGH_SCORE = 1.0   # all distinctive query tokens land in the name
_TOKEN_LOW_SCORE = 0.7    # >70% land → ask "você quis dizer?"


def _normalized_distance(distance: int, query_len: int, name_len: int) -> float:
    """Distance / min(len_a, len_b). Returns 1.0 when either side is empty."""
    denom = min(query_len, name_len)
    if denom <= 0:
        return 1.0
    return distance / denom


_NORM_LEV_HIGH = 0.4      # ≤ this → fuzzy_high (typo trivial relative to length)
_NORM_LEV_LOW = 0.6       # ≤ this → fuzzy_low (suggest)


def _try_token_unicity(text_norm: str, products: list[dict]) -> dict | None:
    """Layer 4 — a 5+ char non-generic token belongs to exactly 1 product."""
    token_owners: dict[str, set[int]] = {}
    for idx, p in enumerate(products):
        name_norm = normalize(p.get("name") or "")
        for token in name_norm.split():
            if len(token) < 5 or token in _GENERIC_TOKENS:
                continue
            token_owners.setdefault(token, set()).add(idx)

    best_product: dict | None = None
    best_score = 0
    for token, owners in token_owners.items():
        if len(owners) != 1:
            continue
        if token in text_norm:
            score = len(token)
            if score > best_score:
                idx = next(iter(owners))
                best_product = products[idx]
                best_score = score
    return best_product


# ── Public API ───────────────────────────────────────────────────────────────

_LEV_DISTANCE_HIGH = 1   # ≤ this → auto-confirm (typo trivial)
_LEV_DISTANCE_LOW = 3    # ≤ this → suggest ("Você quis dizer X?")


def match_product_tolerant(text: str, products: list[dict]) -> MatchResult:
    """Return the best match with explicit ``status``, ``confidence`` and
    ``method`` (Sprint 2.6.4 layering).

    Layered evaluation:
      1. Exact substring (normalized)            → status="exact"
      2. Spaces-collapsed substring              → status="exact"
      3. Token-score (Sprint 2.6.4):
           - 100% of distinctive tokens → exact (single) / ambiguous (tied)
           - ≥70% → fuzzy_low (single) / ambiguous (tied)
      4. SequenceMatcher fuzzy ratio ≥ 0.95      → status="exact"
      5. Levenshtein normalized ≤ 0.4 (or abs ≤ 1) → status="fuzzy_high"
                                                    (multiple tied → "ambiguous")
      6. Levenshtein normalized ≤ 0.6 (or abs ≤ 3) → status="fuzzy_low"
                                                    (multiple tied → "ambiguous")
      7. SequenceMatcher fuzzy ratio ≥ 0.85      → status="fuzzy_low"
      8. Token unicity                           → status="fuzzy_low"
      9. None                                    → status="none"
    """
    if not text or not products:
        return _NO_MATCH
    text_norm_raw = normalize(text)
    if not text_norm_raw:
        return _NO_MATCH

    # Sprint 2.6.8 — strip greetings / courtesy / generic adjectives BEFORE
    # any layer sees the query. Otherwise Levenshtein scores short
    # unrelated catalog products above the real match because the noise
    # words bloat the query length. If cleaning empties the query (the
    # customer wrote only filler, e.g. "vocês têm?"), treat as no-match.
    text_norm = _clean_product_query(text)
    import logging
    _log = logging.getLogger("app.agent.nodes._product_match")
    _log.info(
        "match_attempt query_raw=%.80r query_clean=%.80r total_products=%d",
        text_norm_raw, text_norm, len(products),
    )
    if not text_norm:
        _log.info("match_no_result reason=query_empty_after_cleaning")
        return _NO_MATCH

    # Layer 1 — exact normalized substring.
    p = _try_exact(text_norm, products)
    if p is not None:
        _log.info("match_result status=exact method=exact name=%s", p.get("name"))
        return MatchResult(
            product=p, confidence="high", method="exact", status="exact",
            candidates=[p], distance=0,
        )

    # Layer 2 — collapsed whitespace.
    p = _try_spaces_collapsed(text_norm, products)
    if p is not None:
        _log.info(
            "match_result status=exact method=spaces_collapsed name=%s", p.get("name")
        )
        return MatchResult(
            product=p, confidence="high", method="spaces_collapsed", status="exact",
            candidates=[p], distance=0,
        )

    # Layer 3 (Sprint 2.6.4) — token-score: did the customer name all the
    # distinctive tokens of the product?
    tok_best, tok_tied, tok_score, tok_top = _try_token_score(text, products)
    if tok_best is not None and tok_score >= _TOKEN_HIGH_SCORE:
        if len(tok_tied) > 1:
            _log.info(
                "match_result status=ambiguous method=token score=%.2f n_tied=%d",
                tok_score, len(tok_tied),
            )
            return MatchResult(
                product=None, confidence="none", method="token",
                status="ambiguous", candidates=tok_tied[:5],
            )
        _log.info(
            "match_result status=exact method=token score=%.2f name=%s",
            tok_score, tok_best.get("name"),
        )
        return MatchResult(
            product=tok_best, confidence="high", method="token", status="exact",
            candidates=[tok_best],
        )

    if tok_best is not None and tok_score >= _TOKEN_LOW_SCORE:
        if len(tok_tied) > 1:
            _log.info(
                "match_result status=ambiguous method=token score=%.2f n_tied=%d",
                tok_score, len(tok_tied),
            )
            return MatchResult(
                product=None, confidence="none", method="token",
                status="ambiguous", candidates=tok_tied[:5],
            )
        _log.info(
            "match_result status=fuzzy_low method=token score=%.2f name=%s",
            tok_score, tok_best.get("name"),
        )
        return MatchResult(
            product=tok_best, confidence="low", method="token",
            status="fuzzy_low", candidates=tok_top,
        )

    # Layer 4 — SequenceMatcher fuzzy at ≥0.95 ratio (existing high-confidence path).
    p_fz, conf_fz = _try_fuzzy(text_norm, products)
    if p_fz is not None and conf_fz == "high":
        _log.info("match_result status=exact method=fuzzy name=%s", p_fz.get("name"))
        return MatchResult(
            product=p_fz, confidence="high", method="fuzzy", status="exact",
            candidates=[p_fz],
        )

    # Layer 5/6 — Levenshtein. Sprint 2.6.4 — normalize by min length so
    # long catalog names with extra tail (model/year/code) don't blow up
    # the distance.
    best_p, tied_at_best, top_close, best_d = _try_levenshtein(text_norm, products)
    if best_p is not None:
        q_collapsed = _strip_spaces(text_norm)
        # Use the BEST variant length (shortest) for the normalizer so the
        # "raquete X" prefix doesn't dilute the ratio.
        best_variant = min(
            (_strip_spaces(v) for v in _name_variants(best_p)),
            key=len, default="",
        )
        norm_d = _normalized_distance(best_d, len(q_collapsed), len(best_variant))

        is_high = best_d <= _LEV_DISTANCE_HIGH or norm_d <= _NORM_LEV_HIGH
        is_low = best_d <= _LEV_DISTANCE_LOW or norm_d <= _NORM_LEV_LOW

        if is_high:
            if len(tied_at_best) > 1:
                _log.info(
                    "match_result status=ambiguous method=levenshtein distance=%d norm=%.2f n_tied=%d",
                    best_d, norm_d, len(tied_at_best),
                )
                return MatchResult(
                    product=None, confidence="none", method="levenshtein",
                    status="ambiguous", candidates=tied_at_best[:3], distance=best_d,
                )
            _log.info(
                "match_result status=fuzzy_high method=levenshtein distance=%d norm=%.2f name=%s",
                best_d, norm_d, best_p.get("name"),
            )
            return MatchResult(
                product=best_p, confidence="high", method="levenshtein",
                status="fuzzy_high", candidates=[best_p], distance=best_d,
            )

        if is_low:
            if len(tied_at_best) > 1:
                _log.info(
                    "match_result status=ambiguous method=levenshtein distance=%d norm=%.2f n_tied=%d",
                    best_d, norm_d, len(tied_at_best),
                )
                return MatchResult(
                    product=None, confidence="none", method="levenshtein",
                    status="ambiguous", candidates=tied_at_best[:3], distance=best_d,
                )
            _log.info(
                "match_result status=fuzzy_low method=levenshtein distance=%d norm=%.2f name=%s",
                best_d, norm_d, best_p.get("name"),
            )
            return MatchResult(
                product=best_p, confidence="low", method="levenshtein",
                status="fuzzy_low", candidates=top_close, distance=best_d,
            )

    # Layer 6 — looser fuzzy.
    if p_fz is not None:
        _log.info("match_result status=fuzzy_low method=fuzzy name=%s", p_fz.get("name"))
        return MatchResult(
            product=p_fz, confidence="low", method="fuzzy", status="fuzzy_low",
            candidates=[p_fz],
        )

    # Layer 7 — token uniqueness.
    p_tok = _try_token_unicity(text_norm, products)
    if p_tok is not None:
        _log.info("match_result status=fuzzy_low method=token name=%s", p_tok.get("name"))
        return MatchResult(
            product=p_tok, confidence="low", method="token", status="fuzzy_low",
            candidates=[p_tok],
        )

    # Sprint 2.6.3 — when nothing matched, dump the 3 closest candidates so
    # operators can debug catalog truncation / parsing / brand-name mismatch
    # without re-running the request. ``best_d`` and ``top_close`` come from
    # the Levenshtein layer above and are always defined by the time we
    # reach this final return.
    if top_close:
        diag = [
            {"name": str(p.get("name", "?"))[:80], "id": p.get("id")}
            for p in top_close[:3]
        ]
        _log.info(
            "match_no_result query=%.100r total_products=%d "
            "best_distance=%s top3=%s",
            text_norm, len(products), best_d, diag,
        )
    else:
        _log.info(
            "match_no_result query=%.100r total_products=%d (empty_catalog)",
            text_norm, len(products),
        )
    return _NO_MATCH


def match_product_in_text(text: str, products: list[dict]) -> dict | None:
    """Backward-compatible wrapper: returns just the product (or None).

    Sprint 1.15 — internally uses the new tolerant pipeline. Callers that
    don't care about confidence keep working unchanged.
    """
    return match_product_tolerant(text, products).product


# ── Other shared helpers ─────────────────────────────────────────────────────

def format_price_brl(price_cents: int | float | None) -> str:
    """Return a BRL-formatted price string like 'R$ 1.299' (no centavos)."""
    if price_cents is None:
        return "R$ -"
    try:
        reais = int(price_cents) // 100
    except (TypeError, ValueError):
        return "R$ -"
    return f"R$ {reais:,}".replace(",", ".")
