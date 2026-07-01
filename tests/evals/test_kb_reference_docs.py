"""T2 / T3 — knowledge-base reference docs present and retrievable.

T2: technical racket reference covering Carbono (1k–24k; more carbon = stiffer),
    EVA (Soft macio/conforto/impulsão; Tech duro/controle/batida seca; Pro meio
    termo) and Furação (more holes = softer/elastic; fewer = firmer/harder).
T3: Consultoria Base Sports text (Investimento R$ 350; gratuita comprando a
    raquete; etapas Diagnóstico e Teste em quadra).

Two layers, both deterministic:
  - SOURCE inspection of ``scripts/seed_knowledge_base.py::DOCUMENTS`` (the
    canonical doc source dev-tools edits). RED until the docs are added.
  - WIRING: ``buscar_conhecimento`` surfaces KB content to the LLM (KB search
    patched). Passes today — guards the tool→LLM plumbing regardless of seed.
"""
import importlib.util
import json
from pathlib import Path

import pytest

from tests.evals._helpers import run_buscar_conhecimento

pytestmark = pytest.mark.deterministic

_SEED_PATH = Path(__file__).resolve().parents[2] / "scripts" / "seed_knowledge_base.py"

_RACKET_TITLE = "Materiais da raquete: carbono, EVA e furação (o que muda no jogo)"
_CONSULTORIA_TITLE = "Consultoria Base Sports"


def _load_documents() -> list[dict]:
    spec = importlib.util.spec_from_file_location("seed_kb_for_eval", _SEED_PATH)
    assert spec and spec.loader, f"cannot load seed module at {_SEED_PATH}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.DOCUMENTS)


DOCUMENTS = _load_documents()


def _doc_by_title(title: str) -> dict | None:
    return next((d for d in DOCUMENTS if d.get("title") == title), None)


# ── T2: racket technical reference ───────────────────────────────────────────

def test_seed_has_racket_reference_doc():
    d = _doc_by_title(_RACKET_TITLE)
    assert d is not None, f"missing racket reference doc titled {_RACKET_TITLE!r}"
    assert d.get("category") == "faq", f"racket doc category should be 'faq': {d.get('category')!r}"
    c = d["content"].lower()
    assert "carbono" in c, "racket doc must cover Carbono"
    assert "eva" in c, "racket doc must cover EVA"
    assert "fura" in c, "racket doc must cover Furação"


def test_carbono_detail_present():
    d = _doc_by_title(_RACKET_TITLE)
    assert d is not None
    c = d["content"].lower()
    assert "1k" in c and "24k" in c, "Carbono scale 1k..24k missing"
    assert "mais carbono" in c, "missing the 'mais carbono' relation"
    assert any(w in c for w in ("dura", "rígid", "rigid", "dur")), \
        "Carbono must state more carbon = stiffer/harder"


def test_eva_variants_present():
    d = _doc_by_title(_RACKET_TITLE)
    assert d is not None
    c = d["content"].lower()
    for variant in ("soft", "tech", "pro"):
        assert variant in c, f"EVA variant {variant!r} missing"
    assert any(w in c for w in ("macio", "conforto", "impuls")), "Soft impact descriptor missing"
    assert any(w in c for w in ("controle", "batida seca", "duro")), "Tech impact descriptor missing"
    assert any(w in c for w in ("meio termo", "equilib", "equilíb")), "Pro impact descriptor missing"


def test_furacao_detail_present():
    d = _doc_by_title(_RACKET_TITLE)
    assert d is not None
    c = d["content"].lower()
    assert "furo" in c, "Furação must mention furos"
    assert "mais furos" in c and any(w in c for w in ("macia", "elástic", "elastic")), \
        "missing 'mais furos = mais macia/elástica'"
    assert "menos furos" in c and any(w in c for w in ("firme", "dura", "rígid", "rigid")), \
        "missing 'menos furos = mais firme/dura'"


# ── T3: Consultoria text ─────────────────────────────────────────────────────

def test_seed_has_consultoria_doc():
    d = _doc_by_title(_CONSULTORIA_TITLE)
    assert d is not None, f"missing consultoria doc titled {_CONSULTORIA_TITLE!r}"
    assert d.get("category") == "store", f"consultoria doc category should be 'store': {d.get('category')!r}"
    c = d["content"].lower()
    assert "350" in c, "Consultoria must state R$ 350"
    assert any(w in c for w in ("gratuita", "grátis", "gratis")), "must state it's free when buying the racket"
    assert "diagn" in c, "must mention the Diagnóstico stage"
    assert "teste em quadra" in c, "must mention the Teste em quadra stage"


def test_consultoria_doc_lists_two_stages():
    d = _doc_by_title(_CONSULTORIA_TITLE)
    assert d is not None
    c = d["content"].lower()
    assert "diagn" in c and "teste em quadra" in c, "both stages (Diagnóstico, Teste em quadra) required"


def test_consultoria_doc_brand_is_base_sports():
    d = _doc_by_title(_CONSULTORIA_TITLE)
    assert d is not None
    assert "Base Sports" in d["content"] or "Base Sports" in d["title"]
    assert "Base Esportes" not in d["content"], "brand regression: use 'Base Sports', not 'Base Esportes'"


# ── Wiring: KB → buscar_conhecimento → LLM (deterministic, passes today) ──────

@pytest.mark.asyncio
async def test_buscar_conhecimento_surfaces_racket_doc():
    docs = [{
        "title": _RACKET_TITLE,
        "content": "Carbono vai de 1k a 24k; mais carbono deixa a raquete mais dura. EVA Soft é macio. Furação muda o jogo.",
        "category": "faq",
    }]
    out = await run_buscar_conhecimento(docs, "diferença carbono eva furação")
    blob = json.dumps(out, ensure_ascii=False).lower()
    assert "carbono" in blob and "eva" in blob, f"racket KB content not surfaced: {out}"


@pytest.mark.asyncio
async def test_buscar_conhecimento_surfaces_consultoria():
    docs = [{
        "title": _CONSULTORIA_TITLE,
        "content": "Investimento de R$ 350, gratuita comprando a raquete. Etapas: Diagnóstico e Teste em quadra.",
        "category": "store",
    }]
    out = await run_buscar_conhecimento(docs, "como funciona a consultoria")
    blob = json.dumps(out, ensure_ascii=False).lower()
    assert "350" in blob and "diagn" in blob and "teste em quadra" in blob, \
        f"consultoria KB content not surfaced: {out}"
