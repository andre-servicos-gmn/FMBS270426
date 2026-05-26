"""Populate the Supabase products table with 20 fake dev/test products.

Usage (from project root, with .venv activated):
    python scripts/seed_catalog.py

Requires OPENAI_API_KEY and DATABASE_URL in .env.
Consumes ~$0.0003 in OpenAI embeddings (20 texts x ~100 tokens each).
"""
import asyncio
import logging
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import func, select  # noqa: E402
from sqlalchemy.dialects.postgresql import insert as pg_insert  # noqa: E402

from app.config import configure_logging, get_settings  # noqa: E402
from app.rag.embeddings import embed_batch  # noqa: E402
from app.storage.db import get_session  # noqa: E402
from app.storage.models import Product  # noqa: E402

configure_logging(get_settings())
logger = logging.getLogger(__name__)

# ── 20 fake products ──────────────────────────────────────────────────────────
# Beach Tennis brands: BeachPro, AirBlast, VertexBT, ForceX, NovaSport, SpeedLine
# Padel brands:       PadelMax, ApexPadel, TitanPadel, GridPadel, SpeedLine
# All brand names are fictional.

PRODUCTS: list[dict] = [
    # ── Beach Tennis — iniciante ──────────────────────────────────────────────
    {
        "external_id": "BT001",
        "name": "Raquete AirBlast Starter BT",
        "sport": "beach_tennis",
        "level": "iniciante",
        "weight_g": 370,
        "balance": "médio",
        "material": "fibra de vidro + núcleo EVA 45 Shore",
        "price_cents": 34900,
        "stock": 20,
        "description": (
            "Raquete de beach tennis ideal para quem está começando. "
            "Face de fibra de vidro com núcleo em espuma EVA de dureza média (45 Shore), "
            "proporcionando boa absorção de impacto e controle fácil. "
            "Balanço equilibrado facilita os primeiros golpes e saques."
        ),
    },
    {
        "external_id": "BT002",
        "name": "Raquete BeachPro Foam Series 300",
        "sport": "beach_tennis",
        "level": "iniciante",
        "weight_g": 385,
        "balance": "leve para cabo",
        "material": "polipropileno + EVA 40 Shore",
        "price_cents": 29900,
        "stock": 35,
        "description": (
            "Raquete entry-level para beach tennis com estrutura em polipropileno resistente "
            "e espuma EVA de baixa dureza (40 Shore) para máximo conforto. "
            "Peso mais alto no cabo facilita saques para jogadores iniciantes. "
            "Excelente custo-benefício para treinos diários na areia."
        ),
    },
    # ── Beach Tennis — intermediário ──────────────────────────────────────────
    {
        "external_id": "BT003",
        "name": "Raquete BeachPro Carbon X5",
        "sport": "beach_tennis",
        "level": "intermediário",
        "weight_g": 355,
        "balance": "médio",
        "material": "fibra de carbono 3K + EVA 50 Shore",
        "price_cents": 89900,
        "stock": 15,
        "description": (
            "Raquete de beach tennis com face em fibra de carbono 3K, "
            "entregando boa potência sem abrir mão do controle. "
            "Núcleo em EVA de dureza média-alta (50 Shore) para jogadores que já têm "
            "técnica definida e buscam mais saída de bola. Balanço equilibrado."
        ),
    },
    {
        "external_id": "BT004",
        "name": "Raquete NovaSport Speed BT",
        "sport": "beach_tennis",
        "level": "intermediário",
        "weight_g": 345,
        "balance": "leve para cabeça",
        "material": "fibra de carbono 12K + EVA 48 Shore",
        "price_cents": 69900,
        "stock": 12,
        "description": (
            "Design diamante com balanço leve para a cabeça, ideal para jogadores "
            "intermediários que preferem estilo ofensivo. "
            "Face em carbono 12K com acabamento texturizado para maior efeito na bola. "
            "Núcleo EVA 48 Shore equilibra velocidade e controle."
        ),
    },
    {
        "external_id": "BT005",
        "name": "Raquete AirBlast Carbon Pro",
        "sport": "beach_tennis",
        "level": "intermediário",
        "weight_g": 360,
        "balance": "leve para cabo",
        "material": "carbono unidirecional + EVA 52 Shore",
        "price_cents": 99900,
        "stock": 8,
        "description": (
            "Perfil redondo com balanço voltado ao cabo, priorizando controle e precisão. "
            "Carbono unidirecional na face dá rigidez extra sem aumentar o peso. "
            "EVA de dureza 52 Shore mantém a bola mais tempo na raquete para jogadores "
            "que precisam de mais tempo de reação."
        ),
    },
    # ── Beach Tennis — avançado ───────────────────────────────────────────────
    {
        "external_id": "BT006",
        "name": "Raquete VertexBT Pro Elite",
        "sport": "beach_tennis",
        "level": "avançado",
        "weight_g": 330,
        "balance": "alto (cabeça pesada)",
        "material": "carbono 18K full frame + EVA 58 Shore",
        "price_cents": 179900,
        "stock": 6,
        "description": (
            "Raquete de alto desempenho para beach tennis avançado. "
            "Frame completo em carbono 18K com reforço nas laterais para máxima durabilidade. "
            "EVA de alta dureza (58 Shore) amplifica a potência dos golpes e saques. "
            "Balanço alto (cabeça pesada) para jogadores com swing completo e controlado."
        ),
    },
    {
        "external_id": "BT007",
        "name": "Raquete SpeedLine Ultra BT",
        "sport": "beach_tennis",
        "level": "avançado",
        "weight_g": 340,
        "balance": "médio-alto",
        "material": "fibra de carbono HS + Texalium",
        "price_cents": 129900,
        "stock": 9,
        "description": (
            "Combinação de fibra de carbono HS (high strength) com Texalium na face "
            "para máxima rigidez e resposta de bola. "
            "Para jogadores avançados de beach tennis que buscam velocidade de bola "
            "aliada a uma sweet spot ampliada. Peso otimizado em 340g."
        ),
    },
    # ── Beach Tennis — competidor ─────────────────────────────────────────────
    {
        "external_id": "BT008",
        "name": "Raquete ForceX Competition BT",
        "sport": "beach_tennis",
        "level": "competidor",
        "weight_g": 318,
        "balance": "alto (cabeça pesada)",
        "material": "carbono 24K toray + EVA 62 Shore",
        "price_cents": 249900,
        "stock": 4,
        "description": (
            "Raquete de competição para beach tennis de alto nível. "
            "Face em carbono Toray 24K de altíssima resistência, com perfuração aerodinâmica "
            "para reduzir resistência do ar no swing. "
            "EVA 62 Shore (dura) para máxima transferência de energia nos saques. "
            "Homologada para torneios profissionais."
        ),
    },
    # ── Beach Tennis — acessórios ─────────────────────────────────────────────
    {
        "external_id": "BT009",
        "name": "Bolsa BeachPro Tournament Bag",
        "sport": "beach_tennis",
        "level": None,
        "weight_g": None,
        "balance": None,
        "material": "poliéster 600D impermeável",
        "price_cents": 39900,
        "stock": 25,
        "description": (
            "Bolsa para beach tennis com compartimento principal para 2 raquetes, "
            "bolso térmico para garrafas e bolso frontal para acessórios. "
            "Material em poliéster 600D impermeável, resistente à areia. "
            "Alça de ombro ajustável e reforçada."
        ),
    },
    {
        "external_id": "BT010",
        "name": "Kit Bolas BeachPro Pressurized x3",
        "sport": "beach_tennis",
        "level": None,
        "weight_g": None,
        "balance": None,
        "material": "borracha natural + feltro ITF",
        "price_cents": 8900,
        "stock": 100,
        "description": (
            "Kit com 3 bolas de beach tennis pressurizadas, certificadas pela ITF. "
            "Feltro de alta durabilidade para uso na areia, com costura reforçada. "
            "Pressão ideal para saques e rallies com boa velocidade e trajetória previsível."
        ),
    },
    # ── Padel — iniciante ─────────────────────────────────────────────────────
    {
        "external_id": "PD001",
        "name": "Raquete PadelMax Control Entry",
        "sport": "padel",
        "level": "iniciante",
        "weight_g": 365,
        "balance": "médio",
        "material": "fibra de vidro + EVA Soft 38 Shore",
        "price_cents": 37900,
        "stock": 30,
        "description": (
            "Raquete de padel para iniciantes com perfil redondo e balanço equilibrado, "
            "maximizando a área de contato e facilitando o controle da bola. "
            "Face em fibra de vidro com núcleo EVA Soft (38 Shore) para absorver vibrações "
            "e proteger o cotovelo e pulso em longos treinos."
        ),
    },
    {
        "external_id": "PD002",
        "name": "Raquete TitanPadel Junior",
        "sport": "padel",
        "level": "iniciante",
        "weight_g": 340,
        "balance": "leve para cabo",
        "material": "polipropileno + EVA 40 Shore",
        "price_cents": 32900,
        "stock": 18,
        "description": (
            "Raquete de padel leve para iniciantes e jogadores mais jovens. "
            "Peso reduzido de 340g facilita a mobilidade na rede e nos lobs. "
            "Núcleo EVA 40 Shore confortável para partidas longas. "
            "Design compacto e boa durabilidade para treinos intensivos."
        ),
    },
    # ── Padel — intermediário ─────────────────────────────────────────────────
    {
        "external_id": "PD003",
        "name": "Raquete NovaSport Padel Mid",
        "sport": "padel",
        "level": "intermediário",
        "weight_g": 355,
        "balance": "médio",
        "material": "fibra de carbono + EVA 50 Shore",
        "price_cents": 74900,
        "stock": 14,
        "description": (
            "Raquete de padel polivalente para nível intermediário. "
            "Perfil teardrop (lágrima) com face em carbono para boa mistura de potência e controle. "
            "EVA 50 Shore no núcleo dá sensação de bola boa sem sacrificar velocidade. "
            "Indicada para jogadores que transitam entre defesa e ataque."
        ),
    },
    {
        "external_id": "PD004",
        "name": "Raquete ApexPadel Hybrid Mid",
        "sport": "padel",
        "level": "intermediário",
        "weight_g": 360,
        "balance": "médio-alto",
        "material": "carbono 3K + fibra de vidro híbrido + EVA 52 Shore",
        "price_cents": 109900,
        "stock": 10,
        "description": (
            "Construção híbrida com carbono 3K na face e reforço de fibra de vidro nas laterais "
            "para maior durabilidade em jogos competitivos amadores. "
            "Balanço médio-alto oferece mais potência no smash sem perder precisão nos volleys. "
            "Núcleo EVA 52 Shore ideal para intermediários que querem evoluir."
        ),
    },
    {
        "external_id": "PD005",
        "name": "Raquete GridPadel Power Mid",
        "sport": "padel",
        "level": "intermediário",
        "weight_g": 370,
        "balance": "alto",
        "material": "fibra de carbono + foam HR",
        "price_cents": 89900,
        "stock": 7,
        "description": (
            "Perfil diamante com balanço alto para jogadores intermediários de estilo ofensivo. "
            "Foam HR (alta resiliência) no lugar do EVA tradicional amplia o efeito trampolim "
            "para smashes mais potentes. "
            "Face em carbono com acabamento rugoso para gerar mais efeito na bola."
        ),
    },
    # ── Padel — avançado ──────────────────────────────────────────────────────
    {
        "external_id": "PD006",
        "name": "Raquete ApexPadel Power Pro",
        "sport": "padel",
        "level": "avançado",
        "weight_g": 345,
        "balance": "alto (cabeça pesada)",
        "material": "carbono 12K + FOAM HR 60 Shore",
        "price_cents": 159900,
        "stock": 8,
        "description": (
            "Raquete de padel avançada com perfil diamante e balanço alto para smashes devastadores. "
            "Carbono 12K proporciona rigidez extrema e ótima resposta tátil. "
            "Foam HR (60 Shore) maximiza a velocidade de saída de bola nos remates. "
            "Para jogadores de nível avançado com técnica apurada no smash e bandeja."
        ),
    },
    {
        "external_id": "PD007",
        "name": "Raquete SpeedLine Padel Pro",
        "sport": "padel",
        "level": "avançado",
        "weight_g": 350,
        "balance": "médio-alto",
        "material": "carbono UD + kevlar reforçado",
        "price_cents": 134900,
        "stock": 5,
        "description": (
            "Construção em carbono unidirecional com reforço de kevlar nas bordas para "
            "máxima resistência a impactos nas paredes. "
            "Balanço médio-alto para ótimo equilíbrio entre potência e manobrabilidade. "
            "Sweet spot centralizado e ampliado para jogadores avançados que jogam em alta intensidade."
        ),
    },
    {
        "external_id": "PD008",
        "name": "Raquete PadelMax Titan Advanced",
        "sport": "padel",
        "level": "avançado",
        "weight_g": 355,
        "balance": "médio",
        "material": "carbono 18K full face + EVA 55 Shore",
        "price_cents": 149900,
        "stock": 6,
        "description": (
            "Face completa em carbono 18K com perfil redondo-diamante para jogadores avançados "
            "que priorizam consistência em longos rallies. "
            "EVA 55 Shore equilibrado oferece sensação de bola precisa para volleys e bandeja. "
            "Acabamento matte anti-brilho premium."
        ),
    },
    # ── Padel — competidor ────────────────────────────────────────────────────
    {
        "external_id": "PD009",
        "name": "Raquete TitanPadel Competition",
        "sport": "padel",
        "level": "competidor",
        "weight_g": 335,
        "balance": "alto (cabeça pesada)",
        "material": "carbono 24K toray + FOAM HR 65 Shore",
        "price_cents": 229900,
        "stock": 3,
        "description": (
            "Raquete de padel de competição de alto nível. "
            "Carbono Toray 24K de altíssima performance com perfil diamante extremo. "
            "Foam HR 65 Shore (muito dura) para explosividade máxima nos smashes e voleios. "
            "Peso de 335g reduzido para agilidade sem perder potência. "
            "Escolha de jogadores de nível nacional e internacional."
        ),
    },
    # ── Padel — acessório ─────────────────────────────────────────────────────
    {
        "external_id": "PD010",
        "name": "Bolsa ApexPadel Tour Pro",
        "sport": "padel",
        "level": None,
        "weight_g": None,
        "balance": None,
        "material": "nylon ripstop + couro sintético",
        "price_cents": 44900,
        "stock": 22,
        "description": (
            "Bolsa de padel estilo mochila com compartimento para 2 raquetes com divisória, "
            "bolso térmico para 2 garrafas e compartimento traseiro para roupas. "
            "Alças ergonômicas acolchoadas e reforço no fundo em couro sintético. "
            "Material nylon ripstop resistente à abrasão para uso intensivo em quadras e torneios."
        ),
    },
]


def _embedding_text(p: dict) -> str:
    parts = [p.get("name"), p.get("description"), p.get("sport"), p.get("level")]
    return " ".join(str(x) for x in parts if x)


async def main() -> None:
    logger.info("seed_catalog starting products=%d", len(PRODUCTS))

    texts = [_embedding_text(p) for p in PRODUCTS]

    t0 = time.perf_counter()
    embeddings = await embed_batch(texts)
    embed_ms = (time.perf_counter() - t0) * 1000
    logger.info("embeddings generated count=%d latency_ms=%.0f", len(embeddings), embed_ms)

    inserted = updated = 0

    async with get_session() as session:
        result = await session.execute(
            select(Product.external_id).where(
                Product.external_id.in_([p["external_id"] for p in PRODUCTS])
            )
        )
        existing_ids = {row[0] for row in result.all()}

        t1 = time.perf_counter()
        for product, embedding in zip(PRODUCTS, embeddings):
            stmt = (
                pg_insert(Product)
                .values(
                    id=uuid.uuid4(),
                    external_id=product["external_id"],
                    name=product["name"],
                    sport=product.get("sport"),
                    level=product.get("level"),
                    weight_g=product.get("weight_g"),
                    balance=product.get("balance"),
                    material=product.get("material"),
                    price_cents=product["price_cents"],
                    stock=product["stock"],
                    description=product.get("description"),
                    embedding=embedding,
                    is_active=True,
                )
                .on_conflict_do_update(
                    index_elements=["external_id"],
                    set_={
                        "name": product["name"],
                        "sport": product.get("sport"),
                        "level": product.get("level"),
                        "weight_g": product.get("weight_g"),
                        "balance": product.get("balance"),
                        "material": product.get("material"),
                        "price_cents": product["price_cents"],
                        "stock": product["stock"],
                        "description": product.get("description"),
                        "embedding": embedding,
                        "is_active": True,
                        "updated_at": func.now(),
                    },
                )
            )
            await session.execute(stmt)

            if product["external_id"] in existing_ids:
                updated += 1
            else:
                inserted += 1

        await session.commit()
        db_ms = (time.perf_counter() - t1) * 1000

    logger.info(
        "seed_catalog done inserted=%d updated=%d db_latency_ms=%.0f",
        inserted,
        updated,
        db_ms,
    )
    print(f"\nSeed concluído: {inserted} inseridos, {updated} atualizados")
    print(f"Tempo de embeddings: {embed_ms:.0f} ms")
    print(f"Tempo de escrita no DB: {db_ms:.0f} ms")


if __name__ == "__main__":
    asyncio.run(main())
