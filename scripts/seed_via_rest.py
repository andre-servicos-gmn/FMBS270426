"""Seed products and knowledge_base tables via Supabase REST API + OpenAI embeddings.

Does NOT require DATABASE_URL — uses Supabase PostgREST directly.
Usage (from project root, with .venv activated):
    python scripts/seed_via_rest.py

Requires: OPENAI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY in .env
Consumes ~$0.001 in OpenAI embeddings (30 texts x ~150 tokens each).
"""
import asyncio
import logging
import os
import sys
import time
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
_raw_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
REST_BASE = _raw_url if "/rest/v1" in _raw_url else _raw_url.replace(".co", ".co/rest/v1").replace("supabase.co", "supabase.co")
# Simpler: just ensure it ends at /rest/v1
if not _raw_url.endswith("/rest/v1") and "/rest/v1" not in _raw_url:
    # e.g. https://xxx.supabase.co  →  https://xxx.supabase.co/rest/v1
    REST_BASE = _raw_url.rstrip("/") + "/rest/v1"
else:
    REST_BASE = _raw_url.rstrip("/")

SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
EMBED_MODEL = "text-embedding-3-small"

if not OPENAI_KEY:
    sys.exit("ERROR: OPENAI_API_KEY not set in .env")
if not SUPABASE_KEY:
    sys.exit("ERROR: SUPABASE_SERVICE_ROLE_KEY not set in .env")

SUPA_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "resolution=merge-duplicates,return=minimal",
}

# ── Products ──────────────────────────────────────────────────────────────────

PRODUCTS: list[dict] = [
    # Beach Tennis — iniciante
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
    # Beach Tennis — intermediário
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
            "Núcleo em EVA de dureza média-alta (50 Shore) para jogadores intermediários."
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
            "Face em carbono 12K com acabamento texturizado para maior efeito na bola."
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
            "EVA de dureza 52 Shore para jogadores intermediários que precisam de mais controle."
        ),
    },
    # Beach Tennis — avançado
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
            "Frame completo em carbono 18K com EVA de alta dureza (58 Shore). "
            "Balanço alto para jogadores com swing completo e controlado."
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
            "Combinação de fibra de carbono HS com Texalium para máxima rigidez. "
            "Para jogadores avançados de beach tennis que buscam velocidade de bola "
            "aliada a uma sweet spot ampliada."
        ),
    },
    # Beach Tennis — competidor
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
            "Face em carbono Toray 24K, EVA 62 Shore para máxima transferência de energia. "
            "Homologada para torneios profissionais."
        ),
    },
    # Beach Tennis — acessórios
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
            "Bolsa para beach tennis com compartimento para 2 raquetes, bolso térmico. "
            "Material poliéster 600D impermeável, resistente à areia."
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
            "Feltro de alta durabilidade para uso na areia."
        ),
    },
    # Padel — iniciante
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
            "Raquete de padel para iniciantes com perfil redondo e balanço equilibrado. "
            "Face em fibra de vidro com núcleo EVA Soft (38 Shore) para absorver vibrações."
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
            "Peso reduzido de 340g facilita a mobilidade."
        ),
    },
    # Padel — intermediário
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
            "Perfil teardrop com face em carbono para boa mistura de potência e controle."
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
            "Construção híbrida com carbono 3K na face e fibra de vidro nas laterais. "
            "Balanço médio-alto oferece mais potência no smash sem perder precisão."
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
            "Foam HR amplia o efeito trampolim para smashes mais potentes."
        ),
    },
    # Padel — avançado
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
            "Raquete de padel avançada com perfil diamante para smashes devastadores. "
            "Carbono 12K com Foam HR (60 Shore) maximiza a velocidade de saída de bola."
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
            "Carbono unidirecional com reforço de kevlar nas bordas para resistência a impactos. "
            "Sweet spot centralizado e ampliado para jogadores avançados."
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
            "Face completa em carbono 18K para jogadores avançados que priorizam consistência. "
            "EVA 55 Shore para volleys e bandeja precisos."
        ),
    },
    # Padel — competidor
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
            "Carbono Toray 24K com Foam HR 65 Shore para explosividade máxima. "
            "Escolha de jogadores de nível nacional e internacional."
        ),
    },
    # Padel — acessório
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
            "Bolsa de padel mochila com compartimento para 2 raquetes, bolso térmico e roupas. "
            "Alças ergonômicas e fundo em couro sintético."
        ),
    },
]

# ── Knowledge-base docs summarising the product catalog ──────────────────────

KB_FAQ: list[dict] = [
    # ── Frete ─────────────────────────────────────────────────────────────────
    {
        "title": "Prazo e custo de entrega",
        "category": "shipping",
        "content": (
            "Entregamos para todo o Brasil via Correios (PAC e SEDEX) e transportadoras parceiras. "
            "Frete grátis para compras acima de R$ 299,00 para todo o Brasil. "
            "Para compras abaixo desse valor, o frete é calculado no checkout com base no CEP de destino. "
            "Prazo médio de entrega: 3 a 7 dias úteis para capitais (PAC) e 1 a 3 dias úteis (SEDEX). "
            "Para o interior, acrescente de 2 a 5 dias úteis dependendo da região. "
            "Após o envio, você recebe o código de rastreio por e-mail ou WhatsApp."
        ),
    },
    {
        "title": "Rastreamento do pedido",
        "category": "shipping",
        "content": (
            "Após a confirmação do pagamento, seu pedido é preparado em até 2 dias úteis e despachado. "
            "O código de rastreio é enviado por WhatsApp e e-mail assim que a transportadora coleta o pacote. "
            "Você pode acompanhar o status diretamente no site dos Correios (rastreamento.correios.com.br) "
            "ou no site da transportadora indicada no e-mail. "
            "Se não receber o código em 3 dias úteis após a compra, entre em contato conosco."
        ),
    },
    {
        "title": "Entrega em endereço diferente",
        "category": "shipping",
        "content": (
            "Você pode informar um endereço de entrega diferente do endereço de cobrança durante o checkout. "
            "Não fazemos entrega em caixas postais. "
            "Para entregas em condomínios ou empresas, informe o nome do responsável pelo recebimento."
        ),
    },
    {
        "title": "Entrega para fora do Brasil",
        "category": "shipping",
        "content": (
            "No momento não realizamos entregas internacionais. "
            "Vendemos apenas para endereços dentro do território brasileiro. "
            "Se você mora no exterior mas quer presentear alguém no Brasil, "
            "basta informar o endereço brasileiro como destino de entrega durante o checkout."
        ),
    },
    # ── Troca e devolução ─────────────────────────────────────────────────────
    {
        "title": "Política de troca e devolução",
        "category": "exchange",
        "content": (
            "Aceitamos trocas e devoluções em até 30 dias corridos após o recebimento do produto, "
            "conforme o Código de Defesa do Consumidor (Art. 49). "
            "O produto deve estar em perfeito estado, sem uso, com embalagem original e todos os acessórios. "
            "Para iniciar a troca ou devolução, entre em contato pelo WhatsApp com o número do pedido. "
            "O frete de retorno é por nossa conta quando o motivo for defeito de fabricação ou envio incorreto. "
            "Para trocas por tamanho ou preferência pessoal, o frete de retorno é responsabilidade do cliente. "
            "O reembolso é processado em até 5 dias úteis após recebermos o produto."
        ),
    },
    {
        "title": "Como solicitar troca ou devolução",
        "category": "exchange",
        "content": (
            "Para solicitar troca ou devolução: "
            "1. Entre em contato pelo WhatsApp informando o número do pedido e o motivo. "
            "2. Aguarde a aprovação (resposta em até 1 dia útil). "
            "3. Embale o produto com cuidado e envie para o endereço que informaremos. "
            "4. Após receber e conferir o produto, processamos a troca ou reembolso em até 5 dias úteis. "
            "Não aceitamos devoluções de produtos personalizados ou com uso evidente."
        ),
    },
    # ── Garantia ──────────────────────────────────────────────────────────────
    {
        "title": "Garantia dos produtos",
        "category": "warranty",
        "content": (
            "Todos os produtos têm garantia mínima de 90 dias contra defeitos de fabricação, "
            "conforme o CDC. Raquetes das marcas parceiras têm garantia estendida de 6 a 12 meses "
            "dependendo do fabricante — verifique a embalagem ou a descrição do produto. "
            "A garantia não cobre danos causados por mau uso, quedas, umidade excessiva "
            "ou desgaste natural de materiais como grip e overgrip. "
            "Para acionar a garantia, entre em contato pelo WhatsApp com foto do defeito e número do pedido."
        ),
    },
    {
        "title": "Defeito no produto recebido",
        "category": "warranty",
        "content": (
            "Se você recebeu um produto com defeito de fabricação, nos informe em até 7 dias corridos "
            "após o recebimento pelo WhatsApp. "
            "Envie fotos e vídeo mostrando o defeito. "
            "Após a análise (até 2 dias úteis), enviaremos um produto novo sem custo adicional "
            "ou processaremos o reembolso integral, incluindo o frete pago."
        ),
    },
    # ── Pagamento ─────────────────────────────────────────────────────────────
    {
        "title": "Formas de pagamento aceitas",
        "category": "payment",
        "content": (
            "Aceitamos: Cartão de crédito (Visa, Mastercard, Elo, Amex) em até 12x sem juros para compras acima de R$ 200. "
            "Cartão de débito. PIX (confirmação instantânea). "
            "Boleto bancário (prazo de compensação: 1 a 3 dias úteis). "
            "Compras acima de R$ 500 podem ser divididas em até 18x com juros da operadora."
        ),
    },
    {
        "title": "Desconto no PIX",
        "category": "payment",
        "content": (
            "Pagamentos via PIX têm 5% de desconto sobre o valor total do pedido. "
            "O desconto é aplicado automaticamente no checkout ao selecionar PIX. "
            "O pagamento deve ser feito em até 30 minutos — após esse prazo o pedido é cancelado."
        ),
    },
    {
        "title": "Parcelamento e juros",
        "category": "payment",
        "content": (
            "Parcelamento sem juros: até 6x para qualquer valor, até 12x para compras acima de R$ 200. "
            "Parcelamento com juros da operadora (1,99% ao mês): de 13x a 18x para compras acima de R$ 500. "
            "Valor mínimo de cada parcela: R$ 30."
        ),
    },
    # ── Loja e atendimento ────────────────────────────────────────────────────
    {
        "title": "Horário de atendimento",
        "category": "store",
        "content": (
            "Nosso atendimento pelo WhatsApp funciona de segunda a sexta das 8h às 18h "
            "e aos sábados das 9h às 13h. "
            "Fora desse horário, você pode deixar sua mensagem que respondemos no próximo dia útil. "
            "Para urgências, envie uma mensagem mesmo fora do horário — verificamos diariamente."
        ),
    },
    {
        "title": "Loja física",
        "category": "store",
        "content": (
            "Temos loja física onde você pode ver e testar os produtos antes de comprar. "
            "Fazemos demonstrações de raquetes com profissional especializado. "
            "Consulte o endereço e horário no nosso site ou pergunte ao atendente."
        ),
    },
    {
        "title": "Cupom de desconto e promoções",
        "category": "store",
        "content": (
            "Cupons de desconto são inseridos no campo 'Cupom' durante o checkout. "
            "Cada cupom é de uso único por CPF e não é possível combinar dois cupons. "
            "Promoções relâmpago são anunciadas no Instagram e WhatsApp. "
            "Clientes que indicam amigos ganham R$ 30 de crédito quando o amigo comprar acima de R$ 150."
        ),
    },
    # ── FAQ técnico ───────────────────────────────────────────────────────────
    {
        "title": "Tamanho correto de grip (cabo da raquete)",
        "category": "faq",
        "content": (
            "O tamanho do grip da raquete é medido pela circunferência do cabo em polegadas (L1 a L5). "
            "Para iniciantes, o grip médio (L2 ou L3) é mais confortável. "
            "Jogadores avançados costumam preferir grips menores (L1) para mais sensibilidade. "
            "Dica: segure a raquete como se fosse apertar a mão de alguém — "
            "deve sobrar cerca de 1 cm entre a ponta dos dedos e a base da palma."
        ),
    },
    {
        "title": "Diferença entre raquete de beach tennis e padel",
        "category": "faq",
        "content": (
            "Beach tennis: raquete sem furos, face sólida de fibra de carbono ou vidro, "
            "mais leve (330–370g), projetada para golpes rápidos na areia. "
            "Padel: raquete com furos, mais pesada (360–400g), projetada para quadra coberta com paredes. "
            "As bolas também são diferentes: beach tennis usa bolas de tênis comuns; "
            "padel usa bolas específicas com pressão mais baixa."
        ),
    },
    {
        "title": "Como escolher a primeira raquete",
        "category": "faq",
        "content": (
            "Para iniciantes, recomendamos raquetes com núcleo de EVA macio (35–45 Shore). "
            "Prefira face de fibra de vidro em vez de carbono. "
            "Peso ideal para iniciantes: 350–380g para beach tennis, 360–390g para padel. "
            "Balanço equilibrado ou leve para o cabo dá mais controle. "
            "Orçamento sugerido para uma boa primeira raquete: entre R$ 250 e R$ 450."
        ),
    },
]

KB_PRODUCTS: list[dict] = [
    {
        "title": "Raquetes de beach tennis para iniciantes disponíveis",
        "category": "products",
        "content": (
            "Temos duas raquetes de beach tennis para iniciantes: "
            "AirBlast Starter BT (R$ 349,00, 370g, fibra de vidro, EVA 45 Shore, balanço equilibrado) "
            "e BeachPro Foam Series 300 (R$ 299,00, 385g, polipropileno + EVA 40 Shore, ótimo custo-benefício). "
            "Ambas são ideais para quem está começando e precisa de controle e absorção de impacto."
        ),
    },
    {
        "title": "Raquetes de beach tennis para nível intermediário",
        "category": "products",
        "content": (
            "Para jogadores intermediários de beach tennis temos três modelos: "
            "BeachPro Carbon X5 (R$ 899,00, 355g, carbono 3K, balanço equilibrado), "
            "NovaSport Speed BT (R$ 699,00, 345g, carbono 12K, balanço para cabeça — mais ofensiva) e "
            "AirBlast Carbon Pro (R$ 999,00, 360g, carbono unidirecional — mais controle). "
            "A escolha depende do estilo de jogo: ofensivo ou defensivo."
        ),
    },
    {
        "title": "Raquetes de beach tennis para jogadores avançados e competidores",
        "category": "products",
        "content": (
            "Para nível avançado: VertexBT Pro Elite (R$ 1.799,00, 330g, carbono 18K, cabeça pesada) "
            "e SpeedLine Ultra BT (R$ 1.299,00, 340g, carbono HS + Texalium, sweet spot ampliado). "
            "Para competição profissional: ForceX Competition BT (R$ 2.499,00, 318g, carbono Toray 24K, "
            "homologada para torneios). "
            "Preços mais altos refletem tecnologia de ponta e uso em campeonatos."
        ),
    },
    {
        "title": "Acessórios de beach tennis: bolsas e bolas",
        "category": "products",
        "content": (
            "Acessórios de beach tennis disponíveis: "
            "Bolsa BeachPro Tournament Bag (R$ 399,00, poliéster 600D impermeável, comporta 2 raquetes, bolso térmico) e "
            "Kit Bolas BeachPro Pressurized x3 (R$ 89,00, bolas certificadas ITF, feltro durável para areia). "
            "A bolsa é resistente à areia e tem alça de ombro ajustável."
        ),
    },
    {
        "title": "Raquetes de padel para iniciantes disponíveis",
        "category": "products",
        "content": (
            "Para quem quer começar no padel temos: "
            "PadelMax Control Entry (R$ 379,00, 365g, fibra de vidro + EVA Soft 38 Shore, perfil redondo) — "
            "excelente para controle e absorção de vibrações. "
            "TitanPadel Junior (R$ 329,00, 340g, polipropileno + EVA 40 Shore) — mais leve, "
            "ideal para jovens e iniciantes que precisam de manobrabilidade."
        ),
    },
    {
        "title": "Raquetes de padel para nível intermediário",
        "category": "products",
        "content": (
            "Para padel intermediário temos três opções: "
            "NovaSport Padel Mid (R$ 749,00, 355g, carbono + EVA 50 Shore, perfil teardrop — versátil), "
            "ApexPadel Hybrid Mid (R$ 1.099,00, 360g, carbono 3K + vidro híbrido, balanço médio-alto) e "
            "GridPadel Power Mid (R$ 899,00, 370g, carbono + foam HR, perfil diamante — mais ofensivo). "
            "A escolha depende do estilo: controle, equilíbrio ou potência."
        ),
    },
    {
        "title": "Raquetes de padel para jogadores avançados e competidores",
        "category": "products",
        "content": (
            "Para padel avançado: "
            "ApexPadel Power Pro (R$ 1.599,00, 345g, carbono 12K + Foam HR 60 Shore, diamante, cabeça pesada), "
            "SpeedLine Padel Pro (R$ 1.349,00, 350g, carbono UD + kevlar, médio-alto) e "
            "PadelMax Titan Advanced (R$ 1.499,00, 355g, carbono 18K, equilibrado e consistente). "
            "Para competição: TitanPadel Competition (R$ 2.299,00, 335g, Toray 24K + Foam HR 65 Shore)."
        ),
    },
    {
        "title": "Preços e faixas de raquetes disponíveis",
        "category": "products",
        "content": (
            "Temos raquetes para todos os orçamentos: "
            "Iniciante (R$ 299 a R$ 389): BeachPro Foam 300, AirBlast Starter BT, PadelMax Control Entry, TitanPadel Junior. "
            "Intermediário (R$ 699 a R$ 1.099): NovaSport Speed BT, BeachPro Carbon X5, AirBlast Carbon Pro, e modelos de padel. "
            "Avançado (R$ 1.299 a R$ 1.799): VertexBT Pro Elite, SpeedLine Ultra BT, ApexPadel Power Pro, SpeedLine Padel Pro. "
            "Competição (R$ 2.299 a R$ 2.499): ForceX Competition BT e TitanPadel Competition."
        ),
    },
]


# ── Embedding ─────────────────────────────────────────────────────────────────


async def embed_texts(client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
    resp = await client.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
        json={"model": EMBED_MODEL, "input": texts},
        timeout=60.0,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    ordered = sorted(data, key=lambda e: e["index"])
    return [e["embedding"] for e in ordered]


# ── Supabase upsert helpers ───────────────────────────────────────────────────


async def upsert_products(
    client: httpx.AsyncClient, rows: list[dict]
) -> tuple[int, list[str]]:
    resp = await client.post(
        f"{REST_BASE}/products?on_conflict=external_id",
        headers=SUPA_HEADERS,
        json=rows,
        timeout=60.0,
    )
    if resp.status_code not in (200, 201):
        logger.error("products upsert failed %d: %s", resp.status_code, resp.text[:400])
        return 0, []
    return len(rows), [r["external_id"] for r in rows]


async def upsert_kb(client: httpx.AsyncClient, rows: list[dict]) -> int:
    resp = await client.post(
        f"{REST_BASE}/knowledge_base?on_conflict=title%2Ccategory",
        headers=SUPA_HEADERS,
        json=rows,
        timeout=60.0,
    )
    if resp.status_code not in (200, 201):
        logger.error("knowledge_base upsert failed %d: %s", resp.status_code, resp.text[:400])
        return 0
    return len(rows)


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    async with httpx.AsyncClient() as client:
        # ── 1. Embed and upsert products ──────────────────────────────────────
        logger.info("Generating embeddings for %d products...", len(PRODUCTS))
        t0 = time.perf_counter()
        product_texts = [
            f"{p['name']} {p.get('sport', '')} {p.get('level', '')} {p.get('description', '')}"
            for p in PRODUCTS
        ]
        product_embeddings = await embed_texts(client, product_texts)
        logger.info("Product embeddings done in %.0f ms", (time.perf_counter() - t0) * 1000)

        product_rows = []
        for p, emb in zip(PRODUCTS, product_embeddings):
            row = {
                "id": str(uuid.uuid4()),
                "external_id": p["external_id"],
                "name": p["name"],
                "sport": p.get("sport"),
                "level": p.get("level"),
                "weight_g": p.get("weight_g"),
                "balance": p.get("balance"),
                "material": p.get("material"),
                "price_cents": p["price_cents"],
                "stock": p["stock"],
                "description": p.get("description"),
                "embedding": emb,
                "is_active": True,
            }
            product_rows.append(row)

        t1 = time.perf_counter()
        count, ids = await upsert_products(client, product_rows)
        logger.info(
            "Products upserted count=%d in %.0f ms", count, (time.perf_counter() - t1) * 1000
        )

        # ── 2. Embed and upsert knowledge base docs ───────────────────────────
        logger.info("Generating embeddings for %d KB docs...", len(KB_PRODUCTS))
        t2 = time.perf_counter()
        kb_texts = [f"{d['title']}\n\n{d['content']}" for d in KB_PRODUCTS]
        kb_embeddings = await embed_texts(client, kb_texts)
        logger.info("KB embeddings done in %.0f ms", (time.perf_counter() - t2) * 1000)

        kb_rows = []
        for doc, emb in zip(KB_PRODUCTS, kb_embeddings):
            kb_rows.append(
                {
                    "id": str(uuid.uuid4()),
                    "title": doc["title"],
                    "content": doc["content"],
                    "category": doc["category"],
                    "source": "seed_script",
                    "embedding": emb,
                    "is_active": True,
                }
            )

        t3 = time.perf_counter()
        kb_count = await upsert_kb(client, kb_rows)
        logger.info(
            "KB product docs upserted count=%d in %.0f ms",
            kb_count,
            (time.perf_counter() - t3) * 1000,
        )

        # ── 3. Embed and upsert FAQ knowledge base docs ───────────────────────
        all_faq = KB_FAQ
        logger.info("Generating embeddings for %d FAQ docs...", len(all_faq))
        t4 = time.perf_counter()
        faq_texts = [f"{d['title']}\n\n{d['content']}" for d in all_faq]
        faq_embeddings = await embed_texts(client, faq_texts)
        logger.info("FAQ embeddings done in %.0f ms", (time.perf_counter() - t4) * 1000)

        faq_rows = [
            {
                "id": str(uuid.uuid4()),
                "title": d["title"],
                "content": d["content"],
                "category": d["category"],
                "source": "seed_script",
                "embedding": emb,
                "is_active": True,
            }
            for d, emb in zip(all_faq, faq_embeddings)
        ]

        t5 = time.perf_counter()
        faq_count = await upsert_kb(client, faq_rows)
        logger.info(
            "FAQ docs upserted count=%d in %.0f ms",
            faq_count,
            (time.perf_counter() - t5) * 1000,
        )

    total_ms = (time.perf_counter() - t0) * 1000
    print(f"\nSeed concluido:")
    print(f"  Produtos inseridos/atualizados: {count}")
    print(f"  Docs de produtos na KB:         {kb_count}")
    print(f"  Docs de FAQ/politicas na KB:    {faq_count}")
    print(f"  Tempo total: {total_ms:.0f} ms")


if __name__ == "__main__":
    asyncio.run(main())
