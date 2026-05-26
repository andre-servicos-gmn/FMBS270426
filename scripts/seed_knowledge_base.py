"""Seed the knowledge_base table with FAQ and policy documents.

Usage (from project root, with .venv activated):
    python scripts/seed_knowledge_base.py

Requires OPENAI_API_KEY and DATABASE_URL in .env.
Edit the DOCUMENTS list below to match the real policies of the franchise.
"""
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import configure_logging, get_settings  # noqa: E402
from app.rag.knowledge_ingestion import upsert_documents  # noqa: E402

configure_logging(get_settings())
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Base de conhecimento — edite estes documentos com as políticas reais da loja
# ─────────────────────────────────────────────────────────────────────────────

DOCUMENTS: list[dict] = [

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
            "Para entregas em condomínios ou empresas, informe o nome do responsável pelo recebimento nos dados de entrega."
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
            "ou processaremos o reembolso integral, incluindo o frete pago. "
            "Em caso de dúvida se o dano é de fabricação ou uso, podemos solicitar o envio do produto "
            "para avaliação técnica antes de decidir."
        ),
    },

    # ── Pagamento ─────────────────────────────────────────────────────────────
    {
        "title": "Formas de pagamento aceitas",
        "category": "payment",
        "content": (
            "Aceitamos as seguintes formas de pagamento: "
            "Cartão de crédito (Visa, Mastercard, Elo, Amex) em até 12x sem juros para compras acima de R$ 200. "
            "Cartão de débito (Visa, Mastercard, Elo). "
            "PIX (confirmação instantânea — pedido processado na hora). "
            "Boleto bancário (prazo de compensação: 1 a 3 dias úteis — o pedido só é processado após confirmação). "
            "Compras acima de R$ 500 podem ser divididas em até 18x com juros da operadora do cartão."
        ),
    },
    {
        "title": "Desconto no PIX",
        "category": "payment",
        "content": (
            "Pagamentos via PIX têm 5% de desconto sobre o valor total do pedido. "
            "O desconto é aplicado automaticamente no checkout ao selecionar PIX como forma de pagamento. "
            "O pagamento deve ser feito em até 30 minutos — após esse prazo o pedido é cancelado automaticamente."
        ),
    },
    {
        "title": "Parcelamento e juros",
        "category": "payment",
        "content": (
            "Parcelamento sem juros: até 6x para qualquer valor, até 12x para compras acima de R$ 200. "
            "Parcelamento com juros da operadora (juros de 1,99% ao mês): de 13x a 18x para compras acima de R$ 500. "
            "O valor mínimo de cada parcela é R$ 30. "
            "As condições de parcelamento podem variar de acordo com a operadora do seu cartão."
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
            "Para urgências (problemas com entrega, defeito recém-recebido), "
            "envie uma mensagem mesmo fora do horário — verificamos as mensagens urgentes diariamente."
        ),
    },
    {
        "title": "Loja física",
        "category": "store",
        "content": (
            "Temos loja física onde você pode ver e testar os produtos antes de comprar. "
            "Na loja física também fazemos demonstrações de raquetes com profissional especializado. "
            "Consulte o endereço e horário de funcionamento da loja mais próxima no nosso site ou "
            "pergunte ao atendente para indicar a unidade mais próxima de você."
        ),
    },
    {
        "title": "Cupom de desconto e promoções",
        "category": "store",
        "content": (
            "Cupons de desconto podem ser inseridos no campo 'Cupom' durante o checkout antes de finalizar o pedido. "
            "Cada cupom é de uso único por CPF. Não é possível combinar dois cupons no mesmo pedido. "
            "Promoções relâmpago são anunciadas no nosso Instagram e WhatsApp — siga para não perder. "
            "Clientes que indicam amigos ganham R$ 30 de crédito quando o amigo realizar a primeira compra acima de R$ 150."
        ),
    },

    # ── FAQ geral ─────────────────────────────────────────────────────────────
    {
        "title": "Tamanho correto de grip (cabo da raquete)",
        "category": "faq",
        "content": (
            "O tamanho do grip da raquete é medido pela circunferência do cabo em polegadas (L1 a L5). "
            "Para iniciantes, o grip médio (L2 ou L3) é mais confortável e facilita os movimentos. "
            "Jogadores avançados costumam preferir grips menores (L1) para mais sensibilidade. "
            "Uma dica prática: segure a raquete como se fosse apertar a mão de alguém — "
            "deve sobrar cerca de 1 cm entre a ponta dos dedos e a base da palma. "
            "Dúvidas sobre o tamanho certo? Me fala o tamanho da sua mão e eu te oriento."
        ),
    },
    {
        "title": "Diferença entre raquete de beach tennis e padel",
        "category": "faq",
        "content": (
            "Beach tennis: raquete sem furos, face sólida de fibra de carbono ou vidro, "
            "mais leve (330–370g), projetada para golpes rápidos na areia. "
            "Padel: raquete com furos, mais pesada (360–400g), projetada para quadra coberta com paredes — "
            "o jogo usa as paredes como parte da estratégia. "
            "As bolas também são diferentes: beach tennis usa bolas de tênis comuns; "
            "padel usa bolas específicas com pressão mais baixa. "
            "Não é recomendado usar a raquete de um esporte no outro."
        ),
    },
    {
        "title": "Como escolher a primeira raquete",
        "category": "faq",
        "content": (
            "Para iniciantes, recomendamos raquetes com núcleo de EVA macio (35–45 Shore), "
            "pois absorvem melhor o impacto e facilitam o controle. "
            "Prefira face de fibra de vidro (mais barata e resistente a erros) em vez de carbono (mais cara e para avançados). "
            "Peso ideal para iniciantes: 350–380g para beach tennis, 360–390g para padel. "
            "Balanço equilibrado ou leve para o cabo dá mais controle — balanço para a ponta dá mais potência. "
            "Orçamento sugerido para uma boa primeira raquete: entre R$ 250 e R$ 450."
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
]


async def main() -> None:
    logger.info("Iniciando seed da knowledge_base com %d documentos...", len(DOCUMENTS))
    result = await upsert_documents(DOCUMENTS)
    logger.info(
        "Seed concluido: upserted=%d total=%d",
        result["upserted"],
        result["total"],
    )
    print(f"\nSeed concluido: {result['upserted']} documentos inseridos/atualizados.")


if __name__ == "__main__":
    asyncio.run(main())
