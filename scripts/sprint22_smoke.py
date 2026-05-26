"""Sprint 2.2 — smoke that produces the formatted dossier WhatsApp message
for each of the 4 handoff types.

External I/O (OpenAI, DB, Evolution) is mocked. The script prints the EXACT
string that would be sent via WhatsApp, so the user can paste it into the
sprint report. To do a LIVE end-to-end test, set DOSSIER_RECIPIENT_PHONE in
.env, then run scripts/chat.py and reproduce the 4 conversations.
"""
import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.dossier import (
    _clear_summary_cache,
    build_dossier,
    format_dossier_for_whatsapp,
)
from app.agent.state import AgentState


@asynccontextmanager
async def _mock_db():
    s = MagicMock()
    s.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    )
    s.commit = AsyncMock()
    yield s


def _state(reason: str, **extras) -> AgentState:
    base: AgentState = {  # type: ignore[typeddict-item]
        "messages": [
            HumanMessage(content="oi"),
            AIMessage(content="Oi! Qual seu nome?"),
            HumanMessage(content="Marcelo"),
        ],
        "phone_hash": "abc12345def67890",
        "intent": "handoff",
        "player_profile": {
            "nivel_jogo": "intermediário",
            "lesoes": "tendinite",
            "regiao_lesao": "cotovelo",
            "esporte_raquete_previo": "tênis",
            "modelo_desejado": "BeachPro Carbon X5",
        },
        "recommended_products": [],
        "needs_handoff": True,
        "handoff_reason": reason,
        "consultoria_interest": True,
        "customer_name": "Marcelo",
        "produto_pesquisado": "BeachPro Carbon X5",
    }
    base.update(extras)  # type: ignore[typeddict-item]
    return base


# ── Scenario data ────────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "label": "1. PURCHASE_CLOSING (quero comprar)",
        "reason": "purchase_closing",
        "summary": (
            "Cliente Marcelo chegou perguntando pela BeachPro Carbon X5. "
            "Confirmou estoque, perguntou preço e decidiu fechar a compra. "
            "Aguardando atendente humano pra finalizar a venda."
        ),
        "extras": {},
    },
    {
        "label": "2. SCHEDULING (quer agendar Consultoria)",
        "reason": "scheduling",
        "summary": (
            "Cliente Marcelo perguntou sobre a Consultoria Base Sports e em "
            "seguida pediu pra agendar. Aguardando atendente pra marcar o horário."
        ),
        "extras": {},
    },
    {
        "label": "3. OUT_OF_SCOPE (entregam em casa?)",
        "reason": "out_of_scope",
        "summary": (
            "Cliente Marcelo perguntou se a loja entrega em casa. "
            "Pergunta operacional fora do escopo do agente — atendente pode "
            "responder direto."
        ),
        "extras": {},
    },
    {
        "label": "4. SCHEDULING após bare_recommendation + perfil completo",
        "reason": "scheduling",
        "summary": (
            "Cliente Marcelo veio sem modelo em mente — completou o diagnóstico "
            "(intermediário, dor no cotovelo, vindo do tênis) e topou a "
            "Consultoria. Aguardando atendente pra agendar o horário em quadra."
        ),
        "extras": {
            "produto_pesquisado": None,
            "player_profile": {
                "nivel_jogo": "intermediário",
                "lesoes": "tendinite",
                "regiao_lesao": "cotovelo",
                "esporte_raquete_previo": "tênis",
                "modelo_desejado": "nenhum",  # bare_recommendation path
            },
        },
    },
]


async def main() -> None:
    _clear_summary_cache()
    for scn in SCENARIOS:
        state = _state(scn["reason"], **scn["extras"])
        dossier = build_dossier(state, summary=scn["summary"])
        # Inject purchase-specific extras for scenario 1.
        if scn["reason"] == "purchase_closing":
            dossier["produto_escolhido"] = "Raquete BeachPro Carbon X5"
            dossier["preco_cents"] = 89900

        rendered = format_dossier_for_whatsapp(dossier)
        print(f"\n##### CENÁRIO {scn['label']} #####\n")
        print(rendered)
        print("\n" + "─" * 60)


if __name__ == "__main__":
    asyncio.run(main())
