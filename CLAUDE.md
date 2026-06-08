# CLAUDE.md

Guia operacional para o Claude Code trabalhar neste projeto.

## Contexto

Agente conversacional WhatsApp para franquia de Beach Tennis / Padel. Stack: FastAPI + LangGraph + Redis + Postgres/pgvector (Supabase) + Evolution API + OpenAI.

Este é um projeto da **Nouvaris**. Os padrões de código, segurança e estrutura seguem as convenções da Nouva (produto principal da Nouvaris) sempre que aplicável.

## Princípios

1. **Vibe coding com responsabilidade**: o usuário (Andre) não escreve código manualmente. Você implementa, executa, testa e corrige até o código rodar. Não pare em "tente rodar isso e me diga o resultado".
2. **Resolva bugs autonomamente**: ao encontrar erro de runtime, leia o stack trace, identifique causa, corrija e re-execute. Só envolva o usuário se estiver bloqueado por credencial, decisão de produto, ou após 3 tentativas falhas.
3. **Segurança não é negociável**: PII nunca vai pro provedor LLM sem mascaramento. Logs de auditoria nunca são opcionais.
4. **Código simples > código clever**: prefira soluções diretas. Sem abstrações prematuras.
5. **Português nas mensagens ao usuário, inglês no código**: docstrings, variáveis e logs em inglês; mensagens do agente ao cliente final em PT-BR.

## Stack e Versões

- Python 3.11
- FastAPI 0.115+
- LangGraph 0.2+ (com `MemorySaver`; troca por `RedisSaver` está no roadmap)
- OpenAI Python SDK (chat, embeddings, Whisper, vision)
- redis-py (asyncio)
- asyncpg + pgvector (via Supabase)
- SQLAlchemy 2.0
- pydantic v2 + pydantic-settings
- httpx (chamadas Evolution API)
- pytest + pytest-asyncio
- APScheduler (job de sync do catálogo)

## Modelos OpenAI em uso

- **Chat (triage, diagnose, recommend, faq, smalltalk, close):** `gpt-4o-mini`
- **Embeddings (catálogo + knowledge base):** `text-embedding-3-small` (1536d)
- **Transcrição de áudio (futuro Sprint 2):** `whisper-1`
- **Análise de imagem (futuro Sprint 2):** `gpt-4o` com input vision
- Sempre passar `system` prompt versionado de `prompts.py`
- `max_tokens` default 1024; respostas WhatsApp devem caber em 1–2 mensagens

## Estrutura de Diretórios

```
beachtenis-agent/
├── app/
│   ├── main.py                 # FastAPI entrypoint + lifespan
│   ├── config.py               # Settings via pydantic-settings
│   ├── agent/
│   │   ├── graph.py            # Construção do grafo LangGraph
│   │   ├── state.py            # TypedDict do estado
│   │   ├── nodes/
│   │   │   ├── triage.py
│   │   │   ├── diagnose.py
│   │   │   ├── recommend.py
│   │   │   ├── close.py
│   │   │   ├── faq.py
│   │   │   └── handoff.py
│   │   └── prompts.py          # Prompts versionados (PT-BR + guardrail anti-PII)
│   ├── adapters/
│   │   ├── evolution.py        # Cliente Evolution API (WhatsApp)
│   │   ├── openai_client.py    # Wrapper OpenAI com masking obrigatório de PII
│   │   └── catalog/
│   │       ├── api_source.py
│   │       └── file_source.py
│   ├── rag/
│   │   ├── embeddings.py
│   │   ├── ingestion.py        # Upsert + embedding do catálogo
│   │   ├── knowledge_ingestion.py
│   │   └── retriever.py        # search_products / search_knowledge_base
│   ├── security/
│   │   ├── pii_masker.py       # Regex CPF, CEP, telefone, email
│   │   └── audit_log.py
│   ├── storage/
│   │   ├── redis_session.py    # Sliding TTL + idempotência de mensagens
│   │   ├── db.py
│   │   └── models.py           # SQLAlchemy 2.0 declarative
│   ├── api/
│   │   ├── webhook.py          # POST /webhook/whatsapp
│   │   ├── admin.py            # Rotas admin (leads, audit)
│   │   └── lgpd.py             # DELETE /leads/{phone}
│   └── jobs/
│       └── catalog_sync.py     # APScheduler (só registra se CATALOG_API_URL setada)
├── tests/
├── scripts/
│   ├── chat.py                 # REPL local para testar o agente sem WhatsApp
│   ├── seed_via_rest.py        # Seed do catálogo via REST do Supabase
│   └── ...
├── .env.example
├── pyproject.toml
├── README.md
└── CLAUDE.md
```

## Convenções de Código

- **Async first**: todo I/O é async (FastAPI, httpx, asyncpg, redis-py asyncio, OpenAI SDK async)
- **Pydantic v2** pra validação e settings
- **Type hints obrigatórios** em funções públicas
- **Sem prints**: use `logging` configurado em `app/config.py`
- **Sem hardcode de strings** que viram prompts: tudo em `app/agent/prompts.py`
- **Tests em todo módulo de `security/` e `rag/`** — não opcional

## Variáveis de Ambiente (.env)

```
# OpenAI
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini

# Evolution API
EVOLUTION_API_URL=
EVOLUTION_API_KEY=
EVOLUTION_INSTANCE=
EVOLUTION_WEBHOOK_TOKEN=

# Admin
ADMIN_API_KEY=

# Redis (idempotência + futuro checkpoint)
REDIS_URL=redis://localhost:6379/0
SESSION_TTL_SECONDS=86400        # 24h sliding
SESSION_HARD_CAP_SECONDS=604800  # 7d

# Supabase (Postgres + pgvector)
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
DATABASE_URL=postgresql+asyncpg://...   # Transaction Pooler URI

# Catálogo
CATALOG_SOURCE=api               # api | file
CATALOG_API_URL=                 # se vazio, scheduler não registra o job
CATALOG_API_KEY=
CATALOG_FILE_PATH=
CATALOG_SYNC_CRON=0 */6 * * *    # a cada 6h

# Embeddings
EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=

# Compliance
PII_MASK_ENABLED=true
PII_SALT=change-me-in-production
LEAD_RETENTION_DAYS=365

# App
APP_ENV=development
LOG_LEVEL=INFO
```

## Regras Críticas de Segurança

### Mascaramento de PII (`app/security/pii_masker.py`)
**TODA** chamada à OpenAI passa pelo masker antes (centralizado em `OpenAIClient.chat`). Padrões mascarados:
- CPF: `\d{3}\.?\d{3}\.?\d{3}-?\d{2}` → `[CPF]`
- CEP: `\d{5}-?\d{3}` → `[CEP]`
- Telefone BR: `\(?\d{2}\)?\s?9?\d{4}-?\d{4}` → `[FONE]`
- Email: regex padrão → `[EMAIL]`
- Endereço (heurística): rua/av/r\. seguidos de número → `[ENDERECO]`

Em `APP_ENV=development` há uma defesa em profundidade: após mascarar, o cliente verifica com `is_clean()` e lança `ValueError` se algum padrão de PII sobreviveu — falha imediata em vez de leak silencioso.

O telefone do cliente que chega via WhatsApp é hashed (SHA256 com salt) antes de virar chave de sessão. Nunca aparece em prompt.

### Audit Log
Toda consulta a dados de cliente (lead, perfil, histórico) grava em `access_logs`:
- `actor` (sistema, vendedor X, admin Y)
- `action` (read_lead, read_session, export_data, delete_lead)
- `target_hash` (hash do telefone)
- `created_at` / `ip` quando aplicável

## Como Rodar Localmente

```bash
# Setup
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env  # preencher chaves

# Seed do catálogo (via REST do Supabase, sem precisar DATABASE_URL local)
.venv/Scripts/python scripts/seed_via_rest.py

# Rodar API
.venv/Scripts/python -m uvicorn app.main:app --reload --port 8000

# REPL local (testa o agente sem WhatsApp)
.venv/Scripts/python scripts/chat.py
```

## Workflow Esperado

Quando Andre pedir uma feature ou correção, você deve:

1. **Ler** os arquivos relevantes antes de editar
2. **Planejar** brevemente (1–3 linhas) o que vai mudar
3. **Implementar** com edits pontuais
4. **Rodar testes** se existirem para a área tocada (`pytest tests/test_X.py -xvs`)
5. **Rodar a aplicação** ou o módulo isolado pra validar
6. **Reportar** o que foi feito em PT-BR, conciso, sem bullet points desnecessários

Se um teste quebrar, **não pare**: corrija e rode de novo.

## O Que NÃO Fazer

- Não criar abstrações antes da terceira repetição
- Não adicionar dependência sem avisar (em PT-BR, no resumo final)
- Não commitar `.env` ou secrets
- Não usar `print` — só `logging`
- Não enviar PII pra OpenAI sem passar pelo masker
- Não pular o audit log em rotas que tocam dados de cliente
- Não usar emojis em código ou logs
- Não escrever respostas do agente em outro idioma que não PT-BR

## Observações sobre LangGraph

- Estado é `TypedDict` com `add_messages` reducer pra histórico
- Cada nó é função `async` que recebe `state` e retorna `dict` parcial
- Decisões de roteamento ficam em `conditional_edges` baseadas em `state["intent"]` (e em flags como `recommended_products` para distinguir primeira recomendação vs follow-up)
- Checkpointer: hoje `MemorySaver` (in-memory). Sprint 3 troca por `RedisSaver` (langgraph-checkpoint-redis) com namespace por hash do telefone

## Arquitetura conversacional (Sprint 2.6 — refatoração estratégica)

**Antes (≤ Sprint 2.5):** smalltalk → triage → diagnose (se cliente vago) → recommend → follow-ups.
**Depois (Sprint 2.6+):** smalltalk → triage → [recommend | price_inquiry | help_request | scheduling_inquiry | product_selection | faq | out_of_scope | smalltalk].

`diagnose` foi **removido do grafo do WhatsApp** porque canibalizava o valor da Consultoria presencial (R$350, abatido na compra). A regra estratégica agora é: o agente atende, responde, vende, convida pra loja. Quando o cliente precisa de orientação profunda, o agente oferece a Consultoria (que é onde o diagnóstico acontece, presencialmente).

**Os 9 intents do triage (Sprint 2.6):**
- `smalltalk` — cumprimento / nome / mensagens sem intenção comercial
- `product_inquiry` — pergunta sobre produto específico (estoque, características)
- `price_inquiry` — pergunta de preço
- `purchase_intent` — cliente quer comprar
- `scheduling_inquiry` — quer agendar a Consultoria
- `out_of_scope` — operacional fora do escopo (entrega, pix)
- `faq` — horário / localização / garantia
- `help_request` — pedido de ajuda GENÉRICA sem nomear produto → oferece Consultoria + visita à loja
- `close` — encerramento ("valeu, depois eu volto")

O arquivo `app/agent/nodes/diagnose.py` foi MANTIDO com docstring `DEPRECATED` — pra possível futura Consultoria virtual. Não importar dele em código ativo.

## Arquitetura do diagnose (Sprint 1.8 — DEPRECATED em 2.6, kept for reference)

O diagnose deixou de ser uma única chamada LLM "open-ended" e passou a ser um state-machine controlado por Python com extração e fraseamento delegados ao LLM. Cada turno passa por **4 fases**:

```
USER msg ─► is_meta_question(msg)? ──sim──► [LLM: SYSTEM_DIAGNOSE_META] ──► reply (re-ask)
                                  └──não──► FASE 1 — [LLM: SYSTEM_DIAGNOSE_EXTRACT] → extracted_slots
                                            FASE 2 — _apply_guardrails(merged)  (Python puro)
                                            FASE 3 — _next_pending_slot(merged) (Python puro)
                                                ├─ None → intent="recommend" (sem mensagem; recommend node responde)
                                                └─ slot → FASE 4 — [LLM: SYSTEM_DIAGNOSE_PHRASE] refraseia o molde
                                                          → reply
```

**Por que assim:** o LLM continua flexível para entender mensagens livres do cliente e dar tom natural à pergunta, mas **a ordem das perguntas é decidida em código** (`SLOT_ORDER` em `prompts.py`). Isso elimina drift de fluxo observado em produção (ordem trocada, guardrails ignorados).

**Pontos-chave:**
- `QUESTION_TEMPLATES` (em `prompts.py`) é a fonte única dos 5 moldes de perguntas. Nenhum slot que não esteja aqui é perguntado pelo agente.
- `_apply_guardrails()` pré-preenche slots determinísticos antes da Fase 3: `lesoes="nenhuma" → regiao_lesao="nenhuma"`; `nivel_jogo` intermediário/avançado (normalização Unicode) → `esporte_raquete_previo="nao_aplicavel"`.
- Fase 4 tem **fallback**: se a chamada LLM falhar, o texto canônico do molde é retornado direto. Sem regressão de UX em falha de rede.
- Meta-perguntas (`isso importa?`, `por que pergunta?`, …) são detectadas por substring case/accent-insensitive ANTES da Fase 1, então não consomem turno nem alteram o slot pendente.

## Estratégia "agente consultor light" (Sprint 1.5+)

O agente é **deliberadamente raso** no diagnóstico para preservar o valor da **Consultoria Base Esportes** (R$350 com teste em quadra). Decisões de design que decorrem disso:

- O diagnose pergunta **só 4 slots essenciais**: nível, lesão (+ região), esporte de raquete prévio, modelo desejado.
- Slots como `orcamento`, `frequencia_pratica`, `tempo_pratica`, `estilo_jogo`, `equipamento_atual` são **capturados se o cliente mencionar espontaneamente**, mas **NUNCA perguntados** pelo agente.
- O esporte default é `beach tennis`. O agente só confirma padel se o cliente sinalizar ("pala", "rolinho", "joguei padel"), e mesmo assim em uma única confirmação.
- O `recommend_node` apresenta **uma** raquete adequada ao perfil, usando linguagem calibrada ("é uma ótima raquete para esse perfil"), nunca "a perfeita". Toda recomendação termina com uma menção passageira (1-2 linhas) à Consultoria.
- Quando o cliente pergunta sobre a Consultoria (intent `consultoria`), o agente abre o pitch dedicado (`pitch_consultoria_node`) com valor `CONSULTORIA_PRECO` (default R$350) e flag `consultoria_interest=True` no estado.
- Toggle `CONSULTORIA_ENABLED=false` em franquias que não oferecem a Consultoria desativa tanto a menção quanto o pitch.

## Roadmap

> **ESCOPO.md é a fonte de verdade do roadmap.** Consultar antes de qualquer mudança de sprint. O resumo abaixo reflete a ordem definida lá; em caso de divergência, ESCOPO.md prevalece.

### Sprint 1 — MVP WhatsApp
- Migrar `MemorySaver` → `RedisSaver` (langgraph-checkpoint-redis) com TTL de 7 dias, mantendo `thread_id=phone_hash`
- Adicionar slots `regiao_lesao` e `modelo_desejado` no `player_profile`, com regras de coleta no `SYSTEM_DIAGNOSE` (pergunta de região condicionada a `lesoes != "nenhuma"`)
- Conectar Evolution API real: preencher `.env`, configurar webhook no painel apontando para o túnel ngrok
- CTA de fechamento com dados reais da loja (`STORE_NAME`, `STORE_ADDRESS`, `STORE_HOURS`, `STORE_MAPS_URL`, `STORE_PHONE`) injetados no contexto do `close_node`

### Sprint 2 — Catálogo flexível **(parcialmente implementada)**
- ✅ **Feito na Sprint 1.11** — coluna `category` adicionada à tabela `products` via `supabase/migrations/0004_add_category_to_products.sql` (com migration reversa `_down.sql`). Valores aceitos: `raquete | pala | bola | acessorio | vestuario | calcado | bolsa | outros`. Índice `idx_products_category` criado.
- ✅ **Feito na Sprint 1.11** — `search_products()` SQL function agora aceita `p_category` opcional (migration `0005`). `recommend.py._build_filters` pina automaticamente `raquete` (default) ou `pala` (quando padel) a partir do perfil. `re_recommendation` herda essa lógica e nunca mais retorna Kit Bolas/Bolsa quando o cliente pede uma raquete mais barata.
- ⏳ **Pendente** — `attributes JSONB` na tabela `products` (peso/balance/material como campos estruturados consultáveis no SQL, além das colunas legacy weight_g/balance/material). Vai junto com a Sprint 2 completa.
- ⏳ Pendente — refatoração de `_build_query` por categoria além de raquete/pala (ex: cliente pedindo bolsa, acessório).
- ⏳ Pendente — detector de categoria desejada no triage (cliente que diz "queria uma bolsa" deve sair do fluxo de diagnose de raquete).
- ⏳ Pendente — reseed do catálogo com variedade real (raquetes, bolas, vestuário, acessórios além dos 20 atuais).

### Sprint 3 — Mídia (foto + áudio) **(parcialmente implementada)**
- ✅ **Feito na Sprint 1.12** — `_classify_message` detecta audioMessage / imageMessage / documentMessage / stickerMessage / videoMessage no webhook.
- ✅ **Feito na Sprint 1.12** — `EvolutionClient.get_media_base64(message_key)` baixa mídia via `/chat/getBase64FromMediaMessage/<instance>` e retorna `(bytes, mimetype)`.
- ✅ **Feito na Sprint 1.12** — `app/adapters/media_processor.py` com `transcribe_audio()` usando Whisper `whisper-1`, `language="pt"`, timeout 30s, log estruturado por chamada (auditoria de custo).
- ✅ **Feito na Sprint 1.12** — Áudio transcrito vira `HumanMessage` SEM prefixo (decisão revisada do ESCOPO: prefixar pode confundir o agente, ele pode comentar sobre o áudio em vez de responder à intenção).
- ✅ **Feito na Sprint 1.12** — Imagem e documento: resposta canned, grafo não invocado. Sticker/vídeo: ignorados silenciosamente.
- ⏳ **Pendente** — `describe_image()` com GPT-4o vision (Sprint 1.13).
- ⏳ Pendente — Regra adicional no `SYSTEM_DIAGNOSE` para extrair `modelo_desejado` de foto de raquete.

**Limitações conhecidas do suporte a áudio (registrar antes de produção):**
- Sem rate limit por phone_hash — cliente pode mandar 100 áudios e cada um paga Whisper ($0.006/min).
- Sem cache de transcrições — áudio idêntico (mesmo hash) é re-transcrito.
- Sem limite de duração — WhatsApp permite áudio de até ~16min, custo dispara.
- Base64 transita pela rede inteira a cada áudio (sem streaming).
- Sem PII masking na transcrição: o texto vai pro grafo bruto e só é mascarado pelo `OpenAIClient.chat` antes da chamada LLM downstream.
