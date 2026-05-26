# beachtenis-agent

Agente conversacional WhatsApp para franquia de Beach Tennis / Padel.

Stack: FastAPI + LangGraph + Redis + Supabase (Postgres/pgvector) + Evolution API + Anthropic Claude.

## Pré-requisitos

- Python 3.11+
- Docker e Docker Compose (apenas para Redis local em dev)
- Conta no [Supabase](https://supabase.com) com projeto criado
- Chaves de API: Anthropic, Evolution API, Voyage AI

> **Aviso:** `docker-compose.yml` sobe somente Redis e Postgres para desenvolvimento local. Em produção, use instâncias gerenciadas (Railway, Supabase, Upstash etc.).

## Setup Local

```bash
# 1. Clonar e entrar no diretório
git clone <repo>
cd beachtenis-agent

# 2. Criar e ativar virtualenv
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# 3. Instalar dependências
pip install -e ".[dev]"

# 4. Configurar variáveis de ambiente
cp .env.example .env
# Editar .env com suas chaves (ver seção "Variáveis de Ambiente" abaixo)

# 5. Subir Redis local (apenas dev)
docker compose up -d

# 6. Aplicar migrações no Supabase (ver seção abaixo)

# 7. Iniciar o servidor
# Windows:
.venv/Scripts/python -m uvicorn app.main:app --reload
# Linux/Mac:
uvicorn app.main:app --reload
```

O servidor estará disponível em http://localhost:8000

- Health check: `GET /health`
- Docs interativas: `GET /docs`

## Setup do Supabase

O schema é gerenciado por migrations SQL em `supabase/migrations/`.

### Opção A — Supabase CLI (recomendado)

```bash
npm install -g supabase
supabase link --project-ref SEU_PROJECT_REF
supabase db push
```

### Opção B — Dashboard (sem CLI)

1. Acesse **Supabase Dashboard → SQL Editor**
2. Execute `supabase/migrations/0001_initial_schema.sql`
3. Execute `supabase/migrations/0002_search_function.sql`

### Configurar `DATABASE_URL`

Em **Project Settings → Database → Connection String**, selecione **Transaction** (porta **6543**):

```
DATABASE_URL=postgresql+asyncpg://postgres.PROJECT_REF:[password]@aws-0-REGION.pooler.supabase.com:6543/postgres
```

## Configurar Evolution API

1. Suba uma instância Evolution API (Docker ou serviço gerenciado)
2. Crie uma instância WhatsApp no painel e anote o nome (`EVOLUTION_INSTANCE`)
3. Configure o webhook apontando para:
   ```
   POST https://SEU_DOMINIO/webhook/whatsapp
   ```
   com o header `apikey: SEU_EVOLUTION_WEBHOOK_TOKEN`
4. Preencha no `.env`:
   ```
   EVOLUTION_API_URL=https://sua-evolution.exemplo.com
   EVOLUTION_API_KEY=chave-api-evolution
   EVOLUTION_INSTANCE=nome-da-instancia
   EVOLUTION_WEBHOOK_TOKEN=token-secreto-do-webhook
   ```

## Variáveis de Ambiente

| Variável                    | Descrição                                          |
|-----------------------------|----------------------------------------------------|
| `ANTHROPIC_API_KEY`         | Chave da API Anthropic                             |
| `OPENAI_API_KEY`            | Chave da API OpenAI                                |
| `EVOLUTION_API_URL`         | URL da instância Evolution API                     |
| `EVOLUTION_API_KEY`         | Chave da Evolution API                             |
| `EVOLUTION_INSTANCE`        | Nome da instância WhatsApp                         |
| `EVOLUTION_WEBHOOK_TOKEN`   | Token validado no header `apikey` do webhook       |
| `ADMIN_API_KEY`             | Chave para rotas `/admin/*` (header `X-Admin-Key`) |
| `REDIS_URL`                 | URL do Redis (`redis://localhost:6379/0` em dev)   |
| `SUPABASE_URL`              | URL do projeto Supabase                            |
| `SUPABASE_SERVICE_ROLE_KEY` | Chave service_role do Supabase                     |
| `DATABASE_URL`              | URI do Transaction Pooler (porta 6543)             |
| `EMBEDDING_API_KEY`         | Chave Voyage AI                                    |
| `PII_SALT`                  | Salt para hash HMAC-SHA256 dos telefones           |
| `CATALOG_SYNC_CRON`         | Cron do sync automático (default: `0 */6 * * *`)   |

## Rotas Admin

Todas as rotas `/admin/*` exigem o header `X-Admin-Key: <ADMIN_API_KEY>`.

| Método | Rota                    | Descrição                                     |
|--------|-------------------------|-----------------------------------------------|
| GET    | `/admin/leads`          | Lista leads (phone_hash + profile, sem PII)   |
| GET    | `/admin/leads/{hash}`   | Detalhe do lead + últimas 50 conversas        |
| POST   | `/admin/catalog/resync` | Dispara sync manual do catálogo (background)  |
| GET    | `/admin/audit`          | Busca audit logs com filtros opcionais        |

### Resync manual do catálogo

```bash
curl -X POST https://SEU_DOMINIO/admin/catalog/resync \
  -H "X-Admin-Key: $ADMIN_API_KEY"
# {"status":"accepted","detail":"Sync running in background"}
```

### Buscar audit logs com filtros

```bash
curl "https://SEU_DOMINIO/admin/audit?actor=webhook&action=process_message" \
  -H "X-Admin-Key: $ADMIN_API_KEY"
```

## Atender Requisição LGPD

### Direito ao esquecimento (Art. 18 VI) — exclusão

1. Verifique a identidade do titular (documento + selfie ou outro meio)
2. Execute a exclusão:
   ```bash
   curl -X DELETE https://SEU_DOMINIO/lgpd/lead \
     -H "Content-Type: application/json" \
     -d '{"phone": "5511987654321"}'
   ```
3. A API marca o lead como deletado, zera o conteúdo de todas as conversas (`[DELETED]`), remove a sessão do Redis e registra no audit log com `deleted: true`
4. Confirme ao titular por escrito

### Direito à portabilidade (Art. 18 II) — export

1. Verifique a identidade do titular
2. Exporte os dados:
   ```bash
   curl -X POST https://SEU_DOMINIO/lgpd/lead/export \
     -H "Content-Type: application/json" \
     -d '{"phone": "5511987654321"}'
   ```
3. Entregue o JSON retornado ao titular no prazo legal (15 dias)

> **Importante:** O sistema nunca armazena PII bruto. CPF, telefone, e-mail e CEP são
> substituídos por tokens (`[CPF]`, `[FONE]`, etc.) antes de qualquer persistência.
> O número de telefone é convertido em hash HMAC-SHA256 irreversível.

## Sync do Catálogo

O sync roda automaticamente conforme `CATALOG_SYNC_CRON` (default: a cada 6 horas). Para forçar manualmente via API, veja **Rotas Admin** acima.

Para rodar localmente sem o servidor:

```bash
.venv/Scripts/python -c "
import asyncio
from app.rag.ingestion import sync_catalog
print(asyncio.run(sync_catalog()))
"
```

## Testes

```bash
pytest tests/ -xvs
```

Cobertura por módulo:

- `security/pii_masker` — unitário completo (CPF, CEP, FONE, EMAIL, ENDERECO)
- `storage/redis_session` — TTL deslizante, hard cap, idempotência de mensagens
- `api/webhook` — auth, idempotência, PII masking, chamada Evolution
- `api/admin` — auth, listagem de leads, audit log
- `api/lgpd` — delete (soft-delete + zeragem), export, resiliência Redis
- `agent/graph` — triage, diagnose, recommend, faq, handoff
- `rag/retriever` — busca semântica com mock DB

## Linting

```bash
ruff check .
mypy app/
```
