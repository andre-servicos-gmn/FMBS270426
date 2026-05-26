# Configuração Bling — passo a passo (Andre)

Esse guia leva o agente de zero ao Bling conectado + sync inicial pronto.
Tempo estimado: **20–30 min** (mais ~10 min de espera no sync inicial).

---

## Pré-requisitos

- [ ] Conta Bling com plano que permite app privado OAuth 2.0
- [ ] App Bling já criado: **"Nouvaris - Atendente Virtual"**
  - Escopos read-only: Produtos, Estoque, Categorias, Depósitos
  - Client ID + Client Secret guardados em local seguro
- [ ] ngrok rodando: `ngrok http 8000`
- [ ] Supabase com migrações `0006_bling_credentials.sql` e `0007_bling_products.sql` aplicadas
- [ ] Redis rodando (local ou Cloud)

---

## Passo 1 — Preencher o `.env`

Abrir `.env` (NÃO o `.env.example`) e preencher:

```env
BLING_CLIENT_ID=<seu_client_id>
BLING_CLIENT_SECRET=<seu_client_secret>
BLING_REDIRECT_URI=https://<seu-ngrok>.ngrok-free.dev/bling/oauth/callback

# Já vem com defaults razoáveis — só ajusta se quiser:
BLING_SYNC_CATEGORIES=Raquetes de Praia,RAQUETE PADEL,Bola Beach TEnnis,GRIPS,UNDERGRIP,Anti Vibradores,RAQUETEIRAS MOCHILA,Camisetas e Camisas,Shorts,Short,Top,Saias,Camiseta Babylook,Calça Legging,Calças,Vestidos,Sapatilha,Bonés
BLING_SYNC_HOUR=4
BLING_STOCK_CACHE_TTL=300
```

**Importante:** o `BLING_REDIRECT_URI` precisa ser EXATAMENTE o mesmo cadastrado
no app do Bling. Se o subdomínio do ngrok mudou, atualizar nos dois lugares
(aqui e no painel Bling).

---

## Passo 2 — Aplicar migrações no Supabase

No painel do Supabase (SQL Editor), rodar:

1. `supabase/migrations/0006_bling_credentials.sql`
2. `supabase/migrations/0007_bling_products.sql`

Verificar que as tabelas existem:
- `bling_credentials`
- `bling_sync_logs`
- `bling_products`
- `bling_webhook_events`

---

## Passo 3 — Subir o app

```bash
.venv/Scripts/python -m uvicorn app.main:app --reload --port 8000
```

Confirmar nos logs:
- `legacy catalog sync disabled (no CATALOG_API_URL)` (esperado — não usamos mais)
- `bling_daily_sync registered (hour=4 UTC)` ← isso aqui é o que importa

---

## Passo 4 — Autorizar o app (UMA VEZ)

No navegador, abrir:

```
https://<seu-ngrok>.ngrok-free.dev/bling/oauth/authorize
```

Esperado:
1. Browser redireciona pro painel Bling
2. Bling pede confirmação ("Permitir acesso da app Nouvaris...")
3. Você autoriza
4. Bling redireciona de volta pra `/bling/oauth/callback?code=...`
5. Tela com **"✅ Bling conectado!"**

Se aparecer ❌, conferir:
- `.env` está com os valores certos
- `BLING_REDIRECT_URI` bate com o cadastrado no app do Bling
- ngrok não caiu

A credencial fica salva em `bling_credentials` (1 linha singleton). Não
precisa autorizar de novo — o refresh token é usado automaticamente.

---

## Passo 5 — Rodar o sync inicial

```bash
.venv/Scripts/python scripts/bling_initial_sync.py
```

Saída esperada (varia conforme o catálogo):

```
INFO bling_full_sync_done {'total_processed': 1240, 'inserted': 1200, 'updated': 0, 'skipped': 40, 'errors': 0}
─────────────────────────────────────────────
  total_processed: 1240
  inserted:        1200
  updated:         0
  skipped:         40
  errors:          0
  elapsed:         480.3s
─────────────────────────────────────────────
```

**Demora ~5–15 min pra ~1240 produtos** (Bling limita a 3 req/s). Cada produto exige 1 chamada de listagem + 1 chamada de detalhe.

Se der `BlingNotAuthorizedError`, voltar ao Passo 4.

---

## Passo 6 — Cadastrar webhook no Bling

No painel do Bling (Configurações → Webhooks), criar:

- **URL:** `https://<seu-ngrok>.ngrok-free.dev/bling/webhook`
- **Eventos:** produto criado, produto atualizado, produto excluído
- **Segredo HMAC:** clicar pra gerar, copiar valor

Colar o segredo no `.env`:

```env
BLING_WEBHOOK_SECRET=<segredo_gerado_pelo_bling>
```

Reiniciar o uvicorn. Pronto.

**Teste manual:** alterar um produto no painel do Bling e confirmar nos
logs do app:

```
INFO bling_webhook product.updated applied id=12345 outcome=updated
```

---

## Passo 7 — Confirmar que o sync diário roda

Sync automático às **04:00 UTC** (01:00 Brasília). Logs vão indicar:

```
INFO bling_daily_sync starting
INFO bling_daily_sync finished {'inserted': 0, 'updated': 12, 'skipped': 1228, 'errors': 0}
```

Histórico em `bling_sync_logs`:

```sql
SELECT kind, started_at, finished_at, inserted, updated, errors
FROM bling_sync_logs ORDER BY started_at DESC LIMIT 10;
```

---

## Troubleshooting

| Sintoma | Causa provável | Solução |
|---|---|---|
| `401` no callback | state expirou (5 min) ou redirect_uri diferente | refazer Passo 4 |
| Sync trava em "rate limited" | Bling devolveu 429 várias vezes | aguardar — o cliente faz backoff exponencial automaticamente |
| `dossier_send_skipped reason=no_recipient_configured` | DOSSIER_RECIPIENT_PHONE vazio | normal se não quiser receber dossiês |
| Webhook chega 401 | HMAC bate diferente | reconferir BLING_WEBHOOK_SECRET (sem espaços, sem aspas) |
| Produto novo não aparece no agente | sync diário só roda 1×/dia | rodar `bling_initial_sync.py` manualmente OU aguardar webhook |
| Estoque desatualizado por até 5 min | cache TTL = 300s | é o trade-off (vale o desempenho); reduzir BLING_STOCK_CACHE_TTL se quiser mais "tempo real" |

---

## Trocar instância (ex: piloto → produção)

Quando subir pra produção:

1. Criar **app novo** no Bling com domínio definitivo (não ngrok)
2. Atualizar `BLING_CLIENT_ID`, `BLING_CLIENT_SECRET`, `BLING_REDIRECT_URI` no `.env` de prod
3. Limpar a tabela `bling_credentials` (1 row) — a credencial antiga não vale mais
4. Refazer Passos 4 + 5 + 6 (autorizar, sync inicial, webhook)
