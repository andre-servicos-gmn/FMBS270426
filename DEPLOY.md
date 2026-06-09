# Deploy — easypanel (Docker) on Hostinger KVM2

Production checklist for shipping the Base Sports WhatsApp agent
alongside the Evolution API on the same VPS.

## What the container does

- FastAPI on `$PORT` (host `0.0.0.0`), entrypoint `app.main:app`.
- In-process APScheduler runs the Bling daily sync at 04:00 UTC.
- All credentials come from environment variables. **No secrets in
  the image.**

## Files involved

- `Dockerfile` — production image (python:3.11-slim, non-root, healthcheck).
- `.dockerignore` — blocks `.env`, dumps, tests, and other junk from the
  build context.
- `.env.example` — every env var the app reads, with inline docs.

## Build & run locally (smoke test before pushing to easypanel)

```bash
# Build
docker build -t base-sports-agent:dev .

# Run with a real .env (NOT mounted into the image — passed at runtime)
docker run --rm -p 8000:8000 --env-file .env base-sports-agent:dev

# Health check
curl -fsS http://localhost:8000/health
# → {"status":"ok","env":"development"}
```

## easypanel setup

1. **Create a new service** of type *App* (Dockerfile-based).
2. **Source**: connect the GitHub repo (private is fine — install the
   easypanel GitHub app).
3. **Build**: Dockerfile path = `Dockerfile` (root). No build args needed.
4. **Port**: 8000 (the `EXPOSE` declared in the Dockerfile; easypanel
   sets `$PORT` automatically and our CMD reads it).
5. **Environment variables**: paste from the list below.
6. **Healthcheck**: already in the Dockerfile, easypanel honours it.
7. **Persistent volumes**: NONE needed — every piece of state is in
   Redis Cloud or Supabase.
8. **Restart policy**: `always` (the scheduler must come back after VPS
   reboots).

## Webhook URL

Once easypanel assigns the public URL (e.g.
`https://base-sports-agent.<your-domain>` or the auto-generated
`*.easypanel.host`), configure it in the Evolution Manager:

- **Webhook URL**: `https://<your-easypanel-url>/webhook/whatsapp`
- **Custom Headers**: `apikey: <EVOLUTION_WEBHOOK_TOKEN value>`

(Evolution uses the same header name on outbound webhook calls as it
does for inbound API auth — see `app/api/webhook.py:_require_token`.)

## Env vars — copy/paste into easypanel

Generate every secret with `openssl rand -hex 32` unless noted.

```
# OpenAI
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini

# Evolution
EVOLUTION_API_URL=https://<evolution-host>/
EVOLUTION_API_KEY=
EVOLUTION_INSTANCE=
EVOLUTION_WEBHOOK_TOKEN=

# Supabase
SUPABASE_URL=https://<project_ref>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=
DATABASE_URL=postgresql+asyncpg://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:6543/postgres

# Redis Cloud
REDIS_URL=redis://<user>:<pass>@<host>:<port>
SESSION_TTL_SECONDS=86400
SESSION_HARD_CAP_SECONDS=604800

# Bling
BLING_CLIENT_ID=
BLING_CLIENT_SECRET=
BLING_REDIRECT_URI=https://<your-easypanel-url>/bling/oauth/callback
BLING_SYNC_CATEGORIES=Raquetes de Praia,RAQUETE PADEL,Bola Beach TEnnis,GRIPS,UNDERGRIP,Anti Vibradores,RAQUETEIRAS MOCHILA,Camisetas e Camisas,Shorts,Short,Top,Saias,Camiseta Babylook,Calça Legging,Calças,Vestidos,Sapatilha,Bonés
BLING_SYNC_HOUR=4
BLING_STOCK_CACHE_TTL=300
BLING_WEBHOOK_SECRET=

# Embeddings (uses OpenAI by default)
EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=

# Compliance
PII_MASK_ENABLED=true
PII_SALT=
LEAD_RETENTION_DAYS=365

# Operations
DOSSIER_RECIPIENT_PHONE=
RESET_ALLOWED_PHONES=
ADMIN_API_KEY=

# Loja física
STORE_NAME=Base Sports
STORE_ADDRESS=
STORE_HOURS=
STORE_MAPS_URL=
STORE_PHONE=

# Consultoria
CONSULTORIA_PRECO=350
CONSULTORIA_ENABLED=true

# App
APP_ENV=production
LOG_LEVEL=INFO
```

## Operational notes

- **Logs**: easypanel streams container stdout; grep for
  `webhook_auth_ok`, `webhook_auth_failed`, `reset_authorized`,
  `reset_denied`, `detail_choice_check` for the security-critical paths.
- **Reset**: `/reset` is denied for every phone unless you list it in
  `RESET_ALLOWED_PHONES` (CSV, internacional sem `+`).
- **Scheduler**: the in-process APScheduler restarts whenever the
  container restarts. If you scale to >1 replica, fix this first (the
  Bling sync would run N times per day).
- **Image hygiene**: every push to `main` rebuilds. The .dockerignore
  guarantees `.env` and debug dumps stay out.
