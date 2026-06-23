# syntax=docker/dockerfile:1.7
#
# FMBS270426 — Base Sports WhatsApp agent (FastAPI + LangGraph)
# Production container, suitable for easypanel on a Hostinger KVM2 VPS.
#
# Design notes:
#   - NO SECRETS in the image. Every credential (OpenAI, Supabase, Bling,
#     Redis, Evolution, ADMIN_API_KEY, RESET_ALLOWED_PHONES,
#     EVOLUTION_WEBHOOK_TOKEN, PII_SALT, etc.) is injected at runtime via
#     the platform's env-var panel. See .env.example for the full list and
#     DEPLOY.md for the easypanel walkthrough.
#   - PORT comes from $PORT at runtime (easypanel sets it). Falls back to
#     8000 for plain `docker run`.
#   - Long-running process: APScheduler runs in-process (Bling daily sync
#     at 04:00 UTC). The container must stay up; no serverless assumptions.
#   - Runs as a non-root user (uid 10001).

FROM python:3.11-slim AS runtime

# Reproducible-build env vars.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# System dependencies.
#   - curl            → used by HEALTHCHECK (and handy for ops debugging).
#   - ca-certificates → defensive; usually present, but TLS to Supabase /
#                       OpenAI / Evolution / Bling absolutely depends on it.
# We intentionally OMIT build-essential / gcc: every Python dep in
# pyproject.toml (asyncpg, uvloop, httptools, pgvector, pydantic-core, ...)
# ships pre-built cp311 manylinux wheels, so the build never falls back
# to source. If a future dep change breaks this, add `build-essential`
# here and remove it in a multi-stage refactor.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependency layer ────────────────────────────────────────────────────
# Copy pyproject.toml AND the app/ tree before `pip install .` because
# setuptools' `find` directive needs the package on disk to build the
# distribution. This means changes to app/ DO invalidate the deps layer —
# tolerable for a production image (rebuild ~2 min). If you start
# iterating heavily on this Dockerfile, switch to a generated
# requirements.txt or an editable install.
COPY pyproject.toml ./
COPY app ./app
RUN pip install --upgrade pip \
    && pip install .

# ── Application code (rest) ─────────────────────────────────────────────
# `scripts/` ships in the image so an operator can shell in and run admin
# tasks (e.g. bling_initial_sync, diagnose_attributes, reparse_attributes).
# Tests, supabase migrations, docs and local dumps are excluded by
# .dockerignore — they don't belong in a production image.
COPY scripts ./scripts

# ── Build marker ────────────────────────────────────────────────────────
# Stamp the git commit into the image so the running container logs WHICH
# build is live ("app_build sha=..." at startup). Pass --build-arg GIT_SHA=...
# from the deploy (EasyPanel can inject it). Defaults to "unknown" when unset.
ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA}

# ── Non-root user ───────────────────────────────────────────────────────
# uid 10001 avoids any clash with host uids on a multi-tenant VPS.
RUN useradd --create-home --shell /usr/sbin/nologin --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

# ── Network ─────────────────────────────────────────────────────────────
# EXPOSE is documentation only — it doesn't open ports. easypanel maps
# $PORT to whatever port it injects.
EXPOSE 8000

# ── Healthcheck ─────────────────────────────────────────────────────────
# Hits the /health route registered by app/main.py:55. easypanel's UI
# uses Docker's healthcheck signal to mark the container as healthy.
# --start-period gives the app time to run the lifespan handler (init
# checkpointer + DB connect + scheduler start).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8000}/health" || exit 1

# ── Entrypoint ──────────────────────────────────────────────────────────
# Shell-form CMD so ${PORT:-8000} expands at container start (Docker exec-
# form would treat $PORT as a literal). The leading `exec` replaces the
# `/bin/sh` process with uvicorn, so:
#   - Docker's SIGTERM reaches uvicorn directly (graceful shutdown of
#     in-flight requests + APScheduler).
#   - uvicorn becomes PID 1, no orphaned shell wrapper.
# --host 0.0.0.0 is mandatory: without it the container only binds to
# loopback and easypanel can't proxy traffic in.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
