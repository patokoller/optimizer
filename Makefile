# ═══════════════════════════════════════════════════════════════════════
# Makefile — AI Portfolio Decision-Support Platform
# Run `make help` to see all available commands.
# ═══════════════════════════════════════════════════════════════════════

.PHONY: help dev down logs test test-fidelity test-api migrate deploy-frontend deploy-backend rollback-frontend secrets-check lint type-check build-frontend clean

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ── Config ─────────────────────────────────────────────────────────────
FRONTEND_DIR := frontend
BACKEND_DIR  := backend
COMPOSE      := docker compose

# ── Help ───────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║     AI Portfolio Decision-Support Platform               ║"
	@echo "║     Based on Cohen, Aiche & Eichel (2025), Entropy 550  ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  LOCAL DEVELOPMENT"
	@echo "    make dev               Start all services (API, worker, DB, Redis, frontend)"
	@echo "    make down              Stop all services"
	@echo "    make logs              Stream all logs"
	@echo "    make logs-api          Stream API logs only"
	@echo "    make logs-worker       Stream Celery worker logs"
	@echo ""
	@echo "  DATABASE"
	@echo "    make migrate           Run Alembic migrations"
	@echo "    make migrate-rollback  Downgrade one migration"
	@echo "    make db-shell          Open psql shell"
	@echo ""
	@echo "  TESTING"
	@echo "    make test              Run all backend tests"
	@echo "    make test-fidelity     Run benchmark fidelity tests only (fast)"
	@echo "    make test-api          Run API integration tests"
	@echo "    make lint              ESLint + ruff"
	@echo "    make type-check        TypeScript tsc + mypy"
	@echo ""
	@echo "  DEPLOYMENT (requires GitHub Actions secrets configured)"
	@echo "    make deploy            Trigger full deploy via GitHub Actions"
	@echo "    make deploy-frontend   Trigger frontend deploy only"
	@echo "    make deploy-backend    Trigger backend deploy only"
	@echo "    make rollback          Interactive rollback via GitHub Actions"
	@echo ""
	@echo "  SECRETS"
	@echo "    make secrets-check     Verify required env vars are set"
	@echo "    make secrets-push      Push .env secrets to Railway (requires Railway CLI)"
	@echo ""
	@echo "  UTILITIES"
	@echo "    make build-frontend    Build Next.js for production locally"
	@echo "    make clean             Remove Docker volumes, build artifacts"
	@echo ""

# ── Local development ──────────────────────────────────────────────────
dev:
	@echo "▶ Starting all services…"
	@cp -n .env.example .env 2>/dev/null || true
	$(COMPOSE) up --build

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

logs-api:
	$(COMPOSE) logs -f api

logs-worker:
	$(COMPOSE) logs -f celery_worker

# ── Database ───────────────────────────────────────────────────────────
migrate:
	@echo "▶ Running Alembic migrations…"
	$(COMPOSE) exec api alembic upgrade head

migrate-rollback:
	$(COMPOSE) exec api alembic downgrade -1

db-shell:
	$(COMPOSE) exec postgres psql -U portfolio portfolio_db

# ── Testing ────────────────────────────────────────────────────────────
test:
	@echo "▶ Running all backend tests…"
	$(COMPOSE) exec api pytest backend/tests/ -v --tb=short

test-fidelity:
	@echo "▶ Benchmark fidelity check (locked values — Table 1, Cohen et al. 2025)…"
	cd backend && python tests/test_benchmark_fidelity.py

test-api:
	@echo "▶ API integration tests…"
	$(COMPOSE) exec api pytest backend/tests/test_api.py -v

lint:
	@echo "▶ Frontend lint…"
	cd $(FRONTEND_DIR) && npm run lint
	@echo "▶ Backend lint (ruff)…"
	cd $(BACKEND_DIR) && ruff check app/ || echo "ruff not installed — pip install ruff"

type-check:
	@echo "▶ TypeScript type-check…"
	cd $(FRONTEND_DIR) && npm run type-check
	@echo "▶ Python mypy…"
	cd $(BACKEND_DIR) && mypy app/ --ignore-missing-imports || echo "mypy not installed"

# ── Build ──────────────────────────────────────────────────────────────
build-frontend:
	cd $(FRONTEND_DIR) && npm ci && npm run build

# ── Secrets ────────────────────────────────────────────────────────────
secrets-check:
	@echo "▶ Checking required environment variables…"
	@missing=0; \
	for var in ANTHROPIC_API_KEY ALPACA_API_KEY ALPACA_SECRET_KEY ALPHA_VANTAGE_API_KEY; do \
	  if [ -z "$${!var}" ]; then \
	    echo "  ❌ $$var is not set"; \
	    missing=$$((missing+1)); \
	  else \
	    echo "  ✅ $$var is set"; \
	  fi; \
	done; \
	if [ $$missing -gt 0 ]; then \
	  echo ""; \
	  echo "$$missing secret(s) missing. Copy .env.example → .env and fill in values."; \
	  exit 1; \
	else \
	  echo ""; \
	  echo "✅ All required secrets present."; \
	fi

secrets-push:
	@echo "▶ Pushing secrets to Railway…"
	@[ -f .env ] || (echo "❌ .env not found" && exit 1)
	railway variables set \
	  ANTHROPIC_API_KEY=$$(grep ANTHROPIC_API_KEY .env | cut -d= -f2) \
	  ALPACA_API_KEY=$$(grep ALPACA_API_KEY .env | cut -d= -f2) \
	  ALPACA_SECRET_KEY=$$(grep ALPACA_SECRET_KEY .env | cut -d= -f2) \
	  ALPHA_VANTAGE_API_KEY=$$(grep ALPHA_VANTAGE_API_KEY .env | cut -d= -f2) \
	  EDGAR_USER_AGENT=$$(grep EDGAR_USER_AGENT .env | cut -d= -f2-)

# ── GitHub Actions deployment triggers ────────────────────────────────
deploy:
	@echo "▶ Triggering full deploy via GitHub Actions…"
	gh workflow run deploy.yml \
	  --field target=all \
	  --field environment=production

deploy-frontend:
	gh workflow run deploy.yml --field target=frontend --field environment=production

deploy-backend:
	gh workflow run deploy.yml --field target=backend --field environment=production

rollback:
	@echo "▶ Interactive rollback…"
	@read -p "Target (frontend/backend/worker/all): " TARGET; \
	read -p "Reason: " REASON; \
	gh workflow run rollback.yml \
	  --field target=$$TARGET \
	  --field environment=production \
	  --field reason="$$REASON"
	@echo "✅ Rollback triggered."

# ── Cleanup ────────────────────────────────────────────────────────────
clean:
	$(COMPOSE) down -v --remove-orphans
	cd $(FRONTEND_DIR) && rm -rf .next node_modules 2>/dev/null || true
	find $(BACKEND_DIR) -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find $(BACKEND_DIR) -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Clean complete."
