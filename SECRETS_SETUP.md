# GitHub Secrets & Repository Setup Guide
## AI Portfolio Decision-Support Platform

Run these steps once after creating the GitHub repo and Vercel/Railway projects.
After this, every `git push` to `main` deploys automatically — no manual steps.

---

## 1. Create GitHub repository

```bash
cd platform/
git init
git add .
git commit -m "feat: initial full-stack platform"
gh repo create ai-portfolio-platform --private --source=. --push
```

---

## 2. Vercel — frontend

1. Go to https://vercel.com/new → Import the GitHub repo
2. Set **Root Directory** → `frontend`
3. Set **Framework Preset** → Next.js
4. Add environment variable: `NEXT_PUBLIC_API_URL` = `https://your-railway-backend.railway.app`
5. Deploy once manually to get `VERCEL_PROJECT_ID` and `VERCEL_ORG_ID`

Get the IDs:
```bash
cd frontend
vercel link       # follow prompts; creates .vercel/project.json
cat .vercel/project.json
# → {"orgId": "...", "projectId": "..."}
```

Get a token: https://vercel.com/account/tokens → New Token → "GitHub Actions"

Get your production alias (e.g. `ai-portfolio.vercel.app`) from the Vercel dashboard.

---

## 3. Railway — backend

1. Go to https://railway.app/new → Deploy from GitHub repo
2. Create **two services**:
   - **api** (web): start command = `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **worker**: start command = `celery -A app.workers.tasks.celery_app worker --loglevel=info`
3. Add plugins: **PostgreSQL** and **Redis** (Railway injects `DATABASE_URL` and `REDIS_URL` automatically)
4. Note the Project ID and both Service IDs from the Railway dashboard URLs

Get a Railway token: https://railway.app/account/tokens → New Token

---

## 4. Set GitHub Secrets

Go to: **GitHub repo → Settings → Secrets and variables → Actions → New repository secret**

Add all of the following:

### Vercel secrets
| Secret name | Where to get it |
|---|---|
| `VERCEL_TOKEN` | Vercel account → Tokens |
| `VERCEL_ORG_ID` | `.vercel/project.json` → `orgId` |
| `VERCEL_PROJECT_ID` | `.vercel/project.json` → `projectId` |
| `VERCEL_PROD_ALIAS` | e.g. `ai-portfolio.vercel.app` |
| `NEXT_PUBLIC_API_URL` | Railway API service URL |

### Railway secrets
| Secret name | Where to get it |
|---|---|
| `RAILWAY_TOKEN` | Railway account → Tokens |
| `RAILWAY_PROJECT_ID` | Railway project URL: `railway.app/project/{ID}` |
| `RAILWAY_API_SERVICE_ID` | Railway service URL for API service |
| `RAILWAY_WORKER_SERVICE_ID` | Railway service URL for Celery worker |

### Application secrets
| Secret name | Where to get it |
|---|---|
| `ANTHROPIC_API_KEY` | https://console.anthropic.com → API Keys |
| `ALPACA_API_KEY` | https://app.alpaca.markets → Paper or Live API |
| `ALPACA_SECRET_KEY` | Same as above |
| `ALPHA_VANTAGE_API_KEY` | https://www.alphavantage.co/support/#api-key |

### CLI one-liner (if you have `gh` installed)
```bash
gh secret set VERCEL_TOKEN          --body "your-token"
gh secret set VERCEL_ORG_ID        --body "your-org-id"
gh secret set VERCEL_PROJECT_ID    --body "your-project-id"
gh secret set VERCEL_PROD_ALIAS    --body "ai-portfolio.vercel.app"
gh secret set NEXT_PUBLIC_API_URL  --body "https://api.railway.app"
gh secret set RAILWAY_TOKEN        --body "your-railway-token"
gh secret set RAILWAY_PROJECT_ID   --body "your-project-id"
gh secret set RAILWAY_API_SERVICE_ID    --body "your-api-service-id"
gh secret set RAILWAY_WORKER_SERVICE_ID --body "your-worker-service-id"
gh secret set ANTHROPIC_API_KEY    --body "sk-ant-..."
gh secret set ALPACA_API_KEY       --body "your-key"
gh secret set ALPACA_SECRET_KEY    --body "your-secret"
gh secret set ALPHA_VANTAGE_API_KEY --body "your-key"
```

---

## 5. Verify the pipeline

After setting all secrets:
```bash
# Trigger a full deploy manually
make deploy

# Or push to main (triggers automatically)
git push origin main
```

Watch the pipeline at: **GitHub repo → Actions**

Expected sequence:
1. `CI` runs — lint, type-check, pytest, fidelity tests
2. `Deploy` runs (only if CI passes):
   - `deploy-frontend` → Vercel build + deploy
   - `deploy-backend` → Railway redeploy + health check + migrations
   - `deploy-worker` → Railway Celery worker redeploy

---

## 6. Rollback

Via `make`:
```bash
make rollback
# → prompts for target and reason, triggers GitHub Actions rollback workflow
```

Via GitHub Actions UI:
1. Go to **Actions → Rollback → Run workflow**
2. Select target and environment
3. Enter reason (logged)

---

## 7. Phase 3 — Fly.io migration

When adding the Deep RL optimizer (Phase 3):

```bash
# Install flyctl
brew install flyctl   # or curl -L https://fly.io/install.sh | sh

# Set up Fly.io
flyctl auth login
flyctl launch --dockerfile Dockerfile --name ai-portfolio-platform

# Attach databases
fly postgres create --name portfolio-db
fly postgres attach --app ai-portfolio-platform portfolio-db
fly redis create --name portfolio-redis

# Set secrets
fly secrets set \
  ANTHROPIC_API_KEY="sk-ant-..." \
  ALPACA_API_KEY="..." \
  ALPACA_SECRET_KEY="..." \
  ALPHA_VANTAGE_API_KEY="..."

# Add GitHub secret for CI
gh secret set FLY_API_TOKEN   --body "$(flyctl auth token)"
gh secret set FLY_APP_URL     --body "https://ai-portfolio-platform.fly.dev"

# Trigger Phase 3 deploy
gh workflow run deploy-flyio.yml \
  --field environment=production \
  --field service=all
```

---

## 8. Railway app variables (set once, not in GitHub Secrets)

These go in Railway dashboard → Service → Variables (not in GitHub):

```
EDGAR_USER_AGENT=ai-portfolio-platform research@yourcompany.com
```

Railway auto-injects `DATABASE_URL` and `REDIS_URL` from its plugins.
