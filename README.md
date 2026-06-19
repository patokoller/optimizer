# AI Portfolio Decision-Support Platform

> Based on: **Cohen, Aiche & Eichel (2025)**, *Entropy 27, 550*
> "AI-Driven Portfolio Optimization: Integrating Machine Learning and Large Language Models for Enhanced Decision-Making"

----

## Locked Benchmark Facts

Source: **Table 1, Cohen et al. (2025)** — DO NOT MODIFY

| Strategy | Freq | ML Weight (w) | LLM Weight | Sharpe | Avg Return | Volatility | Cumulative Return |
|---|---|---|---|---|---|---|---|
| Technical | Monthly | **1.00** | 0.00 | 0.6934 | 7.50% | 10.82% | **1977.71%** ← Best Cumul. |
| Entropy | Monthly | 0.70 | 0.30 | 0.4207 | 5.23% | 12.44% | 700.52% |
| Fundamental | Monthly | 0.15 | 0.85 | 0.5001 | 4.32% | **8.63%** | 578.40% |
| Technical | Quarterly | 0.45 | 0.55 | **1.2967** | 24.99% | 19.27% | 573.37% ← Best Sharpe |
| Entropy | Quarterly | 0.40 | 0.60 | 0.6048 | 20.25% | 33.48% | 534.36% |
| Fundamental | Quarterly | 0.00 | **1.00** | 0.4899 | 14.71% | 30.02% | 326.12% |

> **Disclaimer:** Backtested Jan 2020 – Jan 2025, NASDAQ-100 universe only. Not representative of live or future performance. Paper benchmarks used ChatGPT-4o; live system uses Claude.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Next.js Frontend                  │
│  Dashboard · Portfolio · Scoring · Rebalance ·      │
│  Backtest · Export · Settings                       │
│  Zustand · TanStack Table · Recharts · Tailwind     │
└────────────────────┬────────────────────────────────┘
                     │ REST API
┌────────────────────▼────────────────────────────────┐
│                  FastAPI Backend                    │
│  /api/portfolio  /api/scores  /api/optimize         │
│  /api/rebalance  /api/backtest  /api/export         │
└─────┬──────────────────┬──────────────────┬─────────┘
      │                  │                  │
┌─────▼──────┐  ┌────────▼──────┐  ┌────────▼──────┐
│ PostgreSQL  │  │  Celery+Redis  │  │ External APIs  │
│  (data)     │  │  (async jobs)  │  │  Alpaca        │
│             │  │               │  │  Alpha Vantage │
│             │  │  Score Run    │  │  SEC EDGAR     │
│             │  │  Opt. Job     │  │  Claude API    │
└─────────────┘  └───────────────┘  └────────────────┘
```

---

## Scoring Formula

```
CombinedScore(i,t) = w × MLScore(i,t) + (1-w) × LLMScore(i,t)
```

**Optimal weights** locked from Table 1 (above) per strategy-frequency pair.

**ML ensembles:**
- **Fundamental:** Ridge (30%) + XGBoost (30%) + Random Forest (20%) + MLP (20%)
- **Technical:** XGBoost + LightGBM + CatBoost + LSTM (25% each)
- **Entropy:** Same ensemble as Technical; features = fuzzy entropy over 30-day rolling windows

**LLM scoring:** Claude reads full SEC 10-K/10-Q/8-K via 200K context window.

**Fallback:** If Claude API unavailable → `w=1.0` (pure ML). Warning banner shown.

---

## Project Structure

```
platform/
├── frontend/                    # Next.js 14
│   ├── app/
│   │   ├── dashboard/page.tsx   # Screen 1: KPIs + benchmark table + charts
│   │   ├── portfolio/page.tsx   # Screen 2: CSV upload + holdings table
│   │   ├── scoring/page.tsx     # Screen 3: TanStack Table + strategy tabs
│   │   ├── rebalance/page.tsx   # Screen 4: 5 sub-panels (4a–4e)
│   │   ├── backtest/page.tsx    # Screen 5: Source-fact table + empty states
│   │   ├── export/page.tsx      # Screen 6: CSV/IBKR/Schwab/PDF export
│   │   └── settings/page.tsx    # Screen 7: API keys + locked weights
│   ├── components/
│   │   ├── Sidebar.tsx
│   │   ├── NotificationContainer.tsx
│   │   └── ui/index.tsx         # KPI, ScorePill, Badge, WeightBar, etc.
│   ├── store/index.ts           # Zustand global state
│   ├── lib/api-client.ts        # Typed axios API client
│   └── types/index.ts           # Domain types + BENCHMARKS constant
│
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app + CORS + lifespan
│   │   ├── database.py          # SQLAlchemy engine + session
│   │   ├── models.py            # ORM models (all tables)
│   │   ├── schemas.py           # Pydantic v2 request/response models
│   │   ├── routers/
│   │   │   ├── portfolio.py     # Upload CSV, get portfolio, constraints
│   │   │   ├── scores.py        # Run scores, latest, history, polling
│   │   │   ├── optimize.py      # deep-rl / mvo / hrp dispatch
│   │   │   ├── rebalance.py     # Propose, approve, modify, reject, trades
│   │   │   └── backtest.py      # Locked benchmarks + export router
│   │   ├── ml/
│   │   │   ├── scoring.py       # CombinedScore formula + normalize + select_top_n
│   │   │   ├── llm_scoring.py   # Claude API with caching + fallback
│   │   │   ├── fundamental.py   # Ridge + XGB + RF + MLP ensemble
│   │   │   ├── technical.py     # XGB + LGB + CatBoost + LSTM ensemble
│   │   │   └── entropy.py       # Fuzzy entropy features + same ensemble
│   │   ├── data/
│   │   │   └── clients.py       # Alpaca + Alpha Vantage + EDGAR clients
│   │   ├── optimizer/
│   │   │   └── deep_rl.py       # PPO env + trainer + MVO + HRP fallbacks
│   │   └── workers/
│   │       └── tasks.py         # Celery: run_score_job + run_optimization_job
│   ├── migrations/
│   │   └── versions/001_initial_schema.py
│   └── requirements.txt
│
├── Dockerfile
├── docker-compose.yml
├── railway.toml                 # Phase 1–2 deployment
├── fly.toml                     # Phase 3 migration scaffold
└── .env.example
```

---

## Quick Start (Local Development)

### 1. Clone and configure

```bash
git clone <repo>
cd platform
cp .env.example .env
# Fill in ANTHROPIC_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPHA_VANTAGE_API_KEY
```

### 2. Start all services

```bash
docker compose up --build
```

Services:
- Frontend:  http://localhost:3000
- API docs:  http://localhost:8000/docs
- Flower:    http://localhost:5555 (Celery monitoring)
- Postgres:  localhost:5432
- Redis:     localhost:6379

### 3. Run Alembic migrations (first run only)

```bash
docker compose exec api alembic upgrade head
```

### 4. Frontend only (dev mode)

```bash
cd frontend
npm install
npm run dev
```

---

## Data Source Failure Isolation

| Source | If Unavailable | Impact |
|--------|---------------|--------|
| **Alpaca** | Technical + Entropy blocked | Fundamental still runs |
| **Alpha Vantage** | Fundamental blocked | Technical + Entropy still run |
| **SEC EDGAR** | No LLM filing context | All strategies fall back to w=1.0 |
| **Claude API** | No semantic scores | All strategies use w=1.0; warning banner shown; `llm_provider="none"` logged |

---

## Deep RL Optimizer

- Algorithm: PPO via `stable-baselines3`
- State: `[composite_scores, ret_1m, ret_3m, vol_21d, current_weights]` (5 × n_assets)
- Reward: `R(t) = Sharpe(t) − 0.01 × Turnover(t)`
- Training window: Rolling 24 months (no lookahead)
- Retraining: Monthly

**⚠ IMPORTANT:** Deep RL optimizer output is ALWAYS labeled separately from the paper's portfolio selection model (top-10 equal-weight from composite scores). They are distinct outputs.

---

## Railway Deployment (Phase 1–2)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

Set environment variables in Railway dashboard:
- `ANTHROPIC_API_KEY`
- `ALPACA_API_KEY` + `ALPACA_SECRET_KEY`
- `ALPHA_VANTAGE_API_KEY`
- `DATABASE_URL` (auto-injected by PostgreSQL plugin)
- `REDIS_URL` (auto-injected by Redis plugin)

For Celery worker, create a second Railway service with start command:
```
celery -A app.workers.tasks.celery_app worker --loglevel=info --concurrency=2
```

---

## Fly.io Migration (Phase 3)

See `fly.toml` for the scaffold. Key steps:
```bash
fly launch --dockerfile Dockerfile
fly postgres create --name portfolio-db
fly redis create --name portfolio-redis
fly secrets set ANTHROPIC_API_KEY=... ALPACA_API_KEY=...
fly deploy
```

---

## Citation

```bibtex
@article{cohen2025ai,
  title   = {AI-Driven Portfolio Optimization: Integrating Machine Learning and 
             Large Language Models for Enhanced Decision-Making},
  author  = {Cohen, Baruch Amrany and Aiche, Ayelet and Eichel, Roni},
  journal = {Entropy},
  volume  = {27},
  number  = {550},
  year    = {2025},
  doi     = {10.3390/e27050550}
}
```
