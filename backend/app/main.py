"""
AI Portfolio Decision-Support Platform — FastAPI Backend
Cohen, Aiche & Eichel (2025), Entropy 27, 550
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.database import engine, Base
from app.routers import portfolio, scores, optimize, rebalance, backtest, dashboard
from app.routers.backtest import export_router
from app.routers.discovery import router as discovery_router
from app.routers.search import router as search_router
from app.routers.report import router as report_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables verified.")
    yield


app = FastAPI(
    title="AI Portfolio Decision-Support API",
    description="Based on Cohen, Aiche & Eichel (2025), Entropy 27, 550",
    version="1.0.0",
    lifespan=lifespan,
)

import os

# CORS: allow the deployed frontend (and local dev) to call this API from the
# browser. The production frontend is https://optimizer-tau.vercel.app; Vercel
# preview deploys get https://<branch-hash>.vercel.app, covered by the regex.
# Extra origins can be added via CORS_ALLOW_ORIGINS (comma-separated) without a
# code change.
_default_origins = [
    "https://optimizer-tau.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
_env_origins = [
    o.strip()
    for o in os.environ.get("CORS_ALLOW_ORIGINS", "").split(",")
    if o.strip()
]
_allow_origins = list(dict.fromkeys(_default_origins + _env_origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(req: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


app.include_router(portfolio.router,  prefix="/api/portfolio",  tags=["portfolio"])
app.include_router(scores.router,     prefix="/api/scores",     tags=["scores"])
app.include_router(optimize.router,   prefix="/api/optimize",   tags=["optimize"])
app.include_router(rebalance.router,  prefix="/api/rebalance",  tags=["rebalance"])
app.include_router(backtest.router,   prefix="/api/backtest",   tags=["backtest"])
app.include_router(dashboard.router,  prefix="/api/dashboard",  tags=["dashboard"])
app.include_router(export_router,     prefix="/api/export",     tags=["export"])
app.include_router(discovery_router)
app.include_router(search_router)
app.include_router(report_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
