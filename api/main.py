"""
api/main.py
===========
FastAPI backend for FIFA WC 2026 Prediction System.

Endpoints:
  GET /health
  GET /predict?home=France&away=Brazil
  GET /predict/score?home=France&away=Brazil
  GET /simulation
  GET /simulation/history
  GET /standings
  GET /standings/{group_letter}
  GET /teams
  GET /teams/h2h?team1=France&team2=Brazil
  GET /teams/{team_name}

Run locally:
    uvicorn api.main:app --reload --port 8000
"""

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import predict, simulation, standings, teams
from .core.database    import get_conn
from .core.model_loader import get_model, get_poisson

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title       = "FIFA WC 2026 Prediction API",
    description = "ML + Poisson ensemble for match prediction and tournament simulation",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ---------------------------------------------------------------------------
# Startup — pre-load models
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    print("[startup] Loading models into cache...")
    get_model()
    get_poisson()
    print("[startup] Models ready.")

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(predict.router)
app.include_router(simulation.router)
app.include_router(standings.router)
app.include_router(teams.router)

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health", tags=["health"])
def health():
    conn = get_conn()
    matches_played = conn.execute("""
        SELECT COUNT(*) FROM matches
        WHERE date >= '2026-06-11'
          AND tournament = 'FIFA World Cup'
          AND home_score IS NOT NULL
    """).fetchone()[0]

    last_sim = conn.execute("""
        SELECT run_at, n_simulations FROM simulation_results
        ORDER BY run_at DESC LIMIT 1
    """).fetchone()
    conn.close()

    return {
        "status":          "ok",
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "matches_played":  matches_played,
        "last_simulation": dict(last_sim) if last_sim else None,
    }

@app.get("/", tags=["health"])
def root():
    return {
        "message": "FIFA WC 2026 Prediction API",
        "docs":    "/docs",
        "health":  "/health",
    }