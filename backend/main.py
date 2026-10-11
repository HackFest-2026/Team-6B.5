"""
backend/main.py
===============
FastAPI application entry point.

Start with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Architecture:
    React/Vite frontend → FastAPI REST API → PostgreSQL/Supabase
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import sys
from pathlib import Path

_backend_dir = Path(__file__).resolve().parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

try:
    from backend.app.database import init_db, engine
    from backend.app import inference
    from backend.app.routers import care_plans, encounters, governance, predict, summary, datasets
except ModuleNotFoundError:
    from app.database import init_db, engine
    from app import inference
    from app.routers import care_plans, encounters, governance, predict, summary, datasets


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model artifacts and initialise DB tables."""
    # Load ML artifacts (once per process)
    model_loaded = inference.load_artifacts()
    if not model_loaded:
        print("[main] WARNING: Model artifacts not found. /api/predict will return 503.")

    # Create tables if not present (SQLite fallback / first-run)
    try:
        await init_db()
        print("[main] Database tables ready.")
    except Exception as e:
        print(f"[main] DB init warning: {e}")

    yield

    # Shutdown
    print("[main] Shutting down.")


app = FastAPI(
    title="ClinicalAI: Hospital Readmission Risk",
    description=(
        "Research demonstration for 30-day readmission risk prediction in diabetic inpatients. "
        "NOT a validated clinical product. For hackathon demonstration purposes only."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
_allowed_origins = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(summary.router)
app.include_router(encounters.router)
app.include_router(care_plans.router)
app.include_router(predict.router)
app.include_router(governance.router)
app.include_router(datasets.router)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/api/health", tags=["health"])
async def health():
    """Readiness probe for the API and model."""
    db_ok = False
    try:
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    model_ok = inference.is_loaded()
    status = "ok" if (model_ok and db_ok) else "degraded"

    return JSONResponse(
        status_code=200 if status == "ok" else 503,
        content={
            "status": status,
            "model_loaded": model_ok,
            "db_connected": db_ok,
            "model_version": inference.get_model_version() if model_ok else None,
        },
    )


@app.get("/api/predictions/history", tags=["predict"])
async def prediction_history():
    """Stub for prediction run history (not yet paginated)."""
    return {"history": [], "message": "Prediction history available after DB seeding."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
