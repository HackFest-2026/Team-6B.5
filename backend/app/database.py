"""
backend/app/database.py
=======================
SQLAlchemy 2.x async engine and session factory.
DATABASE_URL is read from environment only; never from Vite or client code.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

# Support both postgres:// and postgresql+asyncpg:// URL formats
_raw_url = os.environ.get("DATABASE_URL", "")

if _raw_url.startswith("postgres://"):
    _raw_url = _raw_url.replace("postgres://", "postgresql+asyncpg://", 1)
elif _raw_url.startswith("postgresql://"):
    _raw_url = _raw_url.replace("postgresql://", "postgresql+asyncpg://", 1)

# SQLite fallback for running without a real DB
if not _raw_url:
    import tempfile, pathlib
    _db_file = pathlib.Path(__file__).resolve().parent.parent.parent / "data" / "demo.db"
    _db_file.parent.mkdir(parents=True, exist_ok=True)
    _raw_url = f"sqlite+aiosqlite:///{_db_file}"
    print(f"[db] No DATABASE_URL — using SQLite fallback: {_db_file}")

engine = create_async_engine(
    _raw_url,
    echo=False,
    pool_pre_ping=True,
    # SQLite doesn't support pool_size
    **({} if "sqlite" in _raw_url else {"pool_size": 5, "max_overflow": 10}),
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    """Create tables if not present and ensure required columns exist."""
    async with engine.begin() as conn:
        # Import models so Base has all table metadata
        from . import models_db
        await conn.run_sync(Base.metadata.create_all)

        # For SQLite, ensure newly added columns exist if tables pre-dated them
        if "sqlite" in _raw_url:
            from sqlalchemy import text
            try:
                # check audit_events details column
                res = await conn.execute(text("PRAGMA table_info(audit_events)"))
                cols = [row[1] for row in res.fetchall()]
                if "details" not in cols:
                    await conn.execute(text("ALTER TABLE audit_events ADD COLUMN details TEXT"))

                # check care_plan_actions columns
                res_cp = await conn.execute(text("PRAGMA table_info(care_plan_actions)"))
                cp_cols = [row[1] for row in res_cp.fetchall()]
                if "assigned_to" not in cp_cols:
                    await conn.execute(text("ALTER TABLE care_plan_actions ADD COLUMN assigned_to VARCHAR(100) DEFAULT 'Care Manager'"))
                if "due_date" not in cp_cols:
                    await conn.execute(text("ALTER TABLE care_plan_actions ADD COLUMN due_date TIMESTAMP"))
                if "notes" not in cp_cols:
                    await conn.execute(text("ALTER TABLE care_plan_actions ADD COLUMN notes TEXT"))
            except Exception as mig_err:
                print(f"[db] SQLite auto-migration note: {mig_err}")

