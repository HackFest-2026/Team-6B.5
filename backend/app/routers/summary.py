"""
backend/app/routers/summary.py
================================
Command-center summary endpoint: cohort-level metrics from the database.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..inference import get_model_version, get_threshold, is_loaded
from ..models_db import CarePlan, DemoEncounter
from ..schemas import CarePlanCounts, RiskDistribution, SummaryResponse

router = APIRouter(prefix="/api", tags=["summary"])

COHORT_NOTE = (
    "This cohort is a de-identified research demonstration derived from the UCI Diabetes "
    "130-US Hospitals dataset (1999–2008). It is NOT a live clinical database. "
    "Model outputs are for demonstration purposes only and must not be used "
    "for individual patient care decisions."
)


@router.get("/summary", response_model=SummaryResponse)
@router.get("/overview", response_model=SummaryResponse)
async def get_summary(
    risk_tier: str | None = None,
    facility: int | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Aggregate cohort-level metrics computed from the database with optional filtering."""
    threshold = get_threshold()

    # Base query for encounters
    stmt = select(DemoEncounter)
    if risk_tier:
        stmt = stmt.where(DemoEncounter.risk_tier == risk_tier)
    if facility:
        stmt = stmt.where(DemoEncounter.admission_source_id == facility)

    # Total encounters
    total_result = await db.execute(select(func.count()).select_from(stmt.subquery()))
    total = total_result.scalar_one() or 0

    # Flagged (above threshold)
    flagged_stmt = select(func.count(DemoEncounter.id)).where(
        DemoEncounter.readmission_prob >= threshold
    )
    if risk_tier:
        flagged_stmt = flagged_stmt.where(DemoEncounter.risk_tier == risk_tier)
    if facility:
        flagged_stmt = flagged_stmt.where(DemoEncounter.admission_source_id == facility)
    flagged_result = await db.execute(flagged_stmt)
    flagged = flagged_result.scalar_one() or 0

    # Risk tier counts
    tier_rows = await db.execute(
        select(DemoEncounter.risk_tier, func.count(DemoEncounter.id))
        .group_by(DemoEncounter.risk_tier)
    )
    tier_map = {row[0]: row[1] for row in tier_rows}

    # Care plan status counts
    plan_rows = await db.execute(
        select(CarePlan.status, func.count(CarePlan.id))
        .group_by(CarePlan.status)
    )
    plan_map = {row[0]: row[1] for row in plan_rows}
    # Count encounters without any care plan as "pending"
    planned_result = await db.execute(select(func.count(CarePlan.id)))
    planned_count = planned_result.scalar_one() or 0
    unplanned = max(0, total - planned_count)

    return SummaryResponse(
        total_encounters=total,
        flagged_count=flagged,
        flagged_pct=round(flagged / total * 100, 1) if total > 0 else 0.0,
        risk_distribution=RiskDistribution(
            low=tier_map.get("low", 0),
            elevated=tier_map.get("elevated", 0),
            high=tier_map.get("high", 0),
        ),
        care_plan_counts=CarePlanCounts(
            pending=plan_map.get("pending", 0) + unplanned,
            in_progress=plan_map.get("in_progress", 0),
            completed=plan_map.get("completed", 0),
        ),
        model_version=get_model_version() if is_loaded() else "not loaded",
        threshold=threshold,
        cohort_note=COHORT_NOTE,
    )
