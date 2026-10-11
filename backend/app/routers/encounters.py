"""
backend/app/routers/encounters.py
==================================
Endpoints for the encounter worklist and individual encounter detail.
"""

from __future__ import annotations

import math
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..database import get_db
from ..models_db import CarePlan, DemoEncounter
from ..schemas import EncounterDetail, RiskDriver, RecommendedAction, WorklistItem, WorklistResponse

router = APIRouter(prefix="/api", tags=["encounters"])

MAX_PAGE_SIZE = 100


@router.get("/worklist", response_model=WorklistResponse)
@router.get("/encounters", response_model=WorklistResponse)
async def get_worklist(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=MAX_PAGE_SIZE),
    tier: Optional[str] = Query(None, pattern="^(low|elevated|high)$"),
    status: Optional[str] = Query(None, pattern="^(pending|in_progress|completed|unplanned)$"),
    search: Optional[str] = Query(None, max_length=20),
    sort_by: Optional[str] = Query("readmission_prob", pattern="^(readmission_prob|time_in_hospital|num_medications|number_diagnoses|encounter_id)$"),
    sort_dir: Optional[str] = Query("desc", pattern="^(asc|desc)$"),
    db: AsyncSession = Depends(get_db),
):
    """Paginated, filtered worklist with DB-side pagination and configurable sorting."""
    # Build base query
    stmt = (
        select(DemoEncounter)
        .outerjoin(CarePlan, CarePlan.encounter_id == DemoEncounter.encounter_id)
        .options(selectinload(DemoEncounter.care_plan))
    )

    # Filters
    if tier:
        stmt = stmt.where(DemoEncounter.risk_tier == tier)
    if search:
        try:
            enc_id = int(search)
            stmt = stmt.where(DemoEncounter.encounter_id == enc_id)
        except ValueError:
            stmt = stmt.where(DemoEncounter.encounter_id == -1)  # no match
    if status == "unplanned":
        stmt = stmt.where(CarePlan.id.is_(None))
    elif status in ("pending", "in_progress", "completed"):
        stmt = stmt.where(CarePlan.status == status)

    # Count total
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_result = await db.execute(count_stmt)
    total = total_result.scalar_one()

    # Dynamic sorting
    sort_col = getattr(DemoEncounter, sort_by or "readmission_prob", DemoEncounter.readmission_prob)
    if sort_dir == "asc":
        stmt = stmt.order_by(sort_col.asc().nulls_last())
    else:
        stmt = stmt.order_by(sort_col.desc().nulls_last())

    # Paginate
    offset = (page - 1) * page_size
    stmt = stmt.offset(offset).limit(page_size)
    result = await db.execute(stmt)
    encounters = result.scalars().all()

    items = [
        WorklistItem(
            encounter_id=enc.encounter_id,
            age=enc.age,
            gender=enc.gender,
            race=enc.race,
            time_in_hospital=enc.time_in_hospital,
            num_medications=enc.num_medications,
            readmission_prob=float(enc.readmission_prob or 0),
            risk_tier=enc.risk_tier or "low",
            care_plan_status=enc.care_plan.status if enc.care_plan else None,
            model_version=enc.model_version,
        )
        for enc in encounters
    ]

    return WorklistResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=max(1, math.ceil(total / page_size)),
    )


@router.get("/encounters/{enc_id}", response_model=EncounterDetail)
async def get_encounter(enc_id: int, db: AsyncSession = Depends(get_db)):
    """Full encounter detail with real-time explanation."""
    from ..inference import score, explain, is_loaded

    result = await db.execute(
        select(DemoEncounter).where(DemoEncounter.encounter_id == enc_id)
    )
    enc = result.scalar_one_or_none()
    if enc is None:
        raise HTTPException(status_code=404, detail=f"Encounter {enc_id} not found")

    # Build encounter dict for explanation
    enc_dict = {
        "age": enc.age,
        "gender": enc.gender,
        "race": enc.race,
        "time_in_hospital": enc.time_in_hospital,
        "num_lab_procedures": enc.num_lab_procedures,
        "num_procedures": enc.num_procedures,
        "num_medications": enc.num_medications,
        "number_outpatient": enc.number_outpatient,
        "number_emergency": enc.number_emergency,
        "number_inpatient": enc.number_inpatient,
        "number_diagnoses": enc.number_diagnoses,
        "diag_1": enc.diag_1,
        "diag_2": enc.diag_2,
        "diag_3": enc.diag_3,
        "max_glu_serum": enc.max_glu_serum,
        "A1Cresult": enc.a1c_result,
        "change": enc.change,
        "diabetesMed": enc.diabetes_med,
        "admission_type_id": enc.admission_type_id,
        "discharge_disposition_id": enc.discharge_disposition_id,
        "admission_source_id": enc.admission_source_id,
    }

    # Data quality flags
    quality_flags = []
    for field, val in enc_dict.items():
        if val in (None, "?", "Unknown", "None") and field in (
            "diag_1", "diag_2", "diag_3", "max_glu_serum", "A1Cresult"
        ):
            quality_flags.append(f"{field}: missing/unknown value")

    # Use pre-computed prob or live inference
    prob = float(enc.readmission_prob or 0)
    tier = enc.risk_tier or "low"
    explanation_result = {"drivers": [], "method": "pre_computed", "recommended_actions": []}

    if is_loaded():
        try:
            explanation_result = explain(enc_dict)
            prob = explanation_result.get("prob", prob)
            tier = explanation_result.get("tier", tier)
        except Exception as e:
            print(f"[encounters] Explanation failed for {enc_id}: {e}")

    from src.explainability import build_recommended_actions
    if not explanation_result.get("recommended_actions"):
        explanation_result["recommended_actions"] = build_recommended_actions(tier, [], enc_dict)

    drivers = [
        RiskDriver(
            feature=d.get("feature", ""),
            label=d.get("label", d.get("feature", "")),
            direction=d.get("direction"),
            shap_value=d.get("shap_value"),
            importance=d.get("importance"),
            modifiable=d.get("modifiable", False),
            association_note=d.get("association_note", "association, not causation"),
        )
        for d in explanation_result.get("drivers", [])
    ]

    actions = [
        RecommendedAction(
            key=a["key"],
            label=a["label"],
            rationale=a["rationale"],
            priority=a["priority"],
        )
        for a in explanation_result.get("recommended_actions", [])
    ]

    return EncounterDetail(
        encounter_id=enc.encounter_id,
        age=enc.age,
        gender=enc.gender,
        race=enc.race,
        admission_type_id=enc.admission_type_id,
        discharge_disposition_id=enc.discharge_disposition_id,
        time_in_hospital=enc.time_in_hospital,
        num_lab_procedures=enc.num_lab_procedures,
        num_medications=enc.num_medications,
        number_diagnoses=enc.number_diagnoses,
        number_inpatient=enc.number_inpatient or 0,
        number_emergency=enc.number_emergency or 0,
        number_outpatient=enc.number_outpatient or 0,
        diag_1=enc.diag_1,
        diag_2=enc.diag_2,
        diag_3=enc.diag_3,
        max_glu_serum=enc.max_glu_serum or "None",
        a1c_result=enc.a1c_result or "None",
        readmission_prob=prob,
        risk_tier=tier,
        model_version=enc.model_version,
        drivers=drivers,
        explanation_method=explanation_result.get("method", "pre_computed"),
        recommended_actions=actions,
        data_quality_flags=quality_flags,
    )
