"""
backend/app/routers/predict.py
================================
Prediction / scenario simulator endpoint.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..inference import explain, hash_input, is_loaded, get_model_version, score
from ..models_db import DemoEncounter, PredictionRun
from ..schemas import PredictResponse, RecommendedAction, RiskDriver, ScenarioInput

router = APIRouter(prefix="/api", tags=["predict"])


def _enc_to_dict(enc: DemoEncounter) -> dict:
    return {
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


@router.post("/predict", response_model=PredictResponse)
@router.post("/predictions/score", response_model=PredictResponse)
async def predict(body: ScenarioInput, db: AsyncSession = Depends(get_db)):
    """Score a scenario. Optionally load baseline from a saved encounter."""
    if not is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not available. Ensure training has been run."
        )

    # Load baseline encounter if enc_id given
    baseline_enc: Optional[DemoEncounter] = None
    baseline_prob: Optional[float] = None
    baseline_tier: Optional[str] = None

    if body.encounter_id is not None:
        result = await db.execute(
            select(DemoEncounter).where(DemoEncounter.encounter_id == body.encounter_id)
        )
        baseline_enc = result.scalar_one_or_none()
        if baseline_enc is None:
            raise HTTPException(status_code=404, detail=f"Encounter {body.encounter_id} not found")

        baseline_dict = _enc_to_dict(baseline_enc)
        try:
            baseline_prob, baseline_tier = score(baseline_dict)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Baseline scoring failed: {e}")

    # Build scenario dict
    scenario_dict = {}
    if baseline_enc:
        scenario_dict = _enc_to_dict(baseline_enc)

    # Apply scenario overrides (only non-None fields)
    override_fields = body.model_dump(exclude_none=True, exclude={"encounter_id"})
    for field, val in override_fields.items():
        # Map API field names to internal names
        mapped = {"a1c_result": "A1Cresult", "diabetes_med": "diabetesMed"}.get(field, field)
        scenario_dict[mapped] = val

    try:
        result_explain = explain(scenario_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)[:200]}")

    scenario_prob = result_explain["prob"]
    tier = result_explain["tier"]
    absolute_change = (
        round(scenario_prob - baseline_prob, 4) if baseline_prob is not None else None
    )

    # Persist prediction run (non-fatal)
    try:
        run = PredictionRun(
            run_type="scenario" if body.encounter_id is not None else "baseline",
            encounter_id=body.encounter_id,
            model_version=get_model_version(),
            input_hash=hash_input(scenario_dict),
            probability=scenario_prob,
            tier=tier,
        )
        db.add(run)
        await db.commit()
    except Exception:
        pass  # Don't let audit logging break the scoring response

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
        for d in result_explain.get("drivers", [])
    ]

    actions = [
        RecommendedAction(
            key=a["key"],
            label=a["label"],
            rationale=a["rationale"],
            priority=a["priority"],
        )
        for a in result_explain.get("recommended_actions", [])
    ]

    return PredictResponse(
        encounter_id=body.encounter_id,
        baseline_prob=round(baseline_prob, 4) if baseline_prob is not None else None,
        scenario_prob=round(scenario_prob, 4),
        absolute_change=round(absolute_change, 4) if absolute_change is not None else None,
        tier=tier,
        baseline_tier=baseline_tier,
        model_version=get_model_version(),
        drivers=drivers,
        recommended_actions=actions,
    )
