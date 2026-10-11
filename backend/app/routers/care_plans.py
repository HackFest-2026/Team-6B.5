"""
backend/app/routers/care_plans.py
===================================
Care plan CRUD endpoints — GET, PUT (upsert), and PATCH action update.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..database import get_db
from ..models_db import CarePlan, CarePlanAction, DemoEncounter
from ..schemas import ActionUpdate, CarePlanOut, CarePlanUpsert

router = APIRouter(prefix="/api/care-plans", tags=["care-plans"])

ALLOWED_ACTION_KEYS = {
    "medication_reconciliation",
    "followup_appointment",
    "diabetes_education",
    "telehealth_followup",
    "care_transitions_referral",
    "glycemic_review",
    "pharmacy_consult",
    "home_health",
}

ACTION_LABELS = {
    "medication_reconciliation": "Medication Reconciliation",
    "followup_appointment": "Schedule Follow-up Appointment",
    "diabetes_education": "Diabetes Self-Management Education",
    "telehealth_followup": "Telehealth Follow-up (48–72h)",
    "care_transitions_referral": "Care Transitions Referral",
    "glycemic_review": "Glycaemic Management Review",
    "pharmacy_consult": "Pharmacy Consultation",
    "home_health": "Home Health Coordination",
}


async def _get_encounter_or_404(enc_id: int, db: AsyncSession) -> DemoEncounter:
    result = await db.execute(
        select(DemoEncounter).where(DemoEncounter.encounter_id == enc_id)
    )
    enc = result.scalar_one_or_none()
    if enc is None:
        raise HTTPException(status_code=404, detail=f"Encounter {enc_id} not found")
    return enc


async def _load_plan(enc_id: int, db: AsyncSession) -> Optional[CarePlan]:
    result = await db.execute(
        select(CarePlan)
        .where(CarePlan.encounter_id == enc_id)
        .options(selectinload(CarePlan.actions))
    )
    return result.scalar_one_or_none()


@router.get("", response_model=list[CarePlanOut])
async def list_care_plans(
    status: Optional[str] = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """List all care plans with optional status filtering (for care plan tracker view)."""
    stmt = select(CarePlan).options(selectinload(CarePlan.actions)).order_by(CarePlan.updated_at.desc())
    if status:
        stmt = stmt.where(CarePlan.status == status)
    stmt = stmt.limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/{enc_id}", response_model=CarePlanOut)
async def get_care_plan(enc_id: int, db: AsyncSession = Depends(get_db)):
    """Get the care plan for an encounter, or 404 if none exists yet."""
    await _get_encounter_or_404(enc_id, db)
    plan = await _load_plan(enc_id, db)
    if plan is None:
        raise HTTPException(status_code=404, detail="No care plan for this encounter")
    return plan


@router.post("", response_model=CarePlanOut)
async def create_care_plan(
    body: CarePlanUpsert,
    db: AsyncSession = Depends(get_db),
):
    """Create or update a care plan via POST /api/care-plans."""
    if body.encounter_id is None:
        raise HTTPException(status_code=422, detail="encounter_id is required")
    return await upsert_care_plan(body.encounter_id, body, db)


@router.put("/{enc_id}", response_model=CarePlanOut)
async def upsert_care_plan(
    enc_id: int,
    body: CarePlanUpsert,
    db: AsyncSession = Depends(get_db),
):
    """Create or update the care plan for an encounter."""
    await _get_encounter_or_404(enc_id, db)
    plan = await _load_plan(enc_id, db)

    if plan is None:
        plan = CarePlan(
            encounter_id=enc_id,
            status=body.status,
            acting_user=body.acting_user or "Care Coordinator",
        )
        db.add(plan)
        await db.flush()

    else:
        plan.status = body.status
        if body.acting_user:
            plan.acting_user = body.acting_user
        if body.status == "completed" and plan.completed_at is None:
            plan.completed_at = datetime.now(timezone.utc)

    # Initialise actions if provided as strings
    if body.actions:
        existing_keys = {a.action_key for a in plan.actions}
        for key in body.actions:
            if key not in existing_keys:
                action = CarePlanAction(
                    care_plan_id=plan.id,
                    action_key=key,
                    label=ACTION_LABELS.get(key, key.replace("_", " ").title()),
                    assigned_to=body.acting_user or "Care Coordinator",
                )
                db.add(action)

    # Initialise structured task configs if provided
    if body.task_configs:
        existing_keys = {a.action_key for a in plan.actions}
        for cfg in body.task_configs:
            if cfg.action_key not in existing_keys:
                action = CarePlanAction(
                    care_plan_id=plan.id,
                    action_key=cfg.action_key,
                    label=cfg.label or ACTION_LABELS.get(cfg.action_key, cfg.action_key.replace("_", " ").title()),
                    assigned_to=cfg.assigned_to or body.acting_user or "Care Coordinator",
                    due_date=cfg.due_date,
                    notes=cfg.notes,
                )
                db.add(action)

    # Audit event
    try:
        from ..models_db import AuditEvent
        import json
        audit = AuditEvent(
            event_type="care_plan_upsert",
            entity_type="care_plan",
            entity_id=str(plan.id),
            acting_user=body.acting_user or "Care Coordinator",
            details=json.dumps({"encounter_id": enc_id, "status": plan.status}),
        )
        db.add(audit)
    except Exception:
        pass

    await db.commit()
    await db.refresh(plan)
    plan = await _load_plan(enc_id, db)
    return plan


@router.patch("/{enc_id}/actions/{action_id}", response_model=CarePlanOut)
@router.patch("/{enc_id}/tasks/{action_id}", response_model=CarePlanOut)
async def update_action(
    enc_id: int,
    action_id: str,
    body: ActionUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Mark an action as completed/incomplete, reassign, or add notes."""
    await _get_encounter_or_404(enc_id, db)
    plan = await _load_plan(enc_id, db)
    if plan is None:
        raise HTTPException(status_code=404, detail="No care plan for this encounter")

    action = next((a for a in plan.actions if a.id == action_id or a.action_key == action_id), None)
    if action is None:
        raise HTTPException(status_code=404, detail=f"Action/Task {action_id} not found")

    if body.completed is not None:
        action.completed = body.completed
        if body.completed and action.completed_at is None:
            action.completed_at = datetime.now(timezone.utc)
        elif not body.completed:
            action.completed_at = None

    if body.assigned_to is not None:
        action.assigned_to = body.assigned_to
    if body.due_date is not None:
        action.due_date = body.due_date
    if body.notes is not None:
        action.notes = body.notes

    # Auto-update plan status based on actions
    all_actions = plan.actions
    if all_actions:
        n_done = sum(1 for a in all_actions if a.completed)
        if n_done == 0:
            plan.status = "pending"
        elif n_done == len(all_actions):
            plan.status = "completed"
            if plan.completed_at is None:
                plan.completed_at = datetime.now(timezone.utc)
        else:
            plan.status = "in_progress"

    await db.commit()
    plan = await _load_plan(enc_id, db)
    return plan
