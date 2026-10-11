"""
backend/app/models_db.py
========================
SQLAlchemy ORM models mapped to the PostgreSQL schema.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def _uuid():
    return str(uuid.uuid4())


class DemoEncounter(Base):
    __tablename__ = "demo_encounters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    encounter_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    age: Mapped[str] = mapped_column(String(20), default="[60-70)")
    gender: Mapped[str] = mapped_column(String(20), default="Unknown")
    race: Mapped[str] = mapped_column(String(50), default="Unknown")
    admission_type_id: Mapped[Optional[int]] = mapped_column(Integer)
    discharge_disposition_id: Mapped[Optional[int]] = mapped_column(Integer)
    admission_source_id: Mapped[Optional[int]] = mapped_column(Integer)
    time_in_hospital: Mapped[Optional[int]] = mapped_column(Integer)
    num_lab_procedures: Mapped[Optional[int]] = mapped_column(Integer)
    num_procedures: Mapped[Optional[int]] = mapped_column(Integer)
    num_medications: Mapped[Optional[int]] = mapped_column(Integer)
    number_outpatient: Mapped[int] = mapped_column(Integer, default=0)
    number_emergency: Mapped[int] = mapped_column(Integer, default=0)
    number_inpatient: Mapped[int] = mapped_column(Integer, default=0)
    number_diagnoses: Mapped[Optional[int]] = mapped_column(Integer)
    diag_1: Mapped[Optional[str]] = mapped_column(String(20))
    diag_2: Mapped[Optional[str]] = mapped_column(String(20))
    diag_3: Mapped[Optional[str]] = mapped_column(String(20))
    max_glu_serum: Mapped[str] = mapped_column(String(10), default="None")
    a1c_result: Mapped[str] = mapped_column(String(10), default="None")
    change: Mapped[str] = mapped_column(String(5), default="No")
    diabetes_med: Mapped[str] = mapped_column(String(5), default="Yes")
    readmission_prob: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    risk_tier: Mapped[Optional[str]] = mapped_column(String(10))
    model_version: Mapped[str] = mapped_column(String(20), default="1.0.0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    care_plan: Mapped[Optional["CarePlan"]] = relationship(
        "CarePlan", back_populates="encounter", uselist=False
    )


class CarePlan(Base):
    __tablename__ = "care_plans"
    __table_args__ = (UniqueConstraint("encounter_id", name="uq_care_plan_per_encounter"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    encounter_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("demo_encounters.encounter_id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(20),
        CheckConstraint("status IN ('pending', 'in_progress', 'completed')"),
        default="pending",
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    acting_user: Mapped[Optional[str]] = mapped_column(String(100))

    encounter: Mapped["DemoEncounter"] = relationship("DemoEncounter", back_populates="care_plan")
    actions: Mapped[list["CarePlanAction"]] = relationship(
        "CarePlanAction", back_populates="care_plan", cascade="all, delete-orphan"
    )


class CarePlanAction(Base):
    __tablename__ = "care_plan_actions"
    __table_args__ = (UniqueConstraint("care_plan_id", "action_key", name="uq_action_per_plan"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    care_plan_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("care_plans.id", ondelete="CASCADE"), nullable=False
    )
    action_key: Mapped[str] = mapped_column(String(50), nullable=False)
    label: Mapped[str] = mapped_column(Text, nullable=False)
    completed: Mapped[bool] = mapped_column(Boolean, default=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    assigned_to: Mapped[Optional[str]] = mapped_column(String(100), default="Care Coordinator")
    due_date: Mapped[Optional[str]] = mapped_column(String(30))
    notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    care_plan: Mapped["CarePlan"] = relationship("CarePlan", back_populates="actions")


class PredictionRun(Base):
    __tablename__ = "prediction_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_type: Mapped[str] = mapped_column(String(20), default="scenario")
    encounter_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("demo_encounters.encounter_id")
    )
    model_version: Mapped[str] = mapped_column(String(20), default="1.0.0")
    input_hash: Mapped[Optional[str]] = mapped_column(String(64))
    probability: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    tier: Mapped[str] = mapped_column(String(10), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_type: Mapped[Optional[str]] = mapped_column(String(50))
    entity_id: Mapped[Optional[str]] = mapped_column(String(100))
    acting_user: Mapped[Optional[str]] = mapped_column(String(100), default="System")
    details: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DatasetImport(Base):
    __tablename__ = "dataset_imports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, default=0)
    valid_count: Mapped[int] = mapped_column(Integer, default=0)
    invalid_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="validated")
    summary_report: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
