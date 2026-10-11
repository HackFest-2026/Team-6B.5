"""
backend/app/schemas.py
======================
Pydantic v2 request/response schemas for all API endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ── Health ─────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"] = "ok"
    model_loaded: bool
    db_connected: bool
    model_version: Optional[str] = None
    message: Optional[str] = None


# ── Summary / Command Center ───────────────────────────────────────────────────

class RiskDistribution(BaseModel):
    low: int
    elevated: int
    high: int


class CarePlanCounts(BaseModel):
    pending: int
    in_progress: int
    completed: int


class SummaryResponse(BaseModel):
    total_encounters: int
    flagged_count: int
    flagged_pct: float
    risk_distribution: RiskDistribution
    care_plan_counts: CarePlanCounts
    model_version: str
    threshold: float
    cohort_note: str


# ── Worklist ───────────────────────────────────────────────────────────────────

class WorklistItem(BaseModel):
    encounter_id: int
    age: str
    gender: str
    race: str
    time_in_hospital: Optional[int]
    num_medications: Optional[int]
    readmission_prob: float
    risk_tier: str
    care_plan_status: Optional[str]
    model_version: str


class WorklistResponse(BaseModel):
    items: List[WorklistItem]
    total: int
    page: int
    page_size: int
    total_pages: int


# ── Encounter Detail ───────────────────────────────────────────────────────────

class RiskDriver(BaseModel):
    feature: str
    label: str
    direction: Optional[str] = None
    shap_value: Optional[float] = None
    importance: Optional[float] = None
    modifiable: bool
    association_note: str = "association, not causation"


class RecommendedAction(BaseModel):
    key: str
    label: str
    rationale: str
    priority: str


class EncounterDetail(BaseModel):
    encounter_id: int
    age: str
    gender: str
    race: str
    admission_type_id: Optional[int]
    discharge_disposition_id: Optional[int]
    time_in_hospital: Optional[int]
    num_lab_procedures: Optional[int]
    num_medications: Optional[int]
    number_diagnoses: Optional[int]
    number_inpatient: int
    number_emergency: int
    number_outpatient: int
    diag_1: Optional[str]
    diag_2: Optional[str]
    diag_3: Optional[str]
    max_glu_serum: str
    a1c_result: str
    # Risk output
    readmission_prob: float
    risk_tier: str
    model_version: str
    # Explanation
    drivers: List[RiskDriver]
    explanation_method: str
    recommended_actions: List[RecommendedAction]
    data_quality_flags: List[str]
    research_demo_notice: str = (
        "This is a historical, de-identified research demonstration. "
        "Model output is NOT a validated clinical tool and must not be used for individual care decisions."
    )


# ── Predict ────────────────────────────────────────────────────────────────────

class ScenarioInput(BaseModel):
    encounter_id: Optional[int] = None
    # Scenario overrides — all optional; each has validated bounds
    age: Optional[str] = Field(None, description="Age band e.g. '[60-70)'")
    gender: Optional[str] = Field(None, pattern="^(Male|Female|Unknown)$")
    race: Optional[str] = None
    time_in_hospital: Optional[int] = Field(None, ge=1, le=30)
    num_lab_procedures: Optional[int] = Field(None, ge=1, le=132)
    num_procedures: Optional[int] = Field(None, ge=0, le=6)
    num_medications: Optional[int] = Field(None, ge=1, le=81)
    number_outpatient: Optional[int] = Field(None, ge=0, le=42)
    number_emergency: Optional[int] = Field(None, ge=0, le=76)
    number_inpatient: Optional[int] = Field(None, ge=0, le=21)
    number_diagnoses: Optional[int] = Field(None, ge=1, le=16)
    diag_1: Optional[str] = None
    diag_2: Optional[str] = None
    diag_3: Optional[str] = None
    max_glu_serum: Optional[str] = Field(None, pattern="^(None|>200|>300|Norm)$")
    a1c_result: Optional[str] = Field(None, pattern="^(None|>7|>8|Norm)$")
    change: Optional[str] = Field(None, pattern="^(No|Ch)$")
    diabetes_med: Optional[str] = Field(None, pattern="^(Yes|No)$")
    admission_type_id: Optional[int] = Field(None, ge=1, le=8)
    discharge_disposition_id: Optional[int] = Field(None, ge=1, le=30)
    admission_source_id: Optional[int] = Field(None, ge=1, le=26)
    insulin: Optional[str] = Field(None, pattern="^(No|Steady|Up|Down)$")
    metformin: Optional[str] = Field(None, pattern="^(No|Steady|Up|Down)$")


class PredictResponse(BaseModel):
    encounter_id: Optional[int]
    baseline_prob: Optional[float]
    scenario_prob: float
    absolute_change: Optional[float]
    tier: str
    baseline_tier: Optional[str]
    model_version: str
    drivers: List[RiskDriver]
    recommended_actions: List[RecommendedAction]
    research_demo_notice: str = (
        "Research demonstration only. Not for clinical use."
    )


# ── Care Plans ────────────────────────────────────────────────────────────────

class CarePlanActionOut(BaseModel):
    id: str
    action_key: str
    label: str
    completed: bool
    completed_at: Optional[datetime] = None
    assigned_to: Optional[str] = "Care Coordinator"
    due_date: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class CarePlanOut(BaseModel):
    id: str
    encounter_id: int
    status: str
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    acting_user: Optional[str] = None
    actions: List[CarePlanActionOut]

    model_config = {"from_attributes": True}


class TaskConfigItem(BaseModel):
    action_key: str
    label: Optional[str] = None
    assigned_to: Optional[str] = "Care Coordinator"
    due_date: Optional[str] = None
    notes: Optional[str] = None


class CarePlanUpsert(BaseModel):
    encounter_id: Optional[int] = None
    status: Literal["pending", "in_progress", "completed"] = "pending"
    acting_user: Optional[str] = Field(None, max_length=100)
    actions: Optional[List[str]] = Field(
        None, description="List of action_key strings to initialise"
    )
    task_configs: Optional[List[TaskConfigItem]] = None


class ActionUpdate(BaseModel):
    completed: Optional[bool] = None
    assigned_to: Optional[str] = None
    due_date: Optional[str] = None
    notes: Optional[str] = None
    status: Optional[str] = None


# ── Governance ────────────────────────────────────────────────────────────────

class GovernanceResponse(BaseModel):
    model_comparison: Dict[str, Any]
    fairness: Dict[str, Any]
    mitigation_comparison: Dict[str, Any]
    feature_importances: List[Dict[str, Any]]
    data_quality: Dict[str, Any]
    metric_rationale: str


# ── Data Quality & Validation ──────────────────────────────────────────────────

class DataQualityResponse(BaseModel):
    total_raw: int
    after_filters: int
    train_size: int
    test_size: int
    positive_rate_train: float
    positive_rate_test: float
    after_smote_train_size: int
    missing_value_strategy: str
    outlier_strategy: str
    split_strategy: str
    imbalance_strategy: str


class DatasetValidationResult(BaseModel):
    filename: str
    total_rows: int
    valid_rows: int
    invalid_rows: int
    columns_detected: List[str]
    missing_required_columns: List[str]
    preview_rows: List[Dict[str, Any]]
    validation_report: Dict[str, Any]
    is_valid: bool
    issues: List[str]


class DatasetImportConfirm(BaseModel):
    filename: str
    records: List[Dict[str, Any]]
    acting_user: Optional[str] = "Clinical Data Specialist"


class DatasetImportResponse(BaseModel):
    message: str
    imported_count: int
    status: str
    import_id: str


# ── Provenance & Audit ────────────────────────────────────────────────────────

class AuditEventOut(BaseModel):
    id: int
    event_type: str
    entity_type: Optional[str]
    entity_id: Optional[str]
    acting_user: Optional[str]
    details: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class ModelCardResponse(BaseModel):
    model_name: str
    model_version: str
    model_type: str
    intended_use: str
    target_population: str
    primary_metric: str
    threshold: float
    training_cohort_description: str
    evaluation_split_design: str
    known_limitations: List[str]
    ethical_and_safety_considerations: List[str]
    input_features: List[str]
    governance_owner: str

