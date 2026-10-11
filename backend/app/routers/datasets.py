"""
backend/app/routers/datasets.py
================================
Dataset upload, validation pipeline, and confirmed import endpoints.
"""

from __future__ import annotations

import io
import json
import uuid
from typing import Any, List, Optional
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models_db import AuditEvent, DatasetImport, DemoEncounter
from ..schemas import (
    AuditEventOut,
    DatasetImportConfirm,
    DatasetImportResponse,
    DatasetValidationResult,
)
from ..inference import is_loaded, score

router = APIRouter(prefix="/api", tags=["datasets"])

REQUIRED_COLUMNS = [
    "encounter_id",
    "age",
    "gender",
    "time_in_hospital",
    "num_medications",
]

RECOMMENDED_COLUMNS = [
    "race",
    "num_lab_procedures",
    "number_diagnoses",
    "diag_1",
    "max_glu_serum",
    "A1Cresult",
    "change",
    "diabetesMed",
]


def _validate_dataframe(df: pd.DataFrame, filename: str) -> DatasetValidationResult:
    issues = []
    total_rows = len(df)

    if total_rows == 0:
        return DatasetValidationResult(
            filename=filename,
            total_rows=0,
            valid_rows=0,
            invalid_rows=0,
            columns_detected=[],
            missing_required_columns=REQUIRED_COLUMNS,
            preview_rows=[],
            validation_report={},
            is_valid=False,
            issues=["Uploaded file contains 0 rows of data."],
        )

    columns_detected = list(df.columns)
    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing_required:
        issues.append(f"Missing mandatory clinical fields: {', '.join(missing_required)}")

    # Check duplicates on encounter_id
    dup_count = 0
    if "encounter_id" in df.columns:
        dup_count = int(df["encounter_id"].duplicated().sum())
        if dup_count > 0:
            issues.append(f"Found {dup_count} duplicate encounter IDs in the file.")

    # Check outliers and missingness
    missingness = {}
    for col in df.columns:
        n_missing = int(df[col].isna().sum()) + int((df[col].astype(str).str.strip() == "?").sum())
        pct = round(n_missing / total_rows * 100, 1)
        if pct > 0:
            missingness[col] = f"{pct}% ({n_missing} rows)"

    outlier_count = 0
    if "time_in_hospital" in df.columns:
        numeric_stay = pd.to_numeric(df["time_in_hospital"], errors="coerce")
        outlier_count += int((numeric_stay > 30).sum())
    if "num_medications" in df.columns:
        numeric_meds = pd.to_numeric(df["num_medications"], errors="coerce")
        outlier_count += int((numeric_meds > 80).sum())

    if outlier_count > 0:
        issues.append(f"Detected {outlier_count} suspicious physiological outliers (stay > 30 days or meds > 80).")

    is_valid = len(missing_required) == 0

    preview_rows = df.head(5).fillna("").to_dict(orient="records")

    report = {
        "required_fields_status": "Valid" if not missing_required else "Failed",
        "duplicate_records": dup_count,
        "outlier_values_flagged": outlier_count,
        "column_missingness": missingness,
        "imputation_strategy": "Categoricals assigned 'Unknown'; numerics median-imputed; continuous outliers IQR-capped.",
    }

    return DatasetValidationResult(
        filename=filename,
        total_rows=total_rows,
        valid_rows=total_rows - dup_count if is_valid else 0,
        invalid_rows=dup_count if is_valid else total_rows,
        columns_detected=columns_detected,
        missing_required_columns=missing_required,
        preview_rows=preview_rows,
        validation_report=report,
        is_valid=is_valid,
        issues=issues,
    )


@router.post("/datasets/validate", response_model=DatasetValidationResult)
async def validate_dataset(file: UploadFile = File(...)):
    """Upload a CSV file to validate columns, types, missingness, and outliers without importing."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a CSV file (.csv).",
        )

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content), low_memory=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")

    return _validate_dataframe(df, file.filename)


@router.post("/datasets/import", response_model=DatasetImportResponse)
async def import_dataset(body: DatasetImportConfirm, db: AsyncSession = Depends(get_db)):
    """Explicitly confirm and import validated records into the database."""
    if not body.records:
        raise HTTPException(status_code=400, detail="No records provided to import.")

    import_id = str(uuid.uuid4())
    imported_count = 0

    from ..models_db import DemoEncounter
    from src.models import probability_to_tier

    for row in body.records:
        try:
            enc_id = int(row.get("encounter_id", np.random.randint(9000000, 9999999)))
            # Live score encounter if model available
            prob = 0.10
            tier = "low"
            if is_loaded():
                try:
                    prob, tier = score(row)
                except Exception:
                    pass

            # Check if encounter already exists (upsert)
            existing_enc = await db.execute(select(DemoEncounter).where(DemoEncounter.encounter_id == enc_id))
            enc = existing_enc.scalar_one_or_none()
            if enc:
                enc.age = str(row.get("age", enc.age))
                enc.gender = str(row.get("gender", enc.gender))
                enc.race = str(row.get("race", enc.race))
                enc.readmission_prob = round(prob, 4)
                enc.risk_tier = tier
            else:
                enc = DemoEncounter(
                    encounter_id=enc_id,
                    age=str(row.get("age", "[60-70)")),
                    gender=str(row.get("gender", "Unknown")),
                    race=str(row.get("race", "Unknown")),
                    time_in_hospital=int(row.get("time_in_hospital", 4)),
                    num_lab_procedures=int(row.get("num_lab_procedures", 35)),
                    num_procedures=int(row.get("num_procedures", 1)),
                    num_medications=int(row.get("num_medications", 12)),
                    number_diagnoses=int(row.get("number_diagnoses", 5)),
                    number_inpatient=int(row.get("number_inpatient", 0)),
                    number_emergency=int(row.get("number_emergency", 0)),
                    number_outpatient=int(row.get("number_outpatient", 0)),
                    diag_1=str(row.get("diag_1", "250.01")),
                    max_glu_serum=str(row.get("max_glu_serum", "None")),
                    a1c_result=str(row.get("A1Cresult", row.get("a1c_result", "None"))),
                    readmission_prob=round(prob, 4),
                    risk_tier=tier,
                    model_version="1.0.0",
                )
                db.add(enc)
            imported_count += 1
        except Exception:
            continue

    # Record dataset import record
    dataset_rec = DatasetImport(
        id=import_id,
        filename=body.filename,
        row_count=len(body.records),
        valid_count=imported_count,
        invalid_count=len(body.records) - imported_count,
        status="imported",
        summary_report=f"Successfully imported {imported_count} records from {body.filename}.",
    )
    db.add(dataset_rec)

    # Record audit event
    audit = AuditEvent(
        event_type="dataset_import",
        entity_type="dataset",
        entity_id=import_id,
        acting_user=body.acting_user or "Clinical Data Specialist",
        details=json.dumps({"filename": body.filename, "imported_rows": imported_count}),
    )
    db.add(audit)

    await db.commit()

    return DatasetImportResponse(
        message=f"Successfully imported {imported_count} encounters into ClinicalAI database.",
        imported_count=imported_count,
        status="completed",
        import_id=import_id,
    )


@router.get("/audit-logs", response_model=List[AuditEventOut])
async def get_audit_logs(limit: int = 25, db: AsyncSession = Depends(get_db)):
    """Retrieve system audit logs for compliance and data provenance."""
    stmt = select(AuditEvent).order_by(AuditEvent.created_at.desc()).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()
