"""
seed_database.py
================
Seed real encounters from the UCI Diabetes 130-US Hospitals dataset into the database,
scored with the calibrated XGBoost model, alongside realistic care plans and actions.

Supports both PostgreSQL (when DATABASE_URL is set) and SQLite fallback (data/demo.db).
"""

import asyncio
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import delete, select

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "diabetic_data.csv"
ARTIFACTS_DIR = ROOT / "backend" / "artifacts"

import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.database import engine, init_db, AsyncSessionLocal
from backend.app.models_db import DemoEncounter, CarePlan, CarePlanAction
from src.preprocessing import prepare_features
from src.models import probability_to_tier, RISK_TIERS

async def seed_data(sample_size: int = 5000):
    print(f"[seed] Initializing database schema...")
    await init_db()
    
    print(f"[seed] Loading UCI Diabetes dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    # Exclude terminal discharges (hospice / death)
    terminal_ids = {11, 13, 14, 19, 20, 21}
    df = df[~df["discharge_disposition_id"].isin(terminal_ids)].copy()
    
    # Deduplicate by patient to ensure realistic unique patient encounters
    df = df.sort_values("encounter_id").drop_duplicates(subset=["patient_nbr"], keep="first")
    df_sample = df.head(sample_size).copy().reset_index(drop=True)
    print(f"[seed] Selected {len(df_sample):,} eligible unique patient encounters.")
    
    # Load model and preprocessor
    print(f"[seed] Loading ML model and preprocessor...")
    model = joblib.load(ARTIFACTS_DIR / "model.joblib")
    preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")
    meta = json.loads((ARTIFACTS_DIR / "artifact_meta.json").read_text())
    caps = meta.get("caps", {})
    threshold = float(meta.get("threshold", 0.15))
    
    # Transform and score in batch
    print(f"[seed] Transforming features and scoring with calibrated XGBoost...")
    X, _ = prepare_features(df_sample, caps=caps)
    X_trans = preprocessor.transform(X)
    probs = model.predict_proba(X_trans)[:, 1]
    
    print(f"[seed] Scoring complete: min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")
    
    # Populate database
    async with AsyncSessionLocal() as session:
        print("[seed] Clearing previous encounter & care plan data...")
        await session.execute(delete(CarePlanAction))
        await session.execute(delete(CarePlan))
        await session.execute(delete(DemoEncounter))
        await session.commit()
        
        print(f"[seed] Inserting {len(df_sample):,} encounter records in batches...")
        batch_size = 500
        total_inserted = 0
        
        for start_idx in range(0, len(df_sample), batch_size):
            end_idx = min(start_idx + batch_size, len(df_sample))
            batch_df = df_sample.iloc[start_idx:end_idx]
            
            for i, (_, row) in enumerate(batch_df.iterrows()):
                global_idx = start_idx + i
                prob = float(probs[global_idx])
                
                # Determine risk tier based on calibrated thresholds
                if prob >= 0.20:
                    tier = "high"
                elif prob >= 0.12:
                    tier = "elevated"
                else:
                    tier = "low"
                    
                enc = DemoEncounter(
                    encounter_id=int(row["encounter_id"]),
                    age=str(row.get("age", "[60-70)")),
                    gender=str(row.get("gender", "Unknown")),
                    race=str(row.get("race", "Unknown")).replace("?", "Unknown"),
                    admission_type_id=int(row["admission_type_id"]) if pd.notna(row.get("admission_type_id")) else 1,
                    discharge_disposition_id=int(row["discharge_disposition_id"]) if pd.notna(row.get("discharge_disposition_id")) else 1,
                    admission_source_id=int(row["admission_source_id"]) if pd.notna(row.get("admission_source_id")) else 7,
                    time_in_hospital=int(row.get("time_in_hospital", 3)),
                    num_lab_procedures=int(row.get("num_lab_procedures", 30)),
                    num_procedures=int(row.get("num_procedures", 1)),
                    num_medications=int(row.get("num_medications", 10)),
                    number_outpatient=int(row.get("number_outpatient", 0)),
                    number_emergency=int(row.get("number_emergency", 0)),
                    number_inpatient=int(row.get("number_inpatient", 0)),
                    number_diagnoses=int(row.get("number_diagnoses", 5)),
                    diag_1=str(row.get("diag_1", "250.01")),
                    diag_2=str(row.get("diag_2", "?")),
                    diag_3=str(row.get("diag_3", "?")),
                    max_glu_serum=str(row.get("max_glu_serum", "None")),
                    a1c_result=str(row.get("A1Cresult", "None")),
                    change=str(row.get("change", "No")),
                    diabetes_med=str(row.get("diabetesMed", "Yes")),
                    readmission_prob=round(prob, 4),
                    risk_tier=tier,
                    model_version="1.0.0",
                )
                session.add(enc)
            
            await session.commit()
            total_inserted += (end_idx - start_idx)
            print(f"[seed] Inserted {total_inserted:,} / {len(df_sample):,} encounters...")
            
        # Seed care plans for top high-risk encounters
        print("[seed] Seeding active care plans for high-risk encounters...")
        result = await session.execute(
            select(DemoEncounter).order_by(DemoEncounter.readmission_prob.desc()).limit(120)
        )
        top_encs = result.scalars().all()
        
        clinicians = [
            "Dr. Elena Martinez, MD (Clinical Lead)",
            "Sarah Jenkins, RN (Transitional Care Specialist)",
            "Dr. Marcus Vance, PharmD (Clinical Pharmacy)",
            "James Kowalski, LCSW (Discharge Coordinator)",
        ]
        
        for idx, enc in enumerate(top_encs):
            # 30% completed, 45% in_progress, 25% pending
            if idx < 36:
                status = "completed"
            elif idx < 90:
                status = "in_progress"
            else:
                status = "pending"
                
            plan = CarePlan(
                encounter_id=enc.encounter_id,
                status=status,
                acting_user=clinicians[idx % len(clinicians)],
            )
            session.add(plan)
            await session.flush()
            
            # Action items
            actions_list = [
                ("medication_reconciliation", "Medication Reconciliation", idx < 90),
                ("followup_appointment", "Schedule Follow-up Appointment (within 7d)", idx < 65),
                ("diabetes_education", "Diabetes Self-Management Education", idx < 50),
                ("telehealth_followup", "Telehealth Follow-up (48–72h post-discharge)", idx < 36),
                ("pharmacy_consult", "Clinical Pharmacy Regimen Review", idx < 30 and enc.change == "Ch"),
            ]
            
            for act_key, act_label, is_done in actions_list:
                session.add(
                    CarePlanAction(
                        care_plan_id=plan.id,
                        action_key=act_key,
                        label=act_label,
                        completed=is_done,
                    )
                )
                
        await session.commit()
        print(f"[seed] [SUCCESS] Seeded {len(df_sample):,} encounters and {len(top_encs)} care plans into database.")

if __name__ == "__main__":
    asyncio.run(seed_data(sample_size=5000))
