"""
train.py
========
End-to-end training script. Run once to produce all model artifacts.

Usage
-----
    python src/train.py

The script will:
1. Download the UCI Diabetes dataset (via ucimlrepo) if not already present.
2. Apply preprocessing, patient-grouped splits, SMOTE.
3. Train Logistic Regression, Random Forest, and XGBoost.
4. Compare models on the test split.
5. Calibrate the best model.
6. Run fairness audit.
7. Save all artifacts to backend/artifacts/.
8. Seed the database with demo encounters (if DATABASE_URL is set).
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import (
    DATA_DIR,
    ARTIFACTS_DIR,
    ALL_MODEL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    load_raw_data,
    run_full_pipeline,
)
from src.models import compare_models, save_artifacts, get_feature_importance, probability_to_tier
from src.fairness import audit_fairness, run_threshold_optimizer_comparison, age_to_group


def download_dataset():
    """Download UCI dataset using ucimlrepo if not cached."""
    csv_path = DATA_DIR / "diabetic_data.csv"
    if csv_path.exists():
        print(f"[train] Dataset already exists at {csv_path}")
        return

    print("[train] Downloading UCI Diabetes 130-US Hospitals dataset...")
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=296)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"[train] Saved {len(df):,} rows -> {csv_path}")
    except Exception as e:
        print(f"[train] ucimlrepo download failed: {e}")
        print("[train] Trying direct URL download...")
        import urllib.request
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"
        import zipfile, io
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                zf = zipfile.ZipFile(io.BytesIO(resp.read()))
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                zf.extractall(DATA_DIR)
                # Move if nested
                for f in DATA_DIR.rglob("diabetic_data.csv"):
                    if f != csv_path:
                        import shutil
                        shutil.copy(f, csv_path)
            print(f"[train] Downloaded to {csv_path}")
        except Exception as e2:
            print(f"[train] All download attempts failed: {e2}")
            print("[train] Creating synthetic dataset for demonstration...")
            _create_synthetic_dataset(csv_path)


def _create_synthetic_dataset(csv_path: Path):
    """Create a synthetic dataset that mirrors UCI structure for demo purposes."""
    np.random.seed(42)
    n = 50000
    rng = np.random.default_rng(42)

    ages = rng.choice(
        ["[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[30-40)"],
        n, p=[0.15, 0.20, 0.25, 0.22, 0.12, 0.06]
    )
    race = rng.choice(
        ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"],
        n, p=[0.76, 0.19, 0.02, 0.01, 0.02]
    )
    gender = rng.choice(["Male", "Female"], n, p=[0.46, 0.54])
    
    time_in = rng.integers(1, 15, n)
    num_lab = rng.integers(1, 120, n)
    num_meds = rng.integers(1, 80, n)
    num_diag = rng.integers(1, 9, n)
    num_proc = rng.integers(0, 7, n)
    
    readmitted_prob = (
        0.08
        + 0.10 * (rng.random(n) < 0.3)  # prior inpatient
        + 0.05 * (num_meds > 15) / 1
        + 0.05 * (num_diag > 5) / 1
        + 0.03 * (time_in > 7) / 1
    )
    readmitted_prob = np.clip(readmitted_prob, 0, 0.35)
    readmitted_val = rng.random(n) < readmitted_prob
    readmitted_label = np.where(readmitted_val, "<30", 
                                np.where(rng.random(n) < 0.5, ">30", "NO"))

    df = pd.DataFrame({
        "encounter_id": range(1, n + 1),
        "patient_nbr": rng.integers(1, 30000, n),
        "race": race,
        "gender": gender,
        "age": ages,
        "weight": "?",
        "admission_type_id": rng.integers(1, 9, n),
        "discharge_disposition_id": rng.integers(1, 30, n),
        "admission_source_id": rng.integers(1, 26, n),
        "time_in_hospital": time_in,
        "payer_code": rng.choice(["MC", "MD", "HM", "?"], n),
        "medical_specialty": rng.choice(["InternalMedicine", "Emergency/Trauma", "Cardiology", "?"], n),
        "num_lab_procedures": num_lab,
        "num_procedures": num_proc,
        "num_medications": num_meds,
        "number_outpatient": rng.integers(0, 10, n),
        "number_emergency": rng.integers(0, 8, n),
        "number_inpatient": rng.integers(0, 10, n),
        "diag_1": rng.choice(["250.01", "250.83", "410.01", "428.0", "496", "?"], n),
        "diag_2": rng.choice(["250.00", "401.9", "496", "272.4", "?"], n),
        "diag_3": rng.choice(["?", "272.4", "401.9", "250.00"], n),
        "number_diagnoses": num_diag,
        "max_glu_serum": rng.choice(["None", ">200", ">300", "Norm"], n, p=[0.95, 0.025, 0.015, 0.01]),
        "A1Cresult": rng.choice(["None", ">7", ">8", "Norm"], n, p=[0.82, 0.07, 0.05, 0.06]),
        "metformin": rng.choice(["No", "Steady", "Up", "Down"], n, p=[0.52, 0.41, 0.04, 0.03]),
        "repaglinide": rng.choice(["No", "Steady", "Up", "Down"], n, p=[0.90, 0.08, 0.01, 0.01]),
        "nateglinide": "No",
        "chlorpropamide": "No",
        "glimepiride": rng.choice(["No", "Steady", "Up", "Down"], n, p=[0.82, 0.14, 0.02, 0.02]),
        "acetohexamide": "No",
        "glipizide": rng.choice(["No", "Steady", "Up", "Down"], n, p=[0.71, 0.24, 0.03, 0.02]),
        "glyburide": rng.choice(["No", "Steady", "Up", "Down"], n, p=[0.75, 0.21, 0.02, 0.02]),
        "tolbutamide": "No",
        "pioglitazone": rng.choice(["No", "Steady", "Up", "Down"], n, p=[0.85, 0.12, 0.02, 0.01]),
        "rosiglitazone": rng.choice(["No", "Steady", "Up", "Down"], n, p=[0.90, 0.08, 0.01, 0.01]),
        "acarbose": "No",
        "miglitol": "No",
        "troglitazone": "No",
        "tolazamide": "No",
        "examide": "No",
        "citoglipton": "No",
        "insulin": rng.choice(["No", "Steady", "Up", "Down"], n, p=[0.47, 0.33, 0.12, 0.08]),
        "glyburide-metformin": "No",
        "glipizide-metformin": "No",
        "glimepiride-pioglitazone": "No",
        "metformin-rosiglitazone": "No",
        "metformin-pioglitazone": "No",
        "change": rng.choice(["No", "Ch"], n, p=[0.54, 0.46]),
        "diabetesMed": rng.choice(["Yes", "No"], n, p=[0.78, 0.22]),
        "readmitted": readmitted_label,
    })
    # Remove terminal discharge IDs from synthetic data
    df.loc[df["discharge_disposition_id"].isin({11, 13, 14, 19, 20, 21}), "discharge_disposition_id"] = 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[train] Synthetic dataset created: {len(df):,} rows → {csv_path}")


def seed_db_encounters(df_test: pd.DataFrame, probs: np.ndarray, threshold: float):
    """Write demo encounters to DB (PostgreSQL if DATABASE_URL is set, else SQLite demo.db)."""
    import asyncio
    from backend.app.database import engine, init_db, AsyncSessionLocal
    from backend.app.models_db import DemoEncounter, CarePlan, CarePlanAction
    from sqlalchemy import select, delete

    async def _async_seed():
        await init_db()
        async with AsyncSessionLocal() as session:
            await session.execute(delete(CarePlanAction))
            await session.execute(delete(CarePlan))
            await session.execute(delete(DemoEncounter))
            await session.commit()

            for i, (_, row) in enumerate(df_test.head(300).iterrows()):
                prob = float(probs[i]) if i < len(probs) else 0.15
                tier = probability_to_tier(prob)
                enc_id = int(row.get("encounter_id", 100000 + i))

                enc = DemoEncounter(
                    encounter_id=enc_id,
                    age=str(row.get("age", "[60-70)")),
                    gender=str(row.get("gender", "Unknown")),
                    race=str(row.get("race", "Unknown")),
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
                    a1c_result=str(row.get("A1Cresult", row.get("a1c_result", "None"))),
                    change=str(row.get("change", "No")),
                    diabetes_med=str(row.get("diabetesMed", row.get("diabetes_med", "Yes"))),
                    readmission_prob=round(prob, 4),
                    risk_tier=tier,
                    model_version="1.0.0",
                )
                session.add(enc)

            await session.commit()

            # Seed care plans for top high-risk encounters
            result = await session.execute(
                select(DemoEncounter).order_by(DemoEncounter.readmission_prob.desc()).limit(35)
            )
            top_encs = result.scalars().all()
            for idx, enc in enumerate(top_encs):
                status = "completed" if idx < 10 else ("in_progress" if idx < 22 else "pending")
                plan = CarePlan(
                    encounter_id=enc.encounter_id,
                    status=status,
                    acting_user="Dr. Martinez (Clinical Coordinator)",
                )
                session.add(plan)
                await session.flush()

                actions_data = [
                    ("medication_reconciliation", "Medication Reconciliation", idx < 22),
                    ("followup_appointment", "Schedule Follow-up Appointment", idx < 15),
                    ("diabetes_education", "Diabetes Self-Management Education", idx < 10),
                    ("telehealth_followup", "Telehealth Follow-up (48–72h)", idx < 5),
                ]
                for act_key, act_label, is_done in actions_data:
                    session.add(
                        CarePlanAction(
                            care_plan_id=plan.id,
                            action_key=act_key,
                            label=act_label,
                            completed=is_done,
                        )
                    )

            await session.commit()
            print(f"[train] Seeded 300 demo encounters + 35 active care plans to DB.")

    try:
        asyncio.run(_async_seed())
    except Exception as e:
        print(f"[train] DB seed warning: {e}")


def run_training():
    download_dataset()

    df = load_raw_data()
    if df is None:
        print("[train] Could not load dataset. Aborting.")
        sys.exit(1)

    print(f"[train] Raw data shape: {df.shape}")

    pipeline_outputs = run_full_pipeline(df)
    X_train = pipeline_outputs["X_train"]
    y_train = pipeline_outputs["y_train"]
    X_test = pipeline_outputs["X_test"]
    y_test = pipeline_outputs["y_test"]
    X_train_raw = pipeline_outputs["X_train_raw"]
    preprocessor = pipeline_outputs["preprocessor"]
    caps = pipeline_outputs["caps"]
    quality_report = pipeline_outputs["quality_report"]
    df_test = pipeline_outputs["df_test"]

    print(f"\n[train] X_train: {X_train.shape}, X_test: {X_test.shape}")

    # ── model comparison ─────────────────────────────────────────────────
    metrics, calib_model, threshold = compare_models(X_train, y_train, X_test, y_test)

    # ── feature names after preprocessing ───────────────────────────────
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "cat" and hasattr(trans, "get_feature_names_out"):
            feature_names.extend(trans.get_feature_names_out(cols).tolist())
        elif name in ("num", "bin"):
            feature_names.extend(cols)

    # ── fairness audit ───────────────────────────────────────────────────
    print("\n[train] Running fairness audit...")
    from src.preprocessing import engineer_features

    df_test_eng = engineer_features(df_test)
    df_test_eng["age_group"] = df_test_eng["age_mid"].apply(age_to_group)

    sensitive_df = df_test_eng[["age_group", "gender", "race"]].reset_index(drop=True)
    probs_test = calib_model.predict_proba(X_test)[:, 1]

    fairness_results = audit_fairness(
        y_true=y_test.values,
        y_prob=probs_test,
        sensitive_df=sensitive_df,
        threshold=threshold,
    )

    mitigation_comparison = run_threshold_optimizer_comparison(
        X_train, y_train, X_test, y_test, sensitive_df["race"], calib_model
    )

    # ── feature importances ──────────────────────────────────────────────
    importances = get_feature_importance(calib_model, feature_names)

    # ── governance report ────────────────────────────────────────────────
    governance_report = {
        "model_comparison": metrics,
        "fairness": fairness_results,
        "mitigation_comparison": mitigation_comparison,
        "feature_importances": importances,
        "data_quality": quality_report,
        "feature_names": feature_names,
    }

    with open(ARTIFACTS_DIR / "governance_report.json", "w") as f:
        json.dump(governance_report, f, indent=2, default=str)
    print(f"[train] Governance report saved.")

    # ── save all artifacts ───────────────────────────────────────────────
    save_artifacts(calib_model, preprocessor, caps, metrics, feature_names, threshold)

    # ── seed DB ─────────────────────────────────────────────────────────
    seed_db_encounters(df_test, probs_test, threshold)

    print("\n[train] [SUCCESS] Training complete!")
    print(f"  Selected model : XGBoost (calibrated)")
    print(f"  Threshold      : {threshold:.2f}")
    print(f"  Test AUC       : {metrics['xgboost_calibrated']['roc_auc']:.3f}")
    print(f"  Test Recall    : {metrics['xgboost_calibrated']['recall']:.3f}")
    print(f"  Test Precision : {metrics['xgboost_calibrated']['precision']:.3f}")
    print(f"  Artifacts      : {ARTIFACTS_DIR}")


if __name__ == "__main__":
    run_training()
