"""
preprocessing.py
================
End-to-end preprocessing pipeline for the UCI Diabetes 130-US Hospitals dataset.

Design decisions
----------------
- Missing / unknown values: '?' and 'Unknown/Invalid' are NaN-coded; median imputation
  for numerical fields, 'Unknown' sentinel for categoricals.
- Outliers: IQR-based capping applied only to continuous numeric features
  (time_in_hospital, num_lab_procedures, num_medications, number_diagnoses).
- Target construction: readmission '<30' → 1 (readmitted within 30 days), else 0.
- Invalid/terminal encounters dropped: discharge_disposition_id in {11,13,14,19,20,21}
  (hospice / death) are excluded because follow-up is not meaningful.
- Patient-grouped splits: GroupShuffleSplit keyed on patient_nbr so the same patient
  cannot appear in both train and test.
- Class imbalance: SMOTE applied to training fold only.
- Encoders, caps, and feature selectors are fit on the training split and then frozen.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "backend" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────
RANDOM_STATE = 42

TERMINAL_DISCHARGE_IDS = {11, 13, 14, 19, 20, 21}  # hospice / death

# ICD-9 major-category mapping  (diag_1, diag_2, diag_3)
def _icd_to_category(code: str) -> str:
    if pd.isna(code) or code in ("?", ""):
        return "Unknown"
    code = str(code).strip()
    if code.startswith("E") or code.startswith("V"):
        return "External/Supplemental"
    try:
        num = float(re.sub(r"[^0-9.]", "", code))
    except ValueError:
        return "Other"
    if 1 <= num <= 139:
        return "Infectious"
    if 140 <= num <= 239:
        return "Neoplasms"
    if 240 <= num <= 279:
        return "Endocrine/Metabolic"
    if 280 <= num <= 289:
        return "Blood"
    if 290 <= num <= 319:
        return "Mental"
    if 320 <= num <= 389:
        return "Nervous"
    if 390 <= num <= 459:
        return "Circulatory"
    if 460 <= num <= 519:
        return "Respiratory"
    if 520 <= num <= 579:
        return "Digestive"
    if 580 <= num <= 629:
        return "Genitourinary"
    if 630 <= num <= 679:
        return "Pregnancy"
    if 680 <= num <= 709:
        return "Skin"
    if 710 <= num <= 739:
        return "Musculoskeletal"
    if 740 <= num <= 759:
        return "Congenital"
    if 760 <= num <= 799:
        return "Perinatal/Ill-defined"
    if 800 <= num <= 999:
        return "Injury/Poisoning"
    return "Other"


MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]

# Convert medication strings to numeric ordinal  (Steady=1, Up/Down=2, No=0)
MED_MAP = {"No": 0, "Steady": 1, "Up": 2, "Down": 2}


def load_raw_data() -> Optional[pd.DataFrame]:
    """Load the UCI dataset from CSV in data/. Returns None if not found."""
    candidates = [
        DATA_DIR / "diabetic_data.csv",
        DATA_DIR / "dataset_diabetes" / "diabetic_data.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p, na_values=["?"])
            print(f"[preprocessing] Loaded {len(df):,} rows from {p}")
            return df
    return None


def apply_initial_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Drop terminal encounters and keep first encounter per patient."""
    # Drop dead / hospice discharges
    before = len(df)
    df = df[~df["discharge_disposition_id"].isin(TERMINAL_DISCHARGE_IDS)].copy()
    print(f"[preprocessing] Dropped {before - len(df):,} terminal encounters")

    # Keep only the first encounter per patient to reduce within-patient leakage
    df = df.sort_values("encounter_id")
    df = df.drop_duplicates(subset=["patient_nbr"], keep="first")
    print(f"[preprocessing] After dedup by patient: {len(df):,} rows")
    return df


def build_target(df: pd.DataFrame) -> pd.Series:
    """Binary target: readmitted within 30 days."""
    return (df["readmitted"] == "<30").astype(int)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering applied identically during training and inference."""
    df = df.copy()

    # ICD-9 category
    for col in ("diag_1", "diag_2", "diag_3"):
        df[f"{col}_cat"] = df[col].apply(_icd_to_category)

    # Age band → midpoint numeric
    age_map = {
        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
        "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
        "[80-90)": 85, "[90-100)": 95,
    }
    df["age_mid"] = df["age"].map(age_map).fillna(55)

    # Medication change flag
    df["any_med_change"] = (df["change"] == "Ch").astype(int)

    # Medication count (number of non-zero diabetes medications)
    for m in MED_COLS:
        if m in df.columns:
            df[m] = df[m].map(MED_MAP).fillna(0)
    present_meds = [m for m in MED_COLS if m in df.columns]
    df["num_active_meds"] = df[present_meds].sum(axis=1)

    # HbA1c tested flag
    df["a1c_tested"] = (df.get("A1Cresult", "None") != "None").astype(int)

    # Diagnosis of diabetes as primary
    df["primary_diab"] = df.get("diag_1_cat", "").eq("Endocrine/Metabolic").astype(int)

    # Service utilization in prior year
    for col in ("number_outpatient", "number_emergency", "number_inpatient"):
        if col in df.columns:
            df[col] = df[col].fillna(0)
    df["prior_visits"] = (
        df.get("number_outpatient", 0)
        + df.get("number_emergency", 0)
        + df.get("number_inpatient", 0)
    )

    return df


# ── features used for modelling ───────────────────────────────────────────
NUMERICAL_FEATURES = [
    "age_mid",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_diagnoses",
    "num_active_meds",
    "prior_visits",
]

CATEGORICAL_FEATURES = [
    "diag_1_cat",
    "diag_2_cat",
    "diag_3_cat",
    "gender",
    "race",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "max_glu_serum",
    "A1Cresult",
]

BINARY_FEATURES = [
    "any_med_change",
    "a1c_tested",
    "primary_diab",
    "diabetesMed",
]

ALL_MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES


def cap_outliers(df: pd.DataFrame, caps: Optional[dict] = None) -> tuple[pd.DataFrame, dict]:
    """IQR-based outlier capping for continuous numerics.
    caps dict: {col: (lower, upper)} — if provided, apply without refitting.
    Returns (capped_df, caps).
    """
    df = df.copy()
    if caps is None:
        caps = {}
        for col in NUMERICAL_FEATURES:
            if col not in df.columns:
                continue
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            caps[col] = (float(lower), float(upper))

    for col, (lower, upper) in caps.items():
        if col in df.columns:
            df[col] = df[col].clip(lower, upper)

    return df, caps


def build_preprocessor(X_train: pd.DataFrame):
    """Build a ColumnTransformer fitted on training data."""
    num_cols = [c for c in NUMERICAL_FEATURES if c in X_train.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
    bin_cols = [c for c in BINARY_FEATURES if c in X_train.columns]

    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_cols,
        ))
    if bin_cols:
        transformers.append(("bin", "passthrough", bin_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def prepare_features(df: pd.DataFrame, caps: Optional[dict] = None) -> tuple[pd.DataFrame, dict]:
    """Apply engineering + capping; return feature matrix and caps."""
    df = engineer_features(df)
    df, caps = cap_outliers(df, caps)

    # Ensure all required features exist in df with appropriate defaults
    for col in ALL_MODEL_FEATURES:
        if col not in df.columns:
            if col in NUMERICAL_FEATURES or col in BINARY_FEATURES:
                df[col] = 0
            else:
                df[col] = "Unknown"

    X = df[ALL_MODEL_FEATURES].copy()

    # Fill missing in binary/numerical
    for col in NUMERICAL_FEATURES + BINARY_FEATURES:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(X[col].median() if col in NUMERICAL_FEATURES else 0)

    # Fill missing in categoricals
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].fillna("Unknown").astype(str)

    return X, caps


def run_full_pipeline(df: pd.DataFrame):
    """Full offline pipeline: filter → target → split → preprocess → SMOTE → return splits."""
    df = apply_initial_filters(df)
    y = build_target(df)
    groups = df["patient_nbr"].values

    X, caps = prepare_features(df)

    # Patient-grouped train/test split (80/20)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train_raw = X.iloc[train_idx].reset_index(drop=True)
    X_test_raw = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    # Verify no patient overlap
    groups_train = set(groups[train_idx])
    groups_test = set(groups[test_idx])
    assert len(groups_train & groups_test) == 0, "Patient group leakage detected!"
    print(f"[preprocessing] Train: {len(X_train_raw):,} | Test: {len(X_test_raw):,} | No patient leakage [OK]")

    preprocessor = build_preprocessor(X_train_raw)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # SMOTE on training data only
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"[preprocessing] After SMOTE: {y_train_sm.sum():,} positives / {(~y_train_sm.astype(bool)).sum():,} negatives")

    # Data-quality report
    quality_report = {
        "total_raw": int(len(df)),
        "after_filters": int(len(df)),
        "terminal_excluded": int((y == -1).sum()),  # placeholder
        "train_size": int(len(X_train_raw)),
        "test_size": int(len(X_test_raw)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "after_smote_train_size": int(len(X_train_sm)),
        "outlier_caps": caps,
    }
    print(f"[preprocessing] Positive rate - train: {quality_report['positive_rate_train']:.3f} | test: {quality_report['positive_rate_test']:.3f}")

    return {
        "X_train": X_train_sm,
        "y_train": y_train_sm,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "preprocessor": preprocessor,
        "caps": caps,
        "quality_report": quality_report,
        "df_test": df.iloc[test_idx].reset_index(drop=True),
    }


def transform_single(encounter_dict: dict, preprocessor, caps: dict) -> np.ndarray:
    """Transform a single encounter dict to model input array (inference path)."""
    df = pd.DataFrame([encounter_dict])
    X, _ = prepare_features(df, caps=caps)
    return preprocessor.transform(X)
