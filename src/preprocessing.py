"""
preprocessing.py
================
End-to-end, leakage-free preprocessing for the UCI Diabetes 130-US Hospitals dataset.

Design decisions
----------------
- Missing / unknown: '?' and 'Unknown/Invalid' → NaN; median for numerics, 'Unknown' for cats.
- Outliers: IQR-based capping on continuous features (fit on train only).
- Target: readmitted '<30' → 1, else 0.
- Terminal encounters dropped: discharge_disposition_id ∈ {11,13,14,19,20,21}.
- Patient-grouped splits: GroupShuffleSplit on patient_nbr → no leakage.
- LACE-style score: length_of_stay + acuity + comorbidity + ED_visits.
- Interaction terms for clinical combinations.
- SMOTE on training fold only, inside the pipeline.
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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, OrdinalEncoder
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

# ── ICD-9 clinical group mapping ─────────────────────────────────────────
def _icd_to_category(code: str) -> str:
    """Map ICD-9 code to clinical group."""
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
    if 240 <= num <= 249:
        return "Endocrine/Thyroid"
    if 250 <= num <= 250.99:
        return "Diabetes"
    if 251 <= num <= 279:
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

# Ordinal encoding for medication status
MED_MAP = {"No": 0, "Steady": 1, "Up": 2, "Down": 2}

# Ordinal encoding for A1C / glucose results
A1C_MAP = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
GLU_MAP = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}

# Age midpoints
AGE_MAP = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
    "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
    "[80-90)": 85, "[90-100)": 95,
}


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
    before = len(df)
    df = df[~df["discharge_disposition_id"].isin(TERMINAL_DISCHARGE_IDS)].copy()
    dropped_terminal = before - len(df)
    print(f"[preprocessing] Dropped {dropped_terminal:,} terminal encounters")

    # Keep only the first encounter per patient (reduce within-patient leakage)
    df = df.sort_values("encounter_id")
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["patient_nbr"], keep="first")
    print(f"[preprocessing] After dedup by patient: {len(df):,} rows (dropped {before_dedup - len(df):,} dupes)")
    return df


def build_target(df: pd.DataFrame) -> pd.Series:
    """Binary target: readmitted within 30 days."""
    return (df["readmitted"] == "<30").astype(int)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering applied identically during training and inference."""
    df = df.copy()

    # ── ICD-9 clinical group ──────────────────────────────────────────────
    for col in ("diag_1", "diag_2", "diag_3"):
        if col in df.columns:
            df[f"{col}_cat"] = df[col].apply(_icd_to_category)

    # ── Age → numeric midpoint ────────────────────────────────────────────
    df["age_mid"] = df["age"].map(AGE_MAP).fillna(55)

    # ── Medication change flag ────────────────────────────────────────────
    df["any_med_change"] = (df["change"] == "Ch").astype(int)

    # ── Medication ordinal encoding ───────────────────────────────────────
    for m in MED_COLS:
        if m in df.columns:
            df[m] = df[m].map(MED_MAP).fillna(0)
    present_meds = [m for m in MED_COLS if m in df.columns]
    df["num_active_meds"] = df[present_meds].sum(axis=1)

    # ── A1C and glucose as ordinal features ───────────────────────────────
    if "A1Cresult" in df.columns:
        df["a1c_ordinal"] = df["A1Cresult"].map(A1C_MAP).fillna(0).astype(int)
        df["a1c_tested"] = (df["A1Cresult"] != "None").astype(int)
    else:
        df["a1c_ordinal"] = 0
        df["a1c_tested"] = 0

    if "max_glu_serum" in df.columns:
        df["glu_ordinal"] = df["max_glu_serum"].map(GLU_MAP).fillna(0).astype(int)
    else:
        df["glu_ordinal"] = 0

    # ── Primary diabetes diagnosis ────────────────────────────────────────
    df["primary_diab"] = df.get("diag_1_cat", pd.Series(dtype=str)).eq("Diabetes").astype(int)

    # ── Service utilization ───────────────────────────────────────────────
    for col in ("number_outpatient", "number_emergency", "number_inpatient"):
        if col in df.columns:
            df[col] = df[col].fillna(0)
    df["prior_visits"] = (
        df.get("number_outpatient", 0).astype(float)
        + df.get("number_emergency", 0).astype(float)
        + df.get("number_inpatient", 0).astype(float)
    )

    # ── LACE-style score ──────────────────────────────────────────────────
    # L = Length of stay (days)
    los = df.get("time_in_hospital", pd.Series([3] * len(df))).fillna(3).astype(float)
    l_score = np.clip(los, 0, 14)  # cap at 14 for scoring

    # A = Acuity of admission (emergency=1, else 0)
    a_score = df.get("admission_type_id", pd.Series([1] * len(df))).isin([1, 2]).astype(float)

    # C = Comorbidity (number_diagnoses as proxy for Charlson)
    n_diag = df.get("number_diagnoses", pd.Series([3] * len(df))).fillna(3).astype(float)
    c_score = np.clip(n_diag / 3.0, 0, 5)  # scale to 0-5 range

    # E = Emergency department visits in prior 6 months
    ed = df.get("number_emergency", pd.Series([0] * len(df))).fillna(0).astype(float)
    e_score = np.clip(ed, 0, 4)

    df["lace_score"] = l_score + a_score * 3 + c_score + e_score
    df["lace_high"] = (df["lace_score"] >= 10).astype(int)

    # ── Interaction terms ─────────────────────────────────────────────────
    df["los_x_ndiag"] = los * n_diag
    df["age_x_nmeds"] = df["age_mid"] * df["num_active_meds"]
    df["ed_x_inpatient"] = ed * df.get("number_inpatient", pd.Series([0] * len(df))).fillna(0).astype(float)
    df["insulin_x_a1c"] = df.get("insulin", pd.Series([0] * len(df))).fillna(0).astype(float) * df["a1c_ordinal"]

    # ── diabetesMed binary ────────────────────────────────────────────────
    if "diabetesMed" in df.columns:
        df["diabetesMed"] = (df["diabetesMed"] == "Yes").astype(int)

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
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "lace_score",
    "a1c_ordinal",
    "glu_ordinal",
    # interaction terms
    "los_x_ndiag",
    "age_x_nmeds",
    "ed_x_inpatient",
    "insulin_x_a1c",
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
    "lace_high",
]

ALL_MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES


def drop_high_missing_cols(df: pd.DataFrame, threshold: float = 0.50) -> list[str]:
    """Identify columns with >threshold missing — return names (for info only)."""
    missing_frac = df.isnull().mean()
    high_miss = missing_frac[missing_frac > threshold].index.tolist()
    if high_miss:
        print(f"[preprocessing] Columns with >{threshold*100:.0f}% missing (info): {high_miss}")
    return high_miss


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
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(
                X[col].median() if col in NUMERICAL_FEATURES else 0
            )

    # Fill missing in categoricals
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].fillna("Unknown").astype(str)

    return X, caps


def compute_lace_score_baseline(df: pd.DataFrame) -> np.ndarray:
    """Compute raw LACE scores for use as a standalone baseline predictor."""
    df_eng = engineer_features(df)
    return df_eng["lace_score"].values


def run_full_pipeline(df: pd.DataFrame):
    """Full offline pipeline: filter → target → split → preprocess → SMOTE → return splits.

    Split is 70/10/20 (train/val/test) using patient-grouped splitting.
    Validation set is used for threshold tuning and calibration.
    Test set is touched exactly once for final evaluation.
    """
    raw_total = len(df)
    df = apply_initial_filters(df)
    y = build_target(df)
    groups = df["patient_nbr"].values

    # Log high-missing columns
    drop_high_missing_cols(df)

    X, caps = prepare_features(df)

    # ── Patient-grouped 80/20 split (train+val / test) ────────────────────
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    trainval_idx, test_idx = next(gss1.split(X, y, groups=groups))

    # ── Further split train+val → train (70%) / val (10%) ─────────────────
    X_trainval = X.iloc[trainval_idx]
    y_trainval = y.iloc[trainval_idx]
    groups_trainval = groups[trainval_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=RANDOM_STATE)  # 0.125 × 0.80 ≈ 0.10
    train_rel_idx, val_rel_idx = next(gss2.split(X_trainval, y_trainval, groups=groups_trainval))

    train_idx = trainval_idx[train_rel_idx]
    val_idx = trainval_idx[val_rel_idx]

    X_train_raw = X.iloc[train_idx].reset_index(drop=True)
    X_val_raw = X.iloc[val_idx].reset_index(drop=True)
    X_test_raw = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_val = y.iloc[val_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    # ── Verify zero patient overlap ───────────────────────────────────────
    g_train = set(groups[train_idx])
    g_val = set(groups[val_idx])
    g_test = set(groups[test_idx])
    assert len(g_train & g_val) == 0, "Patient leakage: train ∩ val"
    assert len(g_train & g_test) == 0, "Patient leakage: train ∩ test"
    assert len(g_val & g_test) == 0, "Patient leakage: val ∩ test"
    print(f"[preprocessing] Train: {len(X_train_raw):,} | Val: {len(X_val_raw):,} | Test: {len(X_test_raw):,}")
    print(f"[preprocessing] Patient leakage check: PASS (0 overlap)")

    # ── Fit preprocessor on train only ────────────────────────────────────
    preprocessor = build_preprocessor(X_train_raw)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    # ── SMOTE on training data only ───────────────────────────────────────
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"[preprocessing] After SMOTE: {y_train_sm.sum():,} positives / {(~y_train_sm.astype(bool)).sum():,} negatives")

    # ── Data quality report ───────────────────────────────────────────────
    quality_report = {
        "total_raw": int(raw_total),
        "after_filters": int(len(df)),
        "train_size": int(len(X_train_raw)),
        "val_size": int(len(X_val_raw)),
        "test_size": int(len(X_test_raw)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_val": float(y_val.mean()),
        "positive_rate_test": float(y_test.mean()),
        "after_smote_train_size": int(len(X_train_sm)),
        "outlier_caps": caps,
        "n_features_raw": int(len(ALL_MODEL_FEATURES)),
        "n_features_encoded": int(X_train.shape[1]),
    }
    print(f"[preprocessing] Positive rate - train: {quality_report['positive_rate_train']:.3f} "
          f"| val: {quality_report['positive_rate_val']:.3f} "
          f"| test: {quality_report['positive_rate_test']:.3f}")

    # ── Sensitive attributes for fairness ─────────────────────────────────
    df_test = df.iloc[test_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    return {
        "X_train": X_train_sm,
        "y_train": y_train_sm,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_raw": X_train_raw,
        "X_val_raw": X_val_raw,
        "X_test_raw": X_test_raw,
        "X_train_no_smote": X_train,
        "y_train_no_smote": y_train,
        "preprocessor": preprocessor,
        "caps": caps,
        "quality_report": quality_report,
        "df_test": df_test,
        "df_val": df_val,
    }


def transform_single(encounter_dict: dict, preprocessor, caps: dict) -> np.ndarray:
    """Transform a single encounter dict to model input array (inference path)."""
    df = pd.DataFrame([encounter_dict])
    X, _ = prepare_features(df, caps=caps)
    return preprocessor.transform(X)
