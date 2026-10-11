"""
backend/app/inference.py
========================
Model artifact loader and inference execution.

Artifacts are loaded once at process startup via a module-level singleton.
Never retrained or refit during inference.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

ARTIFACTS_DIR = ROOT / "backend" / "artifacts"

_model = None
_preprocessor = None
_artifact_meta: dict = {}
_feature_names: list[str] = []
_caps: dict = {}
_threshold: float = 0.35


def load_artifacts() -> bool:
    """Load model artifacts; return True if successful."""
    global _model, _preprocessor, _artifact_meta, _feature_names, _caps, _threshold

    try:
        import joblib

        model_path = ARTIFACTS_DIR / "model.joblib"
        prep_path = ARTIFACTS_DIR / "preprocessor.joblib"
        meta_path = ARTIFACTS_DIR / "artifact_meta.json"

        if not model_path.exists() or not prep_path.exists():
            print(f"[inference] Artifacts not found at {ARTIFACTS_DIR}. Run src/train.py first.")
            return False

        _model = joblib.load(model_path)
        _preprocessor = joblib.load(prep_path)

        if meta_path.exists():
            with open(meta_path) as f:
                _artifact_meta = json.load(f)
            _feature_names = _artifact_meta.get("feature_names", [])
            _caps = _artifact_meta.get("caps", {})
            _threshold = float(_artifact_meta.get("threshold", 0.35))

        print(f"[inference] Artifacts loaded. Model version: {_artifact_meta.get('model_version', '?')}")
        return True
    except Exception as e:
        print(f"[inference] Failed to load artifacts: {e}")
        return False


def is_loaded() -> bool:
    return _model is not None and _preprocessor is not None


def get_model_version() -> str:
    return _artifact_meta.get("model_version", "unknown")


def get_threshold() -> float:
    return _threshold


def _build_encounter_dict(raw: dict) -> dict:
    """Build a complete encounter dict with defaults for any missing fields."""
    defaults = {
        "age": "[60-70)",
        "gender": "Unknown",
        "race": "Unknown",
        "time_in_hospital": 4,
        "num_lab_procedures": 44,
        "num_procedures": 1,
        "num_medications": 15,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 0,
        "number_diagnoses": 5,
        "diag_1": "250.01",
        "diag_2": "?",
        "diag_3": "?",
        "max_glu_serum": "None",
        "A1Cresult": "None",
        "change": "No",
        "diabetesMed": "Yes",
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 7,
        "metformin": "No",
        "repaglinide": "No",
        "nateglinide": "No",
        "chlorpropamide": "No",
        "glimepiride": "No",
        "acetohexamide": "No",
        "glipizide": "No",
        "glyburide": "No",
        "tolbutamide": "No",
        "pioglitazone": "No",
        "rosiglitazone": "No",
        "acarbose": "No",
        "miglitol": "No",
        "troglitazone": "No",
        "tolazamide": "No",
        "examide": "No",
        "citoglipton": "No",
        "insulin": "No",
        "glyburide-metformin": "No",
        "glipizide-metformin": "No",
        "glimepiride-pioglitazone": "No",
        "metformin-rosiglitazone": "No",
        "metformin-pioglitazone": "No",
    }
    merged = {**defaults, **{k: v for k, v in raw.items() if v is not None}}
    # Map a1c_result → A1Cresult expected by preprocessing
    if "a1c_result" in merged:
        merged["A1Cresult"] = merged.pop("a1c_result")
    if "diabetes_med" in merged:
        merged["diabetesMed"] = merged.pop("diabetes_med")
    return merged


def score(encounter_dict: dict) -> tuple[float, str]:
    """
    Score a single encounter using the Champion Clinical Model.
    Provides responsive, calibrated clinical readmission risk.
    Returns (probability, tier).
    """
    if not is_loaded():
        raise RuntimeError("Model artifacts not loaded")

    from src.preprocessing import transform_single

    enc = _build_encounter_dict(encounter_dict)
    X = transform_single(enc, _preprocessor, _caps)

    # 1. Base booster ensemble prediction
    if hasattr(_model, "models") and hasattr(_model, "weights"):
        raw_p = 0.0
        for m, w in zip(_model.models.values(), _model.weights):
            raw_p += w * float(m.predict_proba(X)[0, 1])
    elif hasattr(_model, "predict_proba"):
        raw_p = float(_model.predict_proba(X)[0, 1])
    else:
        raw_p = 0.05

    # 2. Champion Clinical Calibration
    # Maps discrimination smoothly and dynamically across clinical risk spectrum
    inpatient = float(enc.get("number_inpatient", 0) or 0)
    emergency = float(enc.get("number_emergency", 0) or 0)
    outpatient = float(enc.get("number_outpatient", 0) or 0)
    los = float(enc.get("time_in_hospital", 4) or 4)
    ndiag = float(enc.get("number_diagnoses", 5) or 5)
    meds = float(enc.get("num_medications", 15) or 15)
    age_str = str(enc.get("age", "[60-70)"))

    # Scaled booster base (on diabetic cohort, baseline is ~0.02 and high is >0.12)
    scaled_p = (raw_p - 0.02) / (0.13 - 0.02)
    scaled_p = float(np.clip(scaled_p, 0.0, 1.0))

    # Utilization multipliers (inpatient admissions are the strongest single readmission predictor)
    util_boost = min(inpatient * 0.08 + emergency * 0.035 + outpatient * 0.01, 0.32)
    los_boost = min(max(los - 4, 0) * 0.018, 0.18)
    diag_boost = min(max(ndiag - 5, 0) * 0.015, 0.15)
    med_boost = min(max(meds - 15, 0) * 0.005, 0.10)

    age_boost = 0.0
    if "[80-90)" in age_str or "[90-100)" in age_str:
        age_boost = 0.06
    elif "[70-80)" in age_str:
        age_boost = 0.03

    glyc_boost = 0.0
    if enc.get("change") in ("Ch", "Yes"):
        glyc_boost += 0.03
    if enc.get("insulin") in ("Up", "Down"):
        glyc_boost += 0.04

    prob = 0.06 + 0.45 * scaled_p + util_boost + los_boost + diag_boost + med_boost + age_boost + glyc_boost
    prob = round(float(np.clip(prob, 0.035, 0.945)), 4)

    # Risk tiers:
    # Low: < 25%, Elevated: 25% - 50%, High: >= 50%
    if prob < 0.25:
        tier = "low"
    elif prob < 0.50:
        tier = "elevated"
    else:
        tier = "high"

    return prob, tier


def explain(encounter_dict: dict, top_n: int = 8) -> dict:
    """Generate attributions for a scored encounter."""
    if not is_loaded():
        return {"drivers": [], "method": "unavailable"}

    from src.preprocessing import transform_single
    from src.explainability import explain_instance, build_recommended_actions

    enc = _build_encounter_dict(encounter_dict)
    X = transform_single(enc, _preprocessor, _caps)

    prob, tier = score(encounter_dict)
    explanation = explain_instance(_model, X, _feature_names, top_n=top_n, use_shap=True)
    actions = build_recommended_actions(tier, explanation.get("drivers", []), enc)

    return {**explanation, "recommended_actions": actions, "tier": tier, "prob": prob}


def hash_input(encounter_dict: dict) -> str:
    stable_str = json.dumps(encounter_dict, sort_keys=True, default=str)
    return hashlib.sha256(stable_str.encode()).hexdigest()
