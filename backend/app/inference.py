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
    Score a single encounter.
    Returns (probability, tier).
    """
    if not is_loaded():
        raise RuntimeError("Model artifacts not loaded")

    from src.preprocessing import transform_single
    from src.models import probability_to_tier

    enc = _build_encounter_dict(encounter_dict)
    X = transform_single(enc, _preprocessor, _caps)
    prob = float(_model.predict_proba(X)[0, 1])
    tier = probability_to_tier(prob)
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
