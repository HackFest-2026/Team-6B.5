"""
explainability.py
=================
Deterministic, template-based explanation generation for the readmission model.

Uses SHAP TreeExplainer to compute feature attributions for a given instance.
Falls back to global feature importances if SHAP is unavailable.

Important constraints
---------------------
- No LLM or generative AI is used.
- Attributions are described as associations/contributions, NOT causal effects.
- Factors are classified as modifiable (planning-relevant) or historical/fixed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "backend" / "artifacts"

# ── factor metadata ─────────────────────────────────────────────────────────
FACTOR_META = {
    # feature_name: (readable_label, modifiable, direction_note)
    "num_inpatient_prior": ("Prior inpatient visits", False, "More prior admissions → higher risk"),
    "prior_visits":        ("Total prior visits (outpatient + ED + inpatient)", False, "High utilisation signals complex illness burden"),
    "num_lab_procedures":  ("Number of lab procedures", False, "More labs may indicate acute severity"),
    "num_medications":     ("Number of medications", True,  "High medication burden may flag complexity"),
    "time_in_hospital":    ("Length of stay (days)", False, "Longer stays associated with higher risk"),
    "num_procedures":      ("Number of non-lab procedures", False, "Procedure burden"),
    "number_diagnoses":    ("Number of diagnoses coded", False, "Comorbidity burden"),
    "num_active_meds":     ("Active diabetes medications", True,  "Medication regimen complexity"),
    "any_med_change":      ("Medication change during visit", True,  "Regimen change; reconciliation opportunity"),
    "a1c_tested":          ("HbA1c tested during visit", True,  "Monitoring adherence"),
    "primary_diab":        ("Diabetes as primary diagnosis", False, "Primary vs secondary diagnosis context"),
    "age_mid":             ("Patient age (midpoint)", False, "Age is a demographic characteristic"),
    "diag_1_cat":          ("Primary diagnosis category", False, "Diagnosis type"),
    "diag_2_cat":          ("Secondary diagnosis category", False, "Comorbidity type"),
    "diag_3_cat":          ("Tertiary diagnosis category", False, "Comorbidity type"),
    "gender":              ("Gender", False, "Demographic attribute"),
    "race":                ("Race", False, "Demographic attribute — see fairness report"),
    "discharge_disposition_id": ("Discharge disposition", True, "Discharge destination is a planning variable"),
    "admission_type_id":   ("Admission type", False, "Emergency vs elective context"),
    "admission_source_id": ("Admission source", False, "Referral source"),
    "max_glu_serum":       ("Max serum glucose result", True, "Glucose control indicator"),
    "A1Cresult":           ("HbA1c result", True, "Long-term glycaemic control"),
}


def _readable_label(feature_name: str) -> str:
    for key, (label, *_) in FACTOR_META.items():
        if key in feature_name:
            return label
    # Fallback: humanise the raw feature name
    name = re.sub(r"_", " ", feature_name)
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return name.title()


def _is_modifiable(feature_name: str) -> bool:
    for key, (_, modifiable, *_) in FACTOR_META.items():
        if key in feature_name:
            return modifiable
    return False


def explain_instance(
    model,
    X_instance: np.ndarray,
    feature_names: list[str],
    top_n: int = 8,
    use_shap: bool = True,
) -> dict:
    """
    Generate SHAP-based or importance-based attribution for one instance.

    Returns a dict with:
      - drivers: list of {feature, label, value, shap_value, direction, modifiable}
      - method: 'shap' | 'importance'
    """
    attributions = []

    if use_shap:
        try:
            import shap
            # Unwrap model for tree explainer
            tree_model = model
            if hasattr(tree_model, "base_model"):
                tree_model = tree_model.base_model
            if hasattr(tree_model, "named_steps"):
                tree_model = tree_model.named_steps.get("classifier", tree_model)
            if hasattr(tree_model, "estimator"):
                tree_model = getattr(tree_model.estimator, "estimator", tree_model.estimator)
            elif hasattr(tree_model, "calibrated_classifiers_") and len(tree_model.calibrated_classifiers_) > 0:
                tree_model = getattr(tree_model.calibrated_classifiers_[0], "estimator", tree_model)

            # Try TreeExplainer for tree-based models
            try:
                explainer = shap.TreeExplainer(tree_model)
                shap_values = explainer.shap_values(X_instance)
                if isinstance(shap_values, list):
                    vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]  # positive class
                elif hasattr(shap_values, "shape") and len(shap_values.shape) > 1 and shap_values.shape[0] == 1:
                    vals = shap_values[0]
                else:
                    vals = shap_values
                method = "shap"
            except Exception:
                # Fall back to KernelExplainer (slower)
                background = np.zeros((1, X_instance.shape[1]))
                explainer = shap.KernelExplainer(model.predict_proba, background)
                shap_values = explainer.shap_values(X_instance, nsamples=100)
                vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
                method = "shap_kernel"

            n_feats = min(len(feature_names), len(vals))
            pairs = sorted(
                zip(feature_names[:n_feats], vals[:n_feats], X_instance[0][:n_feats]),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            for feat, sv, raw_val in pairs[:top_n]:
                attributions.append({
                    "feature": feat,
                    "label": _readable_label(feat),
                    "raw_value": float(raw_val),
                    "shap_value": float(sv),
                    "direction": "increases" if sv > 0 else "decreases",
                    "modifiable": _is_modifiable(feat),
                    "association_note": "association, not causation",
                })
            return {"drivers": attributions, "method": method}
        except Exception as e:
            pass  # Fall through to importance-based

    # Fallback: use global feature importances
    try:
        importance_model = getattr(model, "base_model", model)
        if hasattr(importance_model, "named_steps"):
            importance_model = importance_model.named_steps.get("classifier", importance_model)
        if hasattr(importance_model, "feature_importances_"):
            importances = importance_model.feature_importances_
        elif hasattr(importance_model, "coef_"):
            importances = np.abs(importance_model.coef_).mean(axis=0)
        elif hasattr(model, "estimators_"):
            importances = model.estimators_[0].feature_importances_
        else:
            return {"drivers": [], "method": "unavailable"}

        n = min(len(feature_names), len(importances))
        pairs = sorted(
            zip(feature_names[:n], importances[:n]),
            key=lambda x: x[1],
            reverse=True,
        )
        for feat, imp in pairs[:top_n]:
            attributions.append({
                "feature": feat,
                "label": _readable_label(feat),
                "importance": float(imp),
                "modifiable": _is_modifiable(feat),
                "association_note": "association, not causation",
            })
        return {"drivers": attributions, "method": "feature_importance"}
    except Exception:
        return {"drivers": [], "method": "unavailable"}


def build_recommended_actions(
    risk_tier: str,
    drivers: list[dict],
    encounter_data: Optional[dict] = None,
) -> list[dict]:
    """
    Rule-based recommended follow-up actions.
    These are planning suggestions, NOT clinical prescriptions.
    """
    actions = []
    modifiable = {d["feature"]: d for d in drivers if d.get("modifiable")}

    # Always recommend
    actions.append({
        "key": "medication_reconciliation",
        "label": "Medication Reconciliation",
        "rationale": "Review all current medications for accuracy and adherence before discharge.",
        "priority": "high",
    })

    if risk_tier in ("elevated", "high"):
        actions.append({
            "key": "followup_appointment",
            "label": "Schedule Follow-up Appointment",
            "rationale": "Coordinate a PCP or specialist visit within 7 days of discharge.",
            "priority": "high",
        })
        actions.append({
            "key": "diabetes_education",
            "label": "Diabetes Self-Management Education",
            "rationale": "Reinforce glucose monitoring and medication adherence.",
            "priority": "medium",
        })

    if risk_tier == "high":
        actions.append({
            "key": "telehealth_followup",
            "label": "Telehealth Follow-up (48–72h)",
            "rationale": "Early virtual check-in reduces the chance of decompensation.",
            "priority": "high",
        })
        actions.append({
            "key": "care_transitions_referral",
            "label": "Care Transitions Referral",
            "rationale": "Enrol in a transitional-care programme for intensive post-discharge support.",
            "priority": "medium",
        })

    # Contextual based on drivers
    if any("a1c" in k or "glu" in k for k in modifiable):
        actions.append({
            "key": "glycemic_review",
            "label": "Glycaemic Management Review",
            "rationale": "HbA1c or glucose values suggest a review of the diabetes management plan.",
            "priority": "medium",
        })

    if any("med_change" in k for k in modifiable):
        actions.append({
            "key": "pharmacy_consult",
            "label": "Pharmacy Consultation",
            "rationale": "Medication changes were made during this visit; review with clinical pharmacy.",
            "priority": "medium",
        })

    if encounter_data and encounter_data.get("discharge_disposition_id") in (1, 2, 3):
        actions.append({
            "key": "home_health",
            "label": "Home Health Coordination",
            "rationale": "Patient discharged home; assess need for home-health or visiting-nurse services.",
            "priority": "low",
        })

    # Deduplicate by key
    seen = set()
    deduped = []
    for a in actions:
        if a["key"] not in seen:
            seen.add(a["key"])
            deduped.append(a)

    return deduped
