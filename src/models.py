"""
models.py
=========
Train, tune, calibrate, and compare Logistic Regression, Random Forest, and XGBoost.

Models
------
1. Logistic Regression (baseline)
2. Random Forest (baseline)
3. XGBoost (grouped-CV tuned with SMOTE confined to training folds)

Threshold
---------
Chosen on validation set to maximize F1 (or hit target recall), not default 0.5.

Evaluation
----------
Bootstrap 95% CIs for all metrics on the held-out test set.
LACE score baseline comparison included.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import xgboost as xgb
import joblib

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "backend" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ── risk tiers ───────────────────────────────────────────────────────────────
RISK_TIERS = {
    "low": (0.0, 0.12),
    "elevated": (0.12, 0.20),
    "high": (0.20, 1.01),
}
DEFAULT_THRESHOLD = 0.15


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_prob, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Compute comprehensive classification metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def bootstrap_ci(y_true, y_prob, threshold: float, n_boot: int = 2000, ci: float = 0.95) -> dict:
    """Compute bootstrap 95% confidence intervals for key metrics."""
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(y_true)
    metric_names = ["auroc", "pr_auc", "precision", "recall", "f1", "balanced_accuracy", "brier_score"]
    boot_results = {m: [] for m in metric_names}

    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yt = y_true[idx]
        yp = y_prob[idx]
        # Skip if only one class in bootstrap sample
        if len(np.unique(yt)) < 2:
            continue
        try:
            metrics = compute_metrics(yt, yp, threshold)
            for m in metric_names:
                boot_results[m].append(metrics[m])
        except Exception:
            continue

    alpha = (1 - ci) / 2
    ci_dict = {}
    for m in metric_names:
        arr = np.array(boot_results[m])
        if len(arr) > 0:
            ci_dict[m] = {
                "mean": float(np.mean(arr)),
                "ci_lower": float(np.percentile(arr, alpha * 100)),
                "ci_upper": float(np.percentile(arr, (1 - alpha) * 100)),
            }
    return ci_dict


# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLD OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def optimize_threshold(y_true, y_prob, strategy: str = "f1", min_recall: float = 0.40) -> tuple[float, str]:
    """Find optimal decision threshold on validation set.

    Strategies:
    - 'f1': maximize F1
    - 'recall_target': maximize precision s.t. recall >= min_recall
    """
    thresholds = np.linspace(0.05, 0.70, 131)
    best_t = DEFAULT_THRESHOLD
    best_score = -1.0
    rationale = ""

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        rec = recall_score(y_true, preds, zero_division=0)
        prec = precision_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        if strategy == "f1":
            if f1 > best_score:
                best_score = f1
                best_t = float(t)
                rationale = f"Max F1={f1:.3f} at t={t:.3f} (P={prec:.3f}, R={rec:.3f})"
        elif strategy == "recall_target":
            if rec >= min_recall and prec > best_score:
                best_score = prec
                best_t = float(t)
                rationale = f"Max Precision={prec:.3f} at t={t:.3f} with Recall={rec:.3f}>={min_recall}"

    return best_t, rationale


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

def train_logistic_regression(X_train, y_train):
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE, C=0.1)
    lr.fit(X_train, y_train)
    return lr


def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=10,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


# ═══════════════════════════════════════════════════════════════════════════════
# LACE SCORE BASELINE
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_lace_baseline(lace_scores: np.ndarray, y_true: np.ndarray) -> dict:
    """Evaluate LACE score as a standalone baseline predictor.
    Uses the score directly as a probability proxy (scaled to 0-1).
    """
    # Scale LACE to [0, 1] range
    lace_min, lace_max = lace_scores.min(), lace_scores.max()
    if lace_max > lace_min:
        lace_prob = (lace_scores - lace_min) / (lace_max - lace_min)
    else:
        lace_prob = np.full_like(lace_scores, 0.5, dtype=float)

    # Find best threshold for LACE
    best_f1 = 0
    best_t = 0.5
    for t in np.linspace(0.1, 0.9, 81):
        preds = (lace_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    y_pred = (lace_prob >= best_t).astype(int)
    try:
        auroc = float(roc_auc_score(y_true, lace_prob))
    except Exception:
        auroc = 0.5
    try:
        prauc = float(average_precision_score(y_true, lace_prob))
    except Exception:
        prauc = 0.0

    return {
        "model": "LACE Score (baseline)",
        "threshold": float(best_t),
        "auroc": auroc,
        "pr_auc": prauc,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, lace_prob)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING + COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

class PlattCalibratedModel:
    """Apply a validation-fitted sigmoid to a model's positive-class score."""
    def __init__(self, base_model, name: str):
        self.base_model = base_model
        self.name = name
        self.calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=RANDOM_STATE)

    def fit_calibration(self, X_val, y_val):
        raw = self.base_model.predict_proba(X_val)[:, 1].reshape(-1, 1)
        self.calibrator.fit(raw, y_val)
        return self

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1].reshape(-1, 1)
        return self.calibrator.predict_proba(raw)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    @property
    def feature_importances_(self):
        estimator = self.base_model.named_steps.get("classifier") if hasattr(self.base_model, "named_steps") else self.base_model
        return getattr(estimator, "feature_importances_", None)


def compare_models(
    X_train, y_train, X_val, y_val, X_test, y_test,
    n_optuna_trials: int = 50,
    threshold: Optional[float] = None,
    groups_train=None,
) -> dict:
    """Tune three classifiers with patient-grouped CV and fold-local SMOTE."""
    timings = {}
    templates = {
        "LogisticRegression": LogisticRegression(max_iter=1500, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        "XGBoost": xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", random_state=RANDOM_STATE,
            n_jobs=1, tree_method="hist", scale_pos_weight=1.0,
        ),
    }
    grids = {
        "LogisticRegression": {"classifier__C": [0.1, 0.5, 1.0]},
        "RandomForest": {
            "classifier__n_estimators": [120, 200],
            "classifier__max_depth": [8, 12],
            "classifier__min_samples_leaf": [5],
        },
        "XGBoost": {
            "classifier__n_estimators": [140, 220],
            "classifier__max_depth": [3, 5],
            "classifier__learning_rate": [0.03, 0.05],
        },
    }
    if groups_train is None:
        groups_train = np.arange(len(y_train))
    groups_train = np.asarray(groups_train)
    if len(groups_train) != len(y_train):
        raise ValueError("groups_train must have one patient/group ID per training row")
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    models = {}
    val_probs = {}
    tuned_params = {}
    cv_scores = {}

    for name, template in templates.items():
        print(f"\n[models] Tuning {name} with 3-fold grouped CV and fold-local SMOTE...")
        started = time.time()
        estimator = ImbalancedPipeline([
            ("smote", SMOTE(sampling_strategy=1.0, k_neighbors=5, random_state=RANDOM_STATE)),
            ("classifier", clone(template)),
        ])
        search = GridSearchCV(
            estimator, grids[name], scoring="roc_auc", cv=cv, n_jobs=1,
            refit=True, error_score="raise",
        )
        search.fit(X_train, y_train, groups=groups_train)
        calibrated = PlattCalibratedModel(search.best_estimator_, name).fit_calibration(X_val, y_val)
        models[name] = calibrated
        val_probs[name] = calibrated.predict_proba(X_val)[:, 1]
        tuned_params[name] = search.best_params_
        cv_scores[name] = float(search.best_score_)
        timings[name] = round(time.time() - started, 2)
        print(f"[models] {name}: CV ROC-AUC={cv_scores[name]:.4f}; params={tuned_params[name]}")

    validation_scores = {
        name: {
            "roc_auc": float(roc_auc_score(y_val, probs)),
            "pr_auc": float(average_precision_score(y_val, probs)),
            "brier_score": float(brier_score_loss(y_val, probs)),
        }
        for name, probs in val_probs.items()
    }
    selected_name = sorted(validation_scores, key=lambda name: (
        -validation_scores[name]["roc_auc"], validation_scores[name]["brier_score"]
    ))[0]
    if threshold is None:
        threshold, rationale = optimize_threshold(y_val, val_probs[selected_name], strategy="recall_target")
        if not rationale:
            threshold, rationale = optimize_threshold(y_val, val_probs[selected_name], strategy="f1")
    else:
        rationale = f"User-specified threshold: {threshold}"
    print(f"[models] Selected {selected_name} by validation ROC-AUC/Brier; threshold={threshold:.3f} ({rationale})")

    all_results = []
    test_probs = {}
    curves = {}
    for name, model in models.items():
        t_prob = model.predict_proba(X_test)[:, 1]
        test_probs[name] = t_prob
        metrics = compute_metrics(y_test, t_prob, threshold)
        metrics.update({"model": name, "training_time_s": timings[name], "cv_roc_auc": cv_scores[name]})
        metrics["bootstrap_ci"] = bootstrap_ci(
            y_test.values if hasattr(y_test, "values") else y_test,
            t_prob, threshold, n_boot=1000,
        )
        fpr, tpr, _ = roc_curve(y_test, t_prob)
        pr_prec, pr_rec, _ = precision_recall_curve(y_test, t_prob)
        prob_true, prob_pred = calibration_curve(y_test, t_prob, n_bins=10, strategy="uniform")
        curves[name] = {
            "fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "pr_precision": pr_prec.tolist(), "pr_recall": pr_rec.tolist(),
            "calib_true": prob_true.tolist(), "calib_pred": prob_pred.tolist(),
        }
        all_results.append(metrics)
        print(f"  {name:20s} | AUROC={metrics['auroc']:.4f} | PR-AUC={metrics['pr_auc']:.4f} "
              f"| Acc={metrics['accuracy']:.3f} | P={metrics['precision']:.3f} | R={metrics['recall']:.3f} | F1={metrics['f1']:.3f}")

    comparison = {
        "results": all_results, "test_probs": test_probs, "curves": curves,
        "timings": timings, "tuned_params": tuned_params, "cv_scores": cv_scores,
        "validation_scores": validation_scores, "threshold": float(threshold),
        "threshold_rationale": rationale, "selected_model": selected_name,
        "smote": {"sampling_strategy": 1.0, "k_neighbors": 5, "scope": "training fold only"},
    }
    return comparison, models[selected_name], threshold, models


def probability_to_tier(prob: float) -> str:
    """Map probability to risk tier."""
    for tier, (lo, hi) in RISK_TIERS.items():
        if lo <= prob < hi:
            return tier
    return "high"


def get_feature_importance(model, feature_names: list[str], top_n: int = 15) -> list[dict]:
    """Extract tree importances or absolute logistic-regression coefficients."""
    try:
        m = model
        if hasattr(m, "base_model"):
            m = m.base_model
        if hasattr(m, "named_steps"):
            m = m.named_steps.get("classifier", m)
        if hasattr(m, "estimator"):
            m = getattr(m.estimator, "estimator", m.estimator)
        elif hasattr(m, "calibrated_classifiers_") and len(m.calibrated_classifiers_) > 0:
            m = getattr(m.calibrated_classifiers_[0], "estimator", m)

        if hasattr(m, "feature_importances_"):
            importances = m.feature_importances_
        elif hasattr(m, "coef_"):
            importances = np.abs(m.coef_).mean(axis=0)
        elif hasattr(m, "estimators_") and hasattr(m.estimators_[0], "feature_importances_"):
            importances = m.estimators_[0].feature_importances_
        else:
            return []

        n = min(len(importances), len(feature_names))
        importances = importances[:n]
        names = feature_names[:n]
        pairs = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)
        return [{"feature": n, "importance": float(v)} for n, v in pairs[:top_n]]
    except Exception:
        return []


def save_artifacts(
    final_model, preprocessor, caps: dict, comparison_results: dict,
    feature_names: list[str], threshold: float, all_models: dict = None,
):
    """Persist all inference artifacts to disk."""
    joblib.dump(final_model, ARTIFACTS_DIR / "model.joblib")
    joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.joblib")

    if all_models:
        for name, model in all_models.items():
            fname = name.lower().replace(" ", "_") + ".joblib"
            joblib.dump(model, ARTIFACTS_DIR / fname)

    artifact_meta = {
        "model_version": "2.0.0",
        "model_type": f"{comparison_results.get('selected_model', getattr(final_model, 'name', 'selected classifier'))} (Platt calibrated)",
        "threshold": threshold,
        "risk_tiers": RISK_TIERS,
        "feature_names": feature_names,
        "caps": caps,
    }
    with open(ARTIFACTS_DIR / "artifact_meta.json", "w") as f:
        json.dump(artifact_meta, f, indent=2)

    with open(ARTIFACTS_DIR / "model_metrics.json", "w") as f:
        json.dump(comparison_results, f, indent=2, default=str)

    print(f"[models] Artifacts saved to {ARTIFACTS_DIR}")
