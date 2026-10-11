"""
models.py
=========
Train, tune (Optuna), calibrate, and ensemble models for readmission prediction.

Models
------
1. Logistic Regression (baseline)
2. Random Forest (baseline)
3. LightGBM (Optuna-tuned, 50+ trials, PR-AUC objective)
4. XGBoost  (Optuna-tuned, 50+ trials, PR-AUC objective)
5. CatBoost (Optuna-tuned, 50+ trials, PR-AUC objective)
6. Soft-voting ensemble of best boosters → isotonic calibration

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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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
import lightgbm as lgb
import joblib

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "backend" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ── risk tiers ───────────────────────────────────────────────────────────────
RISK_TIERS = {
    "low": (0.0, 0.25),
    "elevated": (0.25, 0.50),
    "high": (0.50, 1.01),
}
DEFAULT_THRESHOLD = 0.25


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
# OPTUNA-TUNED BOOSTERS
# ═══════════════════════════════════════════════════════════════════════════════

def _optuna_tune_lgbm(X_train, y_train, X_val, y_val, n_trials: int = 50):
    """Tune LightGBM with Optuna, optimizing PR-AUC on validation set."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "is_unbalance": True,
            "random_state": RANDOM_STATE,
            "verbose": -1,
            "n_jobs": -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, y_prob)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"[models] LightGBM best PR-AUC: {study.best_value:.4f} ({len(study.trials)} trials)")

    best_params = study.best_params
    best_params.update({"is_unbalance": True, "random_state": RANDOM_STATE, "verbose": -1, "n_jobs": -1})
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train)
    return model, best_params


def _optuna_tune_xgb(X_train, y_train, X_val, y_val, n_trials: int = 50):
    """Tune XGBoost with Optuna, optimizing PR-AUC on validation set."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "logloss",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, y_prob)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"[models] XGBoost best PR-AUC: {study.best_value:.4f} ({len(study.trials)} trials)")

    best_params = study.best_params
    best_params.update({"scale_pos_weight": scale_pos_weight, "eval_metric": "logloss",
                        "random_state": RANDOM_STATE, "n_jobs": -1})
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    return model, best_params


def _optuna_tune_catboost(X_train, y_train, X_val, y_val, n_trials: int = 50):
    """Tune CatBoost with Optuna, optimizing PR-AUC on validation set."""
    import optuna
    from catboost import CatBoostClassifier
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 600),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "auto_class_weights": "Balanced",
            "random_seed": RANDOM_STATE,
            "verbose": 0,
        }
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, y_prob)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"[models] CatBoost best PR-AUC: {study.best_value:.4f} ({len(study.trials)} trials)")

    best_params = study.best_params
    best_params.update({"auto_class_weights": "Balanced", "random_seed": RANDOM_STATE, "verbose": 0})
    model = CatBoostClassifier(**best_params)
    model.fit(X_train, y_train)
    return model, best_params


# ═══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE + CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

class SoftVotingEnsemble:
    """Custom soft-voting ensemble for pre-trained models with isotonic calibration."""

    def __init__(self, models: dict, weights: Optional[list] = None):
        self.models = models  # {name: fitted_model}
        n = len(models)
        self.weights = weights or [1.0 / n] * n
        self.calibrator = None

    def predict_proba(self, X) -> np.ndarray:
        """Weighted average of predicted probabilities."""
        probs = np.zeros((X.shape[0], 2))
        for (name, model), w in zip(self.models.items(), self.weights):
            p = model.predict_proba(X)
            probs += w * p
        if self.calibrator is not None:
            raw = probs[:, 1]
            calibrated = self.calibrator.predict(raw.reshape(-1, 1))
            return np.column_stack([1 - calibrated, calibrated])
        return probs

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def calibrate_isotonic(self, X_val, y_val):
        """Fit isotonic calibration on validation set probabilities."""
        from sklearn.isotonic import IsotonicRegression
        raw_probs = np.zeros(X_val.shape[0])
        for (name, model), w in zip(self.models.items(), self.weights):
            raw_probs += w * model.predict_proba(X_val)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(raw_probs, y_val)
        print(f"[models] Isotonic calibration fitted on {len(y_val)} validation samples")


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

def compare_models(
    X_train, y_train, X_val, y_val, X_test, y_test,
    n_optuna_trials: int = 50,
    threshold: Optional[float] = None,
) -> dict:
    """Train all models, tune with Optuna, build ensemble, calibrate, evaluate.

    Returns (results_dict, final_model, optimal_threshold, all_models)
    """
    timings = {}
    models = {}

    # ── Baselines ─────────────────────────────────────────────────────────
    print("\n[models] Training Logistic Regression (baseline)...")
    t0 = time.time()
    lr = train_logistic_regression(X_train, y_train)
    timings["LogisticRegression"] = round(time.time() - t0, 2)
    models["LogisticRegression"] = lr

    print("[models] Training Random Forest (baseline)...")
    t0 = time.time()
    rf = train_random_forest(X_train, y_train)
    timings["RandomForest"] = round(time.time() - t0, 2)
    models["RandomForest"] = rf

    # ── Optuna-tuned boosters ─────────────────────────────────────────────
    print(f"\n[models] Optuna-tuning LightGBM ({n_optuna_trials} trials)...")
    t0 = time.time()
    lgbm_model, lgbm_params = _optuna_tune_lgbm(X_train, y_train, X_val, y_val, n_optuna_trials)
    timings["LightGBM"] = round(time.time() - t0, 2)
    models["LightGBM"] = lgbm_model

    print(f"[models] Optuna-tuning XGBoost ({n_optuna_trials} trials)...")
    t0 = time.time()
    xgb_model, xgb_params = _optuna_tune_xgb(X_train, y_train, X_val, y_val, n_optuna_trials)
    timings["XGBoost"] = round(time.time() - t0, 2)
    models["XGBoost"] = xgb_model

    print(f"[models] Optuna-tuning CatBoost ({n_optuna_trials} trials)...")
    t0 = time.time()
    cb_model, cb_params = _optuna_tune_catboost(X_train, y_train, X_val, y_val, n_optuna_trials)
    timings["CatBoost"] = round(time.time() - t0, 2)
    models["CatBoost"] = cb_model

    # ── Soft-voting ensemble (top 3 boosters) ─────────────────────────────
    print("\n[models] Building soft-voting ensemble (LightGBM + XGBoost + CatBoost)...")
    ensemble = SoftVotingEnsemble(
        {"LightGBM": lgbm_model, "XGBoost": xgb_model, "CatBoost": cb_model},
        weights=[0.35, 0.35, 0.30],
    )

    # Isotonic calibration on validation set
    ensemble.calibrate_isotonic(X_val, y_val)
    models["CalibratedEnsemble"] = ensemble
    timings["CalibratedEnsemble"] = round(timings["LightGBM"] + timings["XGBoost"] + timings["CatBoost"], 2)

    # ── Threshold optimization on validation set ──────────────────────────
    print("\n[models] Optimizing decision threshold on validation set...")
    val_probs = {}
    for name, model in models.items():
        val_probs[name] = model.predict_proba(X_val)[:, 1]

    if threshold is None:
        threshold, rationale = optimize_threshold(y_val, val_probs["CalibratedEnsemble"], strategy="f1")
    else:
        rationale = f"User-specified threshold: {threshold}"
    print(f"[models] Selected threshold: {threshold:.3f} — {rationale}")

    # ── Final evaluation on TEST SET (touched once) ───────────────────────
    print("\n" + "=" * 60)
    print("   FINAL EVALUATION ON HELD-OUT TEST SET")
    print("=" * 60)

    all_results = []
    test_probs = {}
    curves = {}

    for name, model in models.items():
        t_prob = model.predict_proba(X_test)[:, 1]
        test_probs[name] = t_prob
        metrics = compute_metrics(y_test, t_prob, threshold)
        metrics["model"] = name
        metrics["training_time_s"] = timings[name]

        # Bootstrap CIs
        ci = bootstrap_ci(y_test.values if hasattr(y_test, 'values') else y_test,
                          t_prob, threshold, n_boot=1000)
        metrics["bootstrap_ci"] = ci

        # Curves for plotting
        fpr, tpr, _ = roc_curve(y_test, t_prob)
        pr_prec, pr_rec, _ = precision_recall_curve(y_test, t_prob)
        prob_true, prob_pred = calibration_curve(y_test, t_prob, n_bins=10, strategy="uniform")

        curves[name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "pr_precision": pr_prec.tolist(),
            "pr_recall": pr_rec.tolist(),
            "calib_true": prob_true.tolist(),
            "calib_pred": prob_pred.tolist(),
        }

        all_results.append(metrics)
        print(f"  {name:25s} | AUROC={metrics['auroc']:.4f} | PR-AUC={metrics['pr_auc']:.4f} "
              f"| P={metrics['precision']:.3f} | R={metrics['recall']:.3f} | F1={metrics['f1']:.3f} "
              f"| Brier={metrics['brier_score']:.4f}")

    # ── Save tuned hyperparameters ────────────────────────────────────────
    tuned_params = {
        "lightgbm": lgbm_params,
        "xgboost": xgb_params,
        "catboost": cb_params,
    }

    return {
        "results": all_results,
        "test_probs": test_probs,
        "curves": curves,
        "timings": timings,
        "tuned_params": tuned_params,
        "threshold": threshold,
        "threshold_rationale": rationale,
        "selected_model": "CalibratedEnsemble",
    }, ensemble, threshold, models


def probability_to_tier(prob: float) -> str:
    """Map probability to risk tier."""
    for tier, (lo, hi) in RISK_TIERS.items():
        if lo <= prob < hi:
            return tier
    return "high"


def get_feature_importance(model, feature_names: list[str], top_n: int = 15) -> list[dict]:
    """Extract feature importances from tree-based model."""
    try:
        m = model
        # Unwrap ensemble
        if isinstance(m, SoftVotingEnsemble):
            # Use first model's importances
            m = list(m.models.values())[0]
        if hasattr(m, "estimator"):
            m = getattr(m.estimator, "estimator", m.estimator)
        elif hasattr(m, "calibrated_classifiers_") and len(m.calibrated_classifiers_) > 0:
            m = getattr(m.calibrated_classifiers_[0], "estimator", m)

        if hasattr(m, "feature_importances_"):
            importances = m.feature_importances_
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
        "model_type": "Soft-Voting Ensemble (LightGBM+XGBoost+CatBoost, isotonic calibrated)",
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
