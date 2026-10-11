"""Validation-only policy-gradient experiment over SMOTE XGBoost and MLP scores.

This is a research candidate; it does not replace the deployed model automatically.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "backend" / "artifacts"
sys.path.insert(0, str(ROOT))
from src.preprocessing import load_raw_data, run_full_pipeline

RNG = np.random.default_rng(42)


def metrics(y, p, cutoff):
    pred = (p >= cutoff).astype(int)
    return {"threshold": float(cutoff), "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
            "brier_score": float(brier_score_loss(y, p)),
            "positive_predictions": int(pred.sum())}


def best_accuracy_threshold(y, p):
    fpr, tpr, thresholds = roc_curve(y, p)
    positives, negatives = int((y == 1).sum()), int((y == 0).sum())
    acc = ((1 - fpr) * negatives + tpr * positives) / len(y)
    best = np.flatnonzero(np.isclose(acc, acc.max()))
    return float(np.max(thresholds[best]))


def train_policy(oof, y, val_state, y_val, iterations=1700, hidden=64):
    mean, scale = oof.mean(axis=0), oof.std(axis=0)
    scale[scale == 0] = 1.0
    x, xv = (oof - mean) / scale, (val_state - mean) / scale
    w1 = RNG.normal(0, 0.08, (2, hidden)); b1 = np.zeros(hidden)
    w2 = RNG.normal(0, 0.08, (hidden, 1)); b2 = np.zeros(1)
    m = [np.zeros_like(a) for a in (w1, b1, w2, b2)]
    v = [np.zeros_like(a) for a in (w1, b1, w2, b2)]
    best, best_key, history, step = None, (-1.0, -1.0), [], 0

    def forward(z):
        h = np.tanh(z @ w1 + b1)
        logits = np.clip(h @ w2 + b2, -25, 25)
        return h, (1 / (1 + np.exp(-logits))).ravel()

    for episode in range(1, iterations + 1):
        n_each = 256
        pos = RNG.choice(np.flatnonzero(y == 1), n_each, replace=True)
        neg = RNG.choice(np.flatnonzero(y == 0), n_each, replace=True)
        ids = np.r_[pos, neg]; RNG.shuffle(ids)
        h, prob = forward(x[ids]); labels = y[ids]
        actions = (RNG.random(len(prob)) < prob).astype(int)
        reward = np.where(actions == labels, 1.0, -1.0)
        advantage = reward - reward.mean()
        grad = (advantage * (actions - prob) + .002 * (.5 - prob))[:, None]
        gw2, gb2 = h.T @ grad / len(ids), grad.mean(axis=0)
        gz = (grad @ w2.T) * (1 - h * h)
        gw1, gb1 = x[ids].T @ gz / len(ids), gz.mean(axis=0)
        step += 1
        for j, (param, g) in enumerate(zip((w1, b1, w2, b2), (gw1, gb1, gw2, gb2))):
            g = np.clip(g, -1, 1); m[j] = .9*m[j] + .1*g; v[j] = .999*v[j] + .001*g*g
            param += .003*(m[j]/(1-.9**step))/(np.sqrt(v[j]/(1-.999**step)) + 1e-8)
        if episode % 25 == 0 or episode == iterations:
            _, pv = forward(xv); cutoff = best_accuracy_threshold(y_val, pv)
            row = metrics(y_val, pv, cutoff); pred = pv >= cutoff
            tn = np.sum((y_val == 0) & ~pred); fp = np.sum((y_val == 0) & pred)
            tp = np.sum((y_val == 1) & pred); fn = np.sum((y_val == 1) & ~pred)
            row["balanced_accuracy"] = float(.5*(tn/max(tn+fp,1) + tp/max(tp+fn,1)))
            history.append({"episode": episode, **row})
            key = (row["accuracy"], row["balanced_accuracy"])
            if key > best_key:
                best_key = key
                best = (w1.copy(), b1.copy(), w2.copy(), b2.copy(), cutoff, episode)
    w1, b1, w2, b2, cutoff, selected_episode = best
    state = {"mean": mean, "scale": scale, "w1": w1, "b1": b1, "w2": w2, "b2": b2,
             "threshold": cutoff, "hidden_units": hidden, "episodes": iterations,
             "selected_episode": selected_episode, "validation_history": history}
    return state


def policy_predict(state, inputs):
    z = (inputs - state["mean"]) / state["scale"]
    h = np.tanh(z @ state["w1"] + state["b1"])
    logits = np.clip(h @ state["w2"] + state["b2"], -25, 25)
    return (1 / (1 + np.exp(-logits))).ravel()


def main():
    df = load_raw_data()
    if df is None:
        raise FileNotFoundError("Expected data/diabetic_data.csv")
    data = run_full_pipeline(df)
    Xtr, Xv, Xt = (data[k] for k in ("X_train", "X_val", "X_test"))
    ytr, yv, yt = (np.asarray(data[k]) for k in ("y_train", "y_val", "y_test"))
    groups = np.asarray(data["groups_train"])
    xgb_model = joblib.load(ARTIFACTS / "xgboost.joblib")
    xgb_template = xgb_model.base_model.named_steps["classifier"]
    smote = SMOTE(sampling_strategy=1.0, k_neighbors=5, random_state=42)
    mlp_template = MLPClassifier(hidden_layer_sizes=(64,), activation="relu", solver="adam",
        alpha=.01, batch_size=2048, learning_rate_init=.0005, max_iter=60,
        early_stopping=True, validation_fraction=.15, n_iter_no_change=8, random_state=42)
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    oof = np.zeros((len(ytr), 2))
    print("[policy] Building grouped OOF probabilities (SMOTE inside each training fold)...", flush=True)
    for fold, (tr, hold) in enumerate(cv.split(Xtr, ytr, groups), start=1):
        for col, template in enumerate((xgb_template, mlp_template)):
            pipe = Pipeline([("smote", clone(smote)), ("classifier", clone(template))])
            pipe.fit(Xtr[tr], ytr[tr])
            oof[hold, col] = pipe.predict_proba(Xtr[hold])[:, 1]
        print(f"[policy] fold {fold}/3 complete", flush=True)
    mlp = Pipeline([("smote", clone(smote)), ("classifier", clone(mlp_template))])
    mlp.fit(Xtr, ytr)
    val_state = np.column_stack([xgb_model.base_model.predict_proba(Xv)[:, 1], mlp.predict_proba(Xv)[:, 1]])
    test_state = np.column_stack([xgb_model.base_model.predict_proba(Xt)[:, 1], mlp.predict_proba(Xt)[:, 1]])
    policy = train_policy(oof, ytr, val_state, yv, iterations=1700, hidden=64)
    policy_val, policy_test = policy_predict(policy, val_state), policy_predict(policy, test_state)
    xgb_val, xgb_test = val_state[:, 0], test_state[:, 0]
    mlp_val, mlp_test = val_state[:, 1], test_state[:, 1]
    blend_weight = max(np.arange(0, 1.0001, .05),
                       key=lambda w: average_precision_score(yv, w*xgb_val+(1-w)*mlp_val))
    blend_val, blend_test = blend_weight*xgb_val+(1-blend_weight)*mlp_val, blend_weight*xgb_test+(1-blend_weight)*mlp_test
    cases = {"SMOTE XGBoost": (xgb_val, xgb_test), "SMOTE MLP (64 hidden)": (mlp_val, mlp_test),
             "Validation-weighted XGBoost+MLP": (blend_val, blend_test),
             "Policy-gradient XGBoost+MLP": (policy_val, policy_test)}
    results = {}
    for name, (pv, pt) in cases.items():
        cutoff = best_accuracy_threshold(yv, pv)
        results[name] = {"validation": metrics(yv, pv, cutoff), "test": metrics(yt, pt, cutoff)}
    artifact = {"architecture": {"input_dim": 2, "hidden_units": 64, "activation": "tanh",
                                  "policy_updates": 1700, "mlp_hidden_units": 64,
                                  "mlp_epochs": int(mlp.named_steps["classifier"].n_iter_)},
                "smote": {"strategy": 1.0, "scope": "training folds only"},
                "blend_xgb_weight": float(blend_weight), "selected_policy_episode": policy["selected_episode"],
                "results": results, "test_positive_rate": float(yt.mean()),
                "note": "Candidate is not automatically promoted; accuracy is paired with recall and PR-AUC."}
    (ARTIFACTS / "policy_experiment_metrics.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact, indent=2), flush=True)


if __name__ == "__main__":
    main()
