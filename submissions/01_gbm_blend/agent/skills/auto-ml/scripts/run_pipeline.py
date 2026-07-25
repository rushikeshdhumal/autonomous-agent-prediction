#!/usr/bin/env python3
"""auto_ml skill — schema-adaptive GBM-blend pipeline for tabular binary classification.

Reads train.csv / test.csv / sample_submission.csv from the working directory, trains a
cross-validated LightGBM + XGBoost + CatBoost ensemble (greedy-weighted on out-of-fold
predictions), writes positive-class probabilities to an output CSV matching
sample_submission.csv, and prints a compact JSON summary to stdout.

Self-contained and dependency-light: uses only libraries pre-installed in
gcr.io/kaggle-images/python (pandas, numpy, scikit-learn, lightgbm, xgboost, catboost).
No internet / pip needed. There is NO DATA.md in the eval sandbox, so column types are
inferred from the data itself.

Usage:
    python3 run_pipeline.py [--data-dir .] [--out submission.csv] [--models lightgbm,xgboost,catboost] [--splits 5]
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
SEED = 42
_ORD_RE = r"^\s*ord_?\d+\s*$"  # string ordinal like 'ord_3'


# --------------------------------------------------------------------------- encoding
def analyze_columns(train: pd.DataFrame, feat_cols: list) -> dict:
    """Classify each feature from data alone: 'num' (numeric, ordered, NaN kept),
    'ord' (string 'ord_<int>' -> ordinal-encode by the integer), 'cat' (other string
    -> nominal one-hot)."""
    kinds = {}
    for c in feat_cols:
        s = train[c]
        if s.dtype.kind in "iufb":
            kinds[c] = "num"
            continue
        nn = s.dropna().astype("string")
        kinds[c] = "ord" if (len(nn) and nn.str.match(_ORD_RE, case=False).all()) else "cat"
    return kinds


def build_matrices(train: pd.DataFrame, test: pd.DataFrame, feat_cols: list):
    kinds = analyze_columns(train, feat_cols)
    ord_cols = [c for c in feat_cols if kinds[c] == "ord"]
    cat_cols = [c for c in feat_cols if kinds[c] == "cat"]
    num_cols = [c for c in feat_cols if kinds[c] == "num"]
    n_tr = len(train)
    combined = pd.concat([train[feat_cols], test[feat_cols]], axis=0, ignore_index=True)
    parts = []
    if ord_cols:
        om = pd.DataFrame(index=combined.index)
        for c in ord_cols:
            codes = combined[c].astype("string").str.extract(r"(\d+)", expand=False)
            om[c] = pd.to_numeric(codes, errors="coerce")
        parts.append(om.astype(np.float32))
    if cat_cols:
        parts.append(pd.get_dummies(combined[cat_cols].astype("object"),
                                    columns=cat_cols, dummy_na=True, dtype=np.float32))
    if num_cols:
        parts.append(combined[num_cols].astype(np.float32))
    X = pd.concat(parts, axis=1) if parts else combined.astype(np.float32)
    return X.iloc[:n_tr].reset_index(drop=True), X.iloc[n_tr:].reset_index(drop=True), kinds


# ----------------------------------------------------------------------------- models
# Each booster trains up to ES_MAX_ITERS trees with a low learning rate and stops early
# on the validation fold's AUC (early stopping adapts the tree count per dataset — it was
# the single biggest offline lever, lifting the blend from ~0.798 to ~0.803 mean AUC).
ES_MAX_ITERS = 3000
ES_ROUNDS = 150


def _fit_es(name, X_tr, y_tr, X_va, y_va, X_test):
    """Fit one booster with early stopping on (X_va, y_va); return (val_proba, test_proba)."""
    if name == "lightgbm":
        from lightgbm import LGBMClassifier, early_stopping
        m = LGBMClassifier(n_estimators=ES_MAX_ITERS, learning_rate=0.02, num_leaves=31,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                           random_state=SEED, n_jobs=-1, verbosity=-1)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
              callbacks=[early_stopping(ES_ROUNDS, verbose=False)])
    elif name == "xgboost":
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=ES_MAX_ITERS, learning_rate=0.02, max_depth=5, subsample=0.8,
                          colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", eval_metric="auc",
                          early_stopping_rounds=ES_ROUNDS, random_state=SEED, n_jobs=-1, verbosity=0)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    elif name == "catboost":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(iterations=ES_MAX_ITERS, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
                               eval_metric="AUC", random_seed=SEED, thread_count=-1, verbose=False,
                               allow_writing_files=False, early_stopping_rounds=ES_ROUNDS)
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    else:
        raise ValueError(name)
    return m.predict_proba(X_va)[:, 1], m.predict_proba(X_test)[:, 1]


def cv_oof_test(X, y, X_test, name, n_splits):
    Xv, Xt = X.values, X_test.values
    oof = np.zeros(len(y))
    test = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr, va in skf.split(Xv, y):
        va_p, test_p = _fit_es(name, Xv[tr], y[tr], Xv[va], y[va], Xt)
        oof[va] = va_p
        test += test_p / n_splits
    return oof, test, roc_auc_score(y, oof)


def greedy_weights(oofs: np.ndarray, y: np.ndarray, n_iter: int = 30) -> np.ndarray:
    """Caruana (2004) forward selection with replacement on OOF predictions."""
    M = len(oofs)
    singles = [roc_auc_score(y, oofs[i]) for i in range(M)]
    picks = [int(np.argmax(singles))]
    best = max(singles)
    for _ in range(n_iter):
        cur = oofs[picks].mean(axis=0)
        scores = [roc_auc_score(y, (cur * len(picks) + oofs[i]) / (len(picks) + 1)) for i in range(M)]
        j = int(np.argmax(scores))
        if scores[j] <= best + 1e-6:
            break
        picks.append(j)
        best = scores[j]
    return np.bincount(picks, minlength=M) / len(picks)


# ------------------------------------------------------------------------------- main
_REQUIRED = ("train.csv", "test.csv", "sample_submission.csv")


def find_data_dir(explicit: str | None) -> str:
    """Locate the directory holding the 3 competition CSVs.

    Skills run via run_skill_script execute in an ephemeral temp cwd, NOT the sandbox
    work dir where the data lives, so we must search rather than assume cwd. Candidates:
    explicit arg, cwd, /work (the harness default work dir), then a few parents of cwd.
    """
    cands = []
    if explicit:
        cands.append(explicit)
    cands.append(os.getcwd())
    cands.append("/work")
    p = os.getcwd()
    for _ in range(4):
        p = os.path.dirname(p)
        if p:
            cands.append(p)
    for c in cands:
        if c and all(os.path.exists(os.path.join(c, f)) for f in _REQUIRED):
            return c
    raise SystemExit(f"Could not locate {_REQUIRED} in any of: {cands}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--models", default="lightgbm,xgboost,catboost")
    ap.add_argument("--splits", type=int, default=5)
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(",") if m.strip()]

    data_dir = find_data_dir(a.data_dir)
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    sample = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
    id_col, target = sample.columns[0], sample.columns[1]
    feat_cols = [c for c in train.columns if c not in (id_col, target)]

    y_raw = train[target]
    classes = np.sort(y_raw.unique())
    if len(classes) != 2:
        raise SystemExit(f"Expected binary target, got classes={classes.tolist()}")
    y = (y_raw.to_numpy() == classes[1]).astype(int)  # positive class = larger label

    X_tr, X_te, kinds = build_matrices(train, test, feat_cols)

    oofs, tests, aucs = [], [], {}
    for m in models:
        o, t, auc = cv_oof_test(X_tr, y, X_te, m, a.splits)
        oofs.append(o)
        tests.append(t)
        aucs[m] = round(float(auc), 5)
    oofs = np.array(oofs)
    tests = np.array(tests)

    if len(models) > 1:
        w = greedy_weights(oofs, y)
    else:
        w = np.array([1.0])
    blend_oof = (oofs * w[:, None]).sum(axis=0)
    blend_test = (tests * w[:, None]).sum(axis=0)
    blend_auc = round(float(roc_auc_score(y, blend_oof)), 5)

    sub = sample.copy()
    sub[target] = blend_test
    out_path = a.out if os.path.isabs(a.out) else os.path.join(data_dir, a.out)
    sub.to_csv(out_path, index=False)

    n_cat = sum(1 for c in feat_cols if kinds[c] == "cat")
    n_ord = sum(1 for c in feat_cols if kinds[c] == "ord")
    n_num = sum(1 for c in feat_cols if kinds[c] == "num")
    summary = {
        "status": "ok",
        "output": out_path,
        "rows": len(sub),
        "n_features": len(feat_cols),
        "schema": {"num": n_num, "ord": n_ord, "cat": n_cat},
        "cv_oof_auc_per_model": aucs,
        "blend_weights": {models[i]: round(float(w[i]), 3) for i in range(len(models))},
        "blend_oof_auc": blend_auc,
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
