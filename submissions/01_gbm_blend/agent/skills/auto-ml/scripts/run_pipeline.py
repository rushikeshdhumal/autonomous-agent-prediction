#!/usr/bin/env python3
"""auto_ml skill — anytime member-pool GBM+MLP blend pipeline for tabular binary classification.

Reads train.csv / test.csv / sample_submission.csv from the working directory. Builds a POOL
of LightGBM/XGBoost/CatBoost models (validated base configs, offline-tuned diverse configs,
bounded eval-time hyperparameter search, multi-seed reruns of the best member) plus a small
PyTorch MLP (a structurally different model family, validated to add real, largely
independent gains — see the MLP section below), greedy (Caruana) blends the pool on
out-of-fold predictions, and writes positive-class probabilities to an output CSV matching
sample_submission.csv.

ANYTIME DESIGN: a valid submission.csv exists after the very first (near-instant) model fit,
and is rewritten after every subsequent pool member — so the script is safe to run under any
real per-command time limit (which is not known in advance) without ever failing to produce a
submission. Work is added in priority order (highest expected value first) and gated by a
self-calibrating time budget: initial cost estimates are heuristic, but after the first fit of
each (model, n_splits) shape the ACTUAL measured time replaces the estimate for the rest of the
run, so later decisions reflect the real speed of the machine running the script.

Self-contained and dependency-light: uses only libraries pre-installed in
gcr.io/kaggle-images/python (pandas, numpy, scikit-learn, lightgbm, xgboost, catboost, torch).
No internet / pip needed. There is NO DATA.md in the eval sandbox, so column types are
inferred from the data itself.

Progress is logged to stderr (free — the harness only returns stdout to the LLM on success);
exactly one compact JSON summary is printed to stdout at the end.

Usage:
    python3 run_pipeline.py [--data-dir .] [--out submission.csv] [--time-budget 200]
                            [--models lightgbm,xgboost,catboost,mlp] [--splits N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
SEED = 42
_ORD_RE = r"^\s*ord_?\d+\s*$"  # string ordinal like 'ord_3'


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


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
# Base hyperparameters validated offline (Phase 1b): low learning rate + early stopping,
# lifting the blend from ~0.798 to ~0.803 mean AUC vs fixed-iteration defaults.
BASE_KWARGS = {
    "lightgbm": dict(learning_rate=0.02, num_leaves=31, subsample=0.8, subsample_freq=1,
                     colsample_bytree=0.8, reg_lambda=1.0, min_child_samples=20),
    "xgboost": dict(learning_rate=0.02, max_depth=5, subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=1.0, min_child_weight=1),
    "catboost": dict(learning_rate=0.03, depth=6, l2_leaf_reg=3.0),
}

# Offline-tuned diverse configs (dev/hpo.py random search + diversity selection, validated
# across all 16 folds: adds +0.0008 mean full AUC over the 3 base members alone, 0.8025 ->
# 0.8033 / private 0.8037 -> 0.8045, per `dev/hpo.py blend`, 2026-07-25).
HPO_SHORTLIST: dict[str, list[dict]] = {
    "lightgbm": [
        {"learning_rate": 0.0405, "num_leaves": 110, "min_child_samples": 26,
         "subsample": 0.715, "colsample_bytree": 0.873, "reg_lambda": 0.1903},
        {"learning_rate": 0.0244, "num_leaves": 25, "min_child_samples": 35,
         "subsample": 0.722, "colsample_bytree": 0.612, "reg_lambda": 0.7472},
    ],
    "xgboost": [
        {"learning_rate": 0.0369, "max_depth": 7, "min_child_weight": 8,
         "subsample": 0.988, "colsample_bytree": 0.957, "reg_lambda": 3.6039},
        {"learning_rate": 0.0308, "max_depth": 7, "min_child_weight": 10,
         "subsample": 0.826, "colsample_bytree": 0.906, "reg_lambda": 1.8597},
    ],
    "catboost": [
        {"learning_rate": 0.0484, "depth": 6, "l2_leaf_reg": 8.231},
        {"learning_rate": 0.0169, "depth": 5, "l2_leaf_reg": 5.718},
    ],
}

FULL_MAX_ITERS = 3000
FULL_ES_ROUNDS = 150

# Eval-time HPO (bounded random search on the actual dataset, screened cheaply then promoted).
HPO_TRIAL_BUDGET = 6           # total random trials to try across all models combined
HPO_SCREEN_SPLITS = 2
HPO_SCREEN_MAX_ITERS = 600
HPO_SCREEN_ES_ROUNDS = 50
HPO_PROMOTE_MARGIN = 0.01      # promote a trial if its screening AUC is within this of the pool's best

# Multi-seed reruns of the current best single member (variance reduction via bagging).
MAX_RESEEDS = 2

# Time estimation defaults (seconds) before any real measurement exists this run; scale with
# dataset size and split count. Self-calibrates after the first observed fit of each shape.
_MIN_COST = 0.4


def _default_cost_estimate(model: str, n_train: int, n_splits: int, max_iters: int) -> float:
    """Heuristic cost estimate before any real observation exists this run. GBM cost scales
    with max_iters (boosting rounds); MLP cost scales with epochs but at a completely
    different (much higher) per-row rate, and MLP_MAX_EPOCHS has different semantics than
    a GBM's max_iters -- reusing the GBM formula for "mlp" badly UNDERESTIMATES its cost
    (by ~100x, empirically), risking a large time-budget overshoot on the fold where the
    first, unobserved MLP fit happens to be slow. Deliberately conservative (errs toward
    overestimating, which only risks skipping an opportunity, never a budget overshoot)."""
    if model == "mlp":
        per_split = max(3.0, n_train * 0.002)
    else:
        per_split = max(_MIN_COST, n_train * 3e-5 * (max_iters / 1000.0))
    return per_split * n_splits


class TimeBudget:
    """Self-calibrating time budget: an initial heuristic estimate per (model, n_splits,
    max_iters) is replaced by the actual observed time after the first such fit, so later
    decisions reflect the real speed of whatever machine is running the script.

    Cost is keyed PER MODEL (not shared across models): boosters can converge at very
    different speeds on the same dataset (e.g. on a high-signal fold, early stopping's
    patience may rarely trigger for one model, so it runs close to max_iters, while another
    model converges much faster) — sharing one estimate across models would let one slow
    model's observed cost wrongly cause another, faster model to be skipped.
    """

    def __init__(self, total_seconds: float, n_train: int):
        self.start = time.time()
        self.total = total_seconds
        self.n_train = n_train
        self._observed: dict[tuple, float] = {}

    def left(self) -> float:
        return self.total - (time.time() - self.start)

    def elapsed(self) -> float:
        return time.time() - self.start

    def estimate(self, model: str, n_splits: int, max_iters: int) -> float:
        key = (model, n_splits, max_iters)
        if key in self._observed:
            return self._observed[key]
        return _default_cost_estimate(model, self.n_train, n_splits, max_iters)

    def can_afford(self, model: str, n_splits: int, max_iters: int, safety: float = 1.3) -> bool:
        return self.left() > self.estimate(model, n_splits, max_iters) * safety

    def record(self, model: str, n_splits: int, max_iters: int, seconds: float) -> None:
        key = (model, n_splits, max_iters)
        # keep the max observed for this shape, a bit more conservative than the first sample
        self._observed[key] = max(self._observed.get(key, 0.0), seconds)


def fit_one(model: str, params: dict, X_tr, y_tr, X_va, y_va, X_test,
           max_iters: int, es_rounds: int):
    """Fit one booster with early stopping on (X_va, y_va); return (val_proba, test_proba)."""
    kwargs = dict(BASE_KWARGS[model])
    kwargs.update(params)
    if model == "lightgbm":
        from lightgbm import LGBMClassifier, early_stopping
        m = LGBMClassifier(n_estimators=max_iters, random_state=SEED, n_jobs=-1,
                           verbosity=-1, **kwargs)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
              callbacks=[early_stopping(es_rounds, verbose=False)])
    elif model == "xgboost":
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=max_iters, tree_method="hist", eval_metric="auc",
                          early_stopping_rounds=es_rounds, random_state=SEED, n_jobs=-1,
                          verbosity=0, **kwargs)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    elif model == "catboost":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(iterations=max_iters, eval_metric="AUC", random_seed=SEED,
                               thread_count=-1, verbose=False, allow_writing_files=False,
                               early_stopping_rounds=es_rounds, **kwargs)
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    else:
        raise ValueError(model)
    return m.predict_proba(X_va)[:, 1], m.predict_proba(X_test)[:, 1]


def cv_run(model: str, params: dict, X, y, X_test, n_splits: int,
          max_iters: int, es_rounds: int, seed: int = SEED):
    """Cross-validated OOF + bagged test predictions for one (model, params) config."""
    Xv, Xt = X, X_test
    oof = np.zeros(len(y))
    test = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(Xv, y):
        va_p, test_p = fit_one(model, params, Xv[tr], y[tr], Xv[va], y[va], Xt,
                               max_iters, es_rounds)
        oof[va] = va_p
        test += test_p / n_splits
    return oof, test, roc_auc_score(y, oof)


# ------------------------------------------------------------------------------- MLP member
# A small PyTorch MLP as a structurally different blend member (Phase 1b bigger-lever
# exploration): validated offline across all 16 folds to add +0.0026 mean full AUC on top
# of the full 9-member GBM pool (0.8033 -> 0.8059, dev/blend_with_nn.py, 2026-07-25) --
# roughly 4x the gain from GBM hyperparameter tuning/bagging alone. Its errors are
# genuinely decorrelated from the GBMs (greedy assigns it real weight on ~10/16 folds and
# safely excludes it, weight 0, on folds where it doesn't help -- e.g. small train_13).
MLP_MAX_EPOCHS = 150
MLP_PATIENCE = 15


def preprocess_for_nn(train_raw, test_raw, feat_cols, kinds):
    """NaN-free, standardized design matrix for the MLP (kept separate from the GBMs'
    NaN-preserving one-hot/ordinal matrix -- necessary preprocessing for the MLP to
    function, not new engineered signal: explicit engineered aggregate features were
    tested separately and found to actively hurt the GBMs, see dev/feature_eng.py)."""
    num_ord_cols = [c for c in feat_cols if kinds[c] in ("num", "ord")]
    cat_cols = [c for c in feat_cols if kinds[c] == "cat"]
    n_tr = len(train_raw)
    combined = pd.concat([train_raw[feat_cols], test_raw[feat_cols]], axis=0, ignore_index=True)

    parts = []
    if num_ord_cols:
        vals = pd.DataFrame(index=combined.index)
        miss = pd.DataFrame(index=combined.index)
        for c in num_ord_cols:
            if kinds[c] == "ord":
                v = pd.to_numeric(combined[c].astype("string").str.extract(r"(\d+)", expand=False),
                                  errors="coerce")
            else:
                v = combined[c].astype(np.float64)
            train_v = v.iloc[:n_tr]
            median = train_v.median()
            std = train_v.std()
            std = std if std and std > 1e-6 else 1.0
            mean = train_v.mean()
            miss[f"{c}_isna"] = v.isna().astype(np.float32)
            vals[c] = ((v.fillna(median) - mean) / std).astype(np.float32)
        parts.append(vals)
        parts.append(miss)
    if cat_cols:
        parts.append(pd.get_dummies(combined[cat_cols].astype("object"),
                                    columns=cat_cols, dummy_na=True, dtype=np.float32))
    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=combined.index)
    return X.iloc[:n_tr].reset_index(drop=True).values.astype(np.float32), \
        X.iloc[n_tr:].reset_index(drop=True).values.astype(np.float32)


def _make_mlp(input_dim: int):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 1),
    )


def _fit_nn_fold(X_tr, y_tr, X_va, y_va, X_test, max_epochs=MLP_MAX_EPOCHS,
                 patience=MLP_PATIENCE, seed=SEED):
    import torch
    torch.manual_seed(seed)
    import torch.nn as nn

    Xtr_t = torch.from_numpy(X_tr)
    ytr_t = torch.from_numpy(y_tr.astype(np.float32))
    Xva_t = torch.from_numpy(X_va)
    Xte_t = torch.from_numpy(X_test)

    model = _make_mlp(X_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    n = len(X_tr)
    batch_size = min(256, max(32, n // 8))
    best_auc = -1.0
    best_va_pred, best_te_pred = None, None
    no_improve = 0

    for _epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            out = model(Xtr_t[idx]).squeeze(-1)
            loss = loss_fn(out, ytr_t[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            va_pred = torch.sigmoid(model(Xva_t).squeeze(-1)).numpy()
        auc = roc_auc_score(y_va, va_pred)
        if auc > best_auc + 1e-5:
            best_auc = auc
            with torch.no_grad():
                best_te_pred = torch.sigmoid(model(Xte_t).squeeze(-1)).numpy()
            best_va_pred = va_pred
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_va_pred, best_te_pred


def cv_run_nn(X, y, X_test, n_splits, seed=SEED):
    oof = np.zeros(len(y))
    test = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        va_pred, te_pred = _fit_nn_fold(X[tr], y[tr], X[va], y[va], X_test)
        oof[va] = va_pred
        test += te_pred / n_splits
    return oof, test, roc_auc_score(y, oof)


def fit_floor(X_tr, y, X_test):
    """Absolute reliability floor: a single fast LightGBM fit on all of X_tr, no CV, no
    validation split. Must complete in low single digits of seconds even on the largest
    fold, so a valid submission always exists almost immediately. Falls back to a constant
    (overall positive rate) if even this fails for any reason."""
    try:
        from lightgbm import LGBMClassifier
        m = LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31,
                           random_state=SEED, n_jobs=-1, verbosity=-1)
        m.fit(X_tr, y)
        return m.predict_proba(X_test)[:, 1]
    except Exception as e:
        log(f"floor model failed ({e}); falling back to constant prediction")
        rate = float(np.mean(y)) if len(y) else 0.5
        return np.full(len(X_test), rate)


def greedy_weights(oofs: np.ndarray, y: np.ndarray, n_iter: int = 40) -> np.ndarray:
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


def _sample_eval_time_params(model: str, rng: np.random.Generator) -> dict:
    """Lightweight random hyperparameter draw for the bounded eval-time search — same
    spirit and ranges as dev/hpo.py's offline search, inlined here since the skill must be
    self-contained (no import of dev/ tooling inside the sandbox)."""
    lr = float(np.exp(rng.uniform(np.log(0.008), np.log(0.06))))
    if model == "lightgbm":
        return dict(learning_rate=round(lr, 4), num_leaves=int(rng.integers(15, 128)),
                    min_child_samples=int(rng.integers(5, 60)),
                    subsample=round(float(rng.uniform(0.6, 1.0)), 3),
                    colsample_bytree=round(float(rng.uniform(0.6, 1.0)), 3),
                    reg_lambda=round(float(np.exp(rng.uniform(np.log(0.1), np.log(10)))), 4))
    if model == "xgboost":
        return dict(learning_rate=round(lr, 4), max_depth=int(rng.integers(3, 10)),
                    min_child_weight=int(rng.integers(1, 21)),
                    subsample=round(float(rng.uniform(0.6, 1.0)), 3),
                    colsample_bytree=round(float(rng.uniform(0.6, 1.0)), 3),
                    reg_lambda=round(float(np.exp(rng.uniform(np.log(0.1), np.log(10)))), 4))
    return dict(learning_rate=round(lr, 4), depth=int(rng.integers(4, 10)),
               l2_leaf_reg=round(float(np.exp(rng.uniform(np.log(1), np.log(15)))), 4))


# ------------------------------------------------------------------------------- main
_REQUIRED = ("train.csv", "test.csv", "sample_submission.csv")

# Maps a pool member's name -> the params dict used to produce it (base members use {}),
# so STEP 4 (multi-seed reruns) can refit the same config with a different CV seed.
_member_params: dict[str, dict] = {}


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
    ap.add_argument("--time-budget", type=float, default=200.0,
                    help="wall-clock seconds this script is allowed to run (safe default; "
                         "override only if you know the session has more headroom)")
    ap.add_argument("--models", default="lightgbm,xgboost,catboost,mlp")
    ap.add_argument("--splits", type=int, default=None,
                    help="override CV folds for all members; default adapts to dataset size")
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

    X_tr_df, X_te_df, kinds = build_matrices(train, test, feat_cols)
    X_tr, X_te = X_tr_df.values, X_te_df.values
    out_path = a.out if os.path.isabs(a.out) else os.path.join(data_dir, a.out)

    budget = TimeBudget(a.time_budget, len(X_tr))
    pool: list[dict] = []  # {"name", "oof", "test", "auc"}

    def write_from_pool():
        oofs = np.array([m["oof"] for m in pool])
        tests = np.array([m["test"] for m in pool])
        w = greedy_weights(oofs, y) if len(pool) > 1 else np.array([1.0])
        blend_test = (tests * w[:, None]).sum(axis=0)
        sub = sample.copy()
        sub[target] = blend_test
        sub.to_csv(out_path, index=False)
        return w

    # STEP 0 — reliability floor: a valid submission exists almost immediately.
    t0 = time.time()
    floor_pred = fit_floor(X_tr, y, X_te)
    sub = sample.copy()
    sub[target] = floor_pred
    sub.to_csv(out_path, index=False)
    log(f"floor: wrote initial submission in {time.time()-t0:.1f}s")

    n_splits = a.splits if a.splits else (3 if len(X_tr) > 20000 else 5)

    # STEP 1 — base members (validated early-stopping configs), priority order.
    for model in [m for m in ("catboost", "lightgbm", "xgboost") if m in models]:
        if not budget.can_afford(model, n_splits, FULL_MAX_ITERS):
            log(f"skip base {model}: insufficient time (left={budget.left():.0f}s)")
            continue
        try:
            t0 = time.time()
            oof, test_p, auc = cv_run(model, {}, X_tr, y, X_te, n_splits,
                                      FULL_MAX_ITERS, FULL_ES_ROUNDS)
            elapsed = time.time() - t0
            budget.record(model, n_splits, FULL_MAX_ITERS, elapsed)
            name = f"{model}_base"
            pool.append({"name": name, "oof": oof, "test": test_p, "auc": auc})
            _member_params[name] = {}
            write_from_pool()
            log(f"base {model}: auc={auc:.4f} ({elapsed:.1f}s, {budget.left():.0f}s left)")
        except Exception as e:
            log(f"base {model} failed: {e}")

    # STEP 1.5 — MLP member: a structurally different model family, validated to add real,
    # largely independent gains on top of the GBM pool (see module docstring above the MLP
    # functions). Positioned after the 3 GBM base members (so a solid pure-GBM blend always
    # exists first) but before HPO-shortlist tuning (which offline evidence shows adds far
    # less value than this). Entirely optional: any failure (including torch being
    # unavailable) is caught and logged, never affecting the rest of the pipeline.
    if "mlp" in models and budget.can_afford("mlp", n_splits, MLP_MAX_EPOCHS):
        try:
            t0 = time.time()
            X_nn_tr, X_nn_te = preprocess_for_nn(train, test, feat_cols, kinds)
            oof, test_p, auc = cv_run_nn(X_nn_tr, y, X_nn_te, n_splits)
            elapsed = time.time() - t0
            budget.record("mlp", n_splits, MLP_MAX_EPOCHS, elapsed)
            name = "mlp_base"
            pool.append({"name": name, "oof": oof, "test": test_p, "auc": auc})
            _member_params[name] = {}
            write_from_pool()
            log(f"mlp: auc={auc:.4f} ({elapsed:.1f}s, {budget.left():.0f}s left)")
        except Exception as e:
            log(f"mlp failed: {e}")
    elif "mlp" in models:
        log(f"skip mlp: insufficient time (left={budget.left():.0f}s)")

    # STEP 2 — offline-tuned diverse configs (baked shortlist).
    for model in [m for m in ("catboost", "lightgbm", "xgboost") if m in models]:
        for idx, params in enumerate(HPO_SHORTLIST.get(model, [])):
            if not budget.can_afford(model, n_splits, FULL_MAX_ITERS):
                log(f"skip shortlist {model}#{idx}: insufficient time (left={budget.left():.0f}s)")
                continue
            try:
                t0 = time.time()
                oof, test_p, auc = cv_run(model, params, X_tr, y, X_te, n_splits,
                                          FULL_MAX_ITERS, FULL_ES_ROUNDS)
                elapsed = time.time() - t0
                budget.record(model, n_splits, FULL_MAX_ITERS, elapsed)
                name = f"{model}_hpo{idx}"
                pool.append({"name": name, "oof": oof, "test": test_p, "auc": auc})
                _member_params[name] = params
                write_from_pool()
                log(f"shortlist {model}#{idx}: auc={auc:.4f} ({elapsed:.1f}s, {budget.left():.0f}s left)")
            except Exception as e:
                log(f"shortlist {model}#{idx} failed: {e}")

    # STEP 3 — bounded eval-time HPO: cheap-screen random trials, promote winners.
    rng = np.random.default_rng(SEED)
    best_pool_auc = max((m["auc"] for m in pool), default=0.5)
    hpo_models = [m for m in ("catboost", "lightgbm", "xgboost") if m in models]
    trial = 0
    while trial < HPO_TRIAL_BUDGET and hpo_models:
        model = hpo_models[trial % len(hpo_models)]
        if not budget.can_afford(model, HPO_SCREEN_SPLITS, HPO_SCREEN_MAX_ITERS):
            log(f"stop eval-time HPO: insufficient time for screening (left={budget.left():.0f}s)")
            break
        params = _sample_eval_time_params(model, rng)
        try:
            t0 = time.time()
            _, _, screen_auc = cv_run(model, params, X_tr, y, X_te, HPO_SCREEN_SPLITS,
                                      HPO_SCREEN_MAX_ITERS, HPO_SCREEN_ES_ROUNDS)
            budget.record(model, HPO_SCREEN_SPLITS, HPO_SCREEN_MAX_ITERS, time.time() - t0)
            log(f"hpo-trial {trial} {model}: screen_auc={screen_auc:.4f} best_pool={best_pool_auc:.4f}")
            if screen_auc >= best_pool_auc - HPO_PROMOTE_MARGIN and budget.can_afford(model, n_splits, FULL_MAX_ITERS):
                t0 = time.time()
                oof, test_p, auc = cv_run(model, params, X_tr, y, X_te, n_splits,
                                          FULL_MAX_ITERS, FULL_ES_ROUNDS)
                budget.record(model, n_splits, FULL_MAX_ITERS, time.time() - t0)
                name = f"{model}_evalhpo{trial}"
                pool.append({"name": name, "oof": oof, "test": test_p, "auc": auc})
                _member_params[name] = params
                write_from_pool()
                best_pool_auc = max(best_pool_auc, auc)
                log(f"promoted hpo-trial {trial} {model}: auc={auc:.4f} ({budget.left():.0f}s left)")
        except Exception as e:
            log(f"eval-time hpo trial {trial} ({model}) failed: {e}")
        trial += 1

    # STEP 4 — multi-seed reruns of the current best single member (variance reduction).
    reseeds_done = 0
    while reseeds_done < MAX_RESEEDS and pool:
        best_member = max(pool, key=lambda m: m["auc"])
        best_model = best_member["name"].split("_")[0]
        cost_shape = MLP_MAX_EPOCHS if best_model == "mlp" else FULL_MAX_ITERS
        if not budget.can_afford(best_model, n_splits, cost_shape):
            log(f"stop reseeding: insufficient time (left={budget.left():.0f}s)")
            break
        # recover the params used for the best member: base members use {}, shortlist/hpo
        # members carry their params in a side table populated when added.
        params = _member_params.get(best_member["name"], {})
        try:
            t0 = time.time()
            new_seed = SEED + 1 + reseeds_done
            if best_model == "mlp":
                X_nn_tr, X_nn_te = preprocess_for_nn(train, test, feat_cols, kinds)
                oof, test_p, auc = cv_run_nn(X_nn_tr, y, X_nn_te, n_splits, seed=new_seed)
            else:
                oof, test_p, auc = cv_run(best_model, params, X_tr, y, X_te, n_splits,
                                          FULL_MAX_ITERS, FULL_ES_ROUNDS, seed=new_seed)
            budget.record(best_model, n_splits, cost_shape, time.time() - t0)
            name = f"{best_member['name']}_seed{new_seed}"
            pool.append({"name": name, "oof": oof, "test": test_p, "auc": auc})
            _member_params[name] = params
            write_from_pool()
            log(f"reseed of {best_member['name']} (seed={new_seed}): auc={auc:.4f} "
                f"({budget.left():.0f}s left)")
        except Exception as e:
            log(f"reseed failed: {e}")
        reseeds_done += 1

    # FINALIZE
    if not pool:
        log("no pool members completed; floor submission remains the final output")
        final_weights = {}
        blend_auc = None
    else:
        w = write_from_pool()
        final_weights = {pool[i]["name"]: round(float(w[i]), 3) for i in range(len(pool)) if w[i] > 0}
        oofs = np.array([m["oof"] for m in pool])
        blend_auc = round(float(roc_auc_score(y, (oofs * w[:, None]).sum(axis=0))), 5) if len(pool) > 1 \
            else round(float(pool[0]["auc"]), 5)

    n_cat = sum(1 for c in feat_cols if kinds[c] == "cat")
    n_ord = sum(1 for c in feat_cols if kinds[c] == "ord")
    n_num = sum(1 for c in feat_cols if kinds[c] == "num")
    summary = {
        "status": "ok",
        "output": out_path,
        "rows": len(sample),
        "n_features": len(feat_cols),
        "schema": {"num": n_num, "ord": n_ord, "cat": n_cat},
        "time_budget_s": a.time_budget,
        "elapsed_s": round(budget.elapsed(), 1),
        "pool_size": len(pool),
        "pool_aucs": {m["name"]: round(float(m["auc"]), 5) for m in pool},
        "blend_weights": final_weights,
        "blend_oof_auc": blend_auc,
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
