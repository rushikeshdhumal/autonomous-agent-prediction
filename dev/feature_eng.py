"""Feature engineering experiment (Phase 1b, bigger-lever exploration).

Generic, domain-agnostic derived features GBMs can't trivially discover from one-hot/ordinal
columns alone, since the data is anonymized (feature_0, feature_1, ...) with no domain
semantics to exploit:

  - row-wise missingness count: how many of this row's raw features are NaN (a holistic
    "how much is missing here" signal; the per-column dummy_na dummies only capture
    column-level missingness, not the row-level pattern).
  - row-wise numeric aggregates on RANK-NORMALIZED num/ord columns: raw numeric columns have
    wildly different scales/units, so aggregating them directly (mean/std/min/max) would mix
    incompatible units; rank-normalizing each column to [0,1] first makes a row-level summary
    meaningful.
  - pairwise categorical interactions: concatenate pairs of low-cardinality categorical
    columns into a new joint category, one-hot encoded -- captures interactions invisible to
    the marginal one-hot encoding of each column alone. Capped to a few columns/pairs to avoid
    combinatorial blowup.

Usage: python feature_eng.py --model catboost [--folds all]
"""
from __future__ import annotations

import argparse
import glob
import itertools
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import pipeline as P

BASE = os.path.join(os.path.dirname(__file__), "..", "data")
MAX_CAT_COLS_FOR_PAIRS = 4  # cap pairwise interactions to avoid combinatorial blowup


def add_engineered_features(train: pd.DataFrame, test: pd.DataFrame, feat_cols: list, kinds: dict):
    """Returns extra columns (as a DataFrame, train+test combined) to concat onto the
    existing one-hot/ordinal design matrix."""
    num_ord_cols = [c for c in feat_cols if kinds[c] in ("num", "ord")]
    cat_cols = [c for c in feat_cols if kinds[c] == "cat"]
    n_tr = len(train)
    combined_raw = pd.concat([train[feat_cols], test[feat_cols]], axis=0, ignore_index=True)
    extra = pd.DataFrame(index=combined_raw.index)

    # 1. row-wise missingness count (any column, works regardless of kind)
    extra["_nan_count"] = combined_raw.isna().sum(axis=1).astype(np.float32)

    # 2. rank-normalized numeric/ordinal row-wise aggregates
    if num_ord_cols:
        ranked = pd.DataFrame(index=combined_raw.index)
        for c in num_ord_cols:
            if kinds[c] == "ord":
                v = pd.to_numeric(combined_raw[c].astype("string").str.extract(r"(\d+)", expand=False),
                                  errors="coerce")
            else:
                v = combined_raw[c].astype(np.float64)
            ranked[c] = v.rank(pct=True, na_option="keep")
        extra["_rank_mean"] = ranked.mean(axis=1, skipna=True).astype(np.float32)
        extra["_rank_std"] = ranked.std(axis=1, skipna=True).astype(np.float32)
        extra["_rank_min"] = ranked.min(axis=1, skipna=True).astype(np.float32)
        extra["_rank_max"] = ranked.max(axis=1, skipna=True).astype(np.float32)

    # 3. pairwise categorical interactions (capped) -- optional, off by default (see below):
    # naive one-hot of the joint category tends to explode column count on small folds and
    # hurt via dimensionality (empirically confirmed: up to +413 columns, net negative AUC).
    if _INCLUDE_INTERACTIONS:
        pair_cols = cat_cols[:MAX_CAT_COLS_FOR_PAIRS]
        interaction_dummies = []
        for c1, c2 in itertools.combinations(pair_cols, 2):
            joint = combined_raw[c1].astype("string").fillna("NA") + "__" + combined_raw[c2].astype("string").fillna("NA")
            interaction_dummies.append(pd.get_dummies(joint, prefix=f"{c1}x{c2}", dummy_na=False, dtype=np.float32))
        if interaction_dummies:
            extra = pd.concat([extra] + interaction_dummies, axis=1)

    return extra.iloc[:n_tr].reset_index(drop=True), extra.iloc[n_tr:].reset_index(drop=True)


_INCLUDE_INTERACTIONS = False


def build_matrices_fe(train, test, feat_cols):
    """build_matrices (existing encoding) + engineered features appended."""
    X_tr, X_te, kinds = P.build_matrices(train, test, feat_cols)
    extra_tr, extra_te = add_engineered_features(train, test, feat_cols, kinds)
    X_tr_fe = pd.concat([X_tr, extra_tr], axis=1)
    X_te_fe = pd.concat([X_te, extra_te], axis=1)
    return X_tr_fe, X_te_fe, kinds


def _true_aucs(data_dir, id_col, target, sample, pred):
    sol = pd.read_csv(os.path.join(data_dir, "solution.csv"))
    keep = [id_col, target] + (["Usage"] if "Usage" in sol.columns else [])
    m = sample[[id_col]].merge(sol[keep], on=id_col, how="left")
    yt = m[target].to_numpy(); out = {"full": roc_auc_score(yt, pred)}
    if "Usage" in m.columns:
        for s in ("Public", "Private"):
            mask = m["Usage"].to_numpy() == s
            out[s.lower()] = roc_auc_score(yt[mask], pred[mask])
    return out


def cv_run(model, X, y, X_test, n_splits, max_iters=3000, es_rounds=150, seed=42):
    Xv, Xt = X, X_test
    oof = np.zeros(len(y)); test = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(Xv, y):
        if model == "lightgbm":
            from lightgbm import LGBMClassifier, early_stopping
            m = LGBMClassifier(n_estimators=max_iters, learning_rate=0.02, num_leaves=31,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                               reg_lambda=1.0, random_state=seed, n_jobs=-1, verbosity=-1)
            m.fit(Xv[tr], y[tr], eval_set=[(Xv[va], y[va])], eval_metric="auc",
                  callbacks=[early_stopping(es_rounds, verbose=False)])
        elif model == "catboost":
            from catboost import CatBoostClassifier
            m = CatBoostClassifier(iterations=max_iters, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
                                   eval_metric="AUC", random_seed=seed, thread_count=-1, verbose=False,
                                   allow_writing_files=False, early_stopping_rounds=es_rounds)
            m.fit(Xv[tr], y[tr], eval_set=(Xv[va], y[va]), verbose=False)
        else:
            raise ValueError(model)
        oof[va] = m.predict_proba(Xv[va])[:, 1]
        test += m.predict_proba(Xt)[:, 1] / n_splits
    return oof, test, roc_auc_score(y, oof)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="catboost", choices=["lightgbm", "catboost"])
    ap.add_argument("--folds", default="all")
    ap.add_argument("--splits", type=int, default=5)
    a = ap.parse_args()

    dirs = sorted(glob.glob(os.path.join(BASE, "train_*")))
    if a.folds != "all":
        want = set(a.folds.split(","))
        dirs = [d for d in dirs if os.path.basename(d) in want]

    rows = []
    t0 = time.time()
    for d in dirs:
        name = os.path.basename(d)
        train, test, sample, id_col, target, feat_cols = P.load_fold(d)
        classes = np.sort(train[target].unique())
        y = (train[target].to_numpy() == classes[1]).astype(int)

        # baseline (no FE)
        Xb_tr, Xb_te, _ = P.build_matrices(train, test, feat_cols)
        ft = time.time()
        oof_b, test_b, auc_b = cv_run(a.model, Xb_tr.values, y, Xb_te.values, a.splits)
        au_b = _true_aucs(d, id_col, target, sample, test_b)
        t_baseline = time.time() - ft

        # with feature engineering
        Xf_tr, Xf_te, _ = build_matrices_fe(train, test, feat_cols)
        ft = time.time()
        oof_f, test_f, auc_f = cv_run(a.model, Xf_tr.values, y, Xf_te.values, a.splits)
        au_f = _true_aucs(d, id_col, target, sample, test_f)
        t_fe = time.time() - ft

        rows.append(dict(fold=name, n_extra=Xf_tr.shape[1] - Xb_tr.shape[1],
                         base_oof=round(auc_b, 4), base_full=round(au_b["full"], 4),
                         fe_oof=round(auc_f, 4), fe_full=round(au_f["full"], 4),
                         delta=round(au_f["full"] - au_b["full"], 4),
                         t_base=round(t_baseline, 1), t_fe=round(t_fe, 1)))
        print(f"  {name}: base_full={au_b['full']:.4f} fe_full={au_f['full']:.4f} "
              f"delta={au_f['full']-au_b['full']:+.4f} (+{Xf_tr.shape[1]-Xb_tr.shape[1]} cols)", flush=True)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("=" * 100)
    print(df.to_string(index=False))
    print("-" * 100)
    print(f"MODEL={a.model}  BASE mean full={df.base_full.mean():.4f}  "
          f"FE mean full={df.fe_full.mean():.4f}  mean delta={df.delta.mean():+.4f}")
    print(f"FE wins on {(df.delta > 0).sum()}/{len(df)} folds. time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
