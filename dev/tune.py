"""Phase 1b tuning harness: test model/encoding variants across all 16 folds, scored vs
solution.csv. Compares against the deployed baseline (greedy 3-GBM blend, mean full 0.798).

Variants:
  cat_onehot  : CatBoost on the one-hot design (current baseline encoding)
  cat_native  : CatBoost with NATIVE categorical handling (cat_* passed as categories)
  cat_native_es : cat_native + more iterations + early stopping
  lgb_native_es : LightGBM native categorical + early stopping

Usage: python tune.py --variant cat_native_es [--folds all] [--splits 5]
"""
from __future__ import annotations
import argparse, glob, os, time
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import pipeline as P

BASE = os.path.join(os.path.dirname(__file__), "..", "data")
_ORD_RE = r"^\s*ord_?\d+\s*$"


def native_design(train, test, feat_cols):
    """ord_* -> int (numeric); cat_* / other string -> string category; numeric -> float NaN kept.
    Returns combined-encoded X_tr, X_te and the list of categorical column names."""
    kinds = P.analyze_columns(train, feat_cols)
    ord_cols = [c for c in feat_cols if kinds[c] == "ord"]
    cat_cols = [c for c in feat_cols if kinds[c] == "cat"]
    num_cols = [c for c in feat_cols if kinds[c] == "num"]
    n_tr = len(train)
    comb = pd.concat([train[feat_cols], test[feat_cols]], axis=0, ignore_index=True)
    out = pd.DataFrame(index=comb.index)
    for c in ord_cols:
        out[c] = pd.to_numeric(comb[c].astype("string").str.extract(r"(\d+)", expand=False), errors="coerce").astype(np.float32)
    for c in num_cols:
        out[c] = comb[c].astype(np.float32)
    for c in cat_cols:
        # CatBoost needs categorical cells as str with no NaN
        out[c] = comb[c].astype("string").fillna("NA").astype(str)
    return out.iloc[:n_tr].reset_index(drop=True), out.iloc[n_tr:].reset_index(drop=True), cat_cols


def true_aucs(data_dir, id_col, target, sample, pred):
    sol = pd.read_csv(os.path.join(data_dir, "solution.csv"))
    keep = [id_col, target] + (["Usage"] if "Usage" in sol.columns else [])
    m = sample[[id_col]].merge(sol[keep], on=id_col, how="left")
    yt = m[target].to_numpy(); out = {"full": roc_auc_score(yt, pred)}
    if "Usage" in m.columns:
        for s in ("Public", "Private"):
            mask = m["Usage"].to_numpy() == s
            out[s.lower()] = roc_auc_score(yt[mask], pred[mask])
    return out


def run_variant(variant, data_dir, splits):
    train, test, sample, id_col, target, feat_cols = P.load_fold(data_dir)
    y = train[target].to_numpy()
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=P.SEED)

    if variant in ("cat_onehot", "cat_onehot_es", "xgb_onehot_es", "lgb_onehot_es"):
        X_tr, X_te, _ = P.build_matrices(train, test, feat_cols)
        Xv, Xt = X_tr.values, X_te.values
        oof, test_p = np.zeros(len(y)), np.zeros(len(X_te))
        for tr, va in skf.split(Xv, y):
            if variant == "cat_onehot":
                from catboost import CatBoostClassifier
                m = CatBoostClassifier(iterations=600, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
                                       random_seed=P.SEED, thread_count=-1, verbose=False, allow_writing_files=False)
                m.fit(Xv[tr], y[tr])
            elif variant == "cat_onehot_es":
                from catboost import CatBoostClassifier
                m = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
                                       eval_metric="AUC", random_seed=P.SEED, thread_count=-1, verbose=False,
                                       allow_writing_files=False, early_stopping_rounds=150)
                m.fit(Xv[tr], y[tr], eval_set=(Xv[va], y[va]), verbose=False)
            elif variant == "lgb_onehot_es":
                from lightgbm import LGBMClassifier, early_stopping
                m = LGBMClassifier(n_estimators=3000, learning_rate=0.02, num_leaves=31, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                   random_state=P.SEED, n_jobs=-1, verbosity=-1)
                m.fit(Xv[tr], y[tr], eval_set=[(Xv[va], y[va])], eval_metric="auc",
                      callbacks=[early_stopping(150, verbose=False)])
            else:  # xgb_onehot_es
                from xgboost import XGBClassifier
                m = XGBClassifier(n_estimators=3000, learning_rate=0.02, max_depth=5, subsample=0.8,
                                  colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", eval_metric="auc",
                                  early_stopping_rounds=150, random_state=P.SEED, n_jobs=-1, verbosity=0)
                m.fit(Xv[tr], y[tr], eval_set=[(Xv[va], y[va])], verbose=False)
            oof[va] = m.predict_proba(Xv[va])[:,1]; test_p += m.predict_proba(Xt)[:,1]/splits
        return oof, test_p, y, id_col, target, sample

    Xtr, Xte, cat_cols = native_design(train, test, feat_cols)
    oof, test_p = np.zeros(len(y)), np.zeros(len(Xte))

    if variant in ("cat_native", "cat_native_es"):
        from catboost import CatBoostClassifier, Pool
        cat_idx = [Xtr.columns.get_loc(c) for c in cat_cols]
        for tr, va in skf.split(Xtr, y):
            if variant == "cat_native_es":
                m = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
                                       eval_metric="AUC", random_seed=P.SEED, thread_count=-1,
                                       verbose=False, allow_writing_files=False, early_stopping_rounds=150)
                m.fit(Pool(Xtr.iloc[tr], y[tr], cat_features=cat_idx),
                      eval_set=Pool(Xtr.iloc[va], y[va], cat_features=cat_idx), verbose=False)
            else:
                m = CatBoostClassifier(iterations=600, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
                                       random_seed=P.SEED, thread_count=-1, verbose=False, allow_writing_files=False)
                m.fit(Pool(Xtr.iloc[tr], y[tr], cat_features=cat_idx))
            oof[va] = m.predict_proba(Xtr.iloc[va])[:,1]
            test_p += m.predict_proba(Xte)[:,1]/splits
        return oof, test_p, y, id_col, target, sample

    if variant == "lgb_native_es":
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation
        Xc = Xtr.copy(); Xtc = Xte.copy()
        for c in cat_cols:
            Xc[c] = Xc[c].astype("category"); Xtc[c] = pd.Categorical(Xtc[c], categories=Xc[c].cat.categories)
        for tr, va in skf.split(Xc, y):
            m = LGBMClassifier(n_estimators=3000, learning_rate=0.02, num_leaves=31, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                               random_state=P.SEED, n_jobs=-1, verbosity=-1)
            m.fit(Xc.iloc[tr], y[tr], eval_set=[(Xc.iloc[va], y[va])], eval_metric="auc",
                  categorical_feature=cat_cols, callbacks=[early_stopping(150, verbose=False)])
            oof[va] = m.predict_proba(Xc.iloc[va])[:,1]
            test_p += m.predict_proba(Xtc)[:,1]/splits
        return oof, test_p, y, id_col, target, sample

    raise ValueError(variant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="cat_native_es")
    ap.add_argument("--folds", default="all")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--save", action="store_true", help="cache oof/test arrays to dev/tuned_cache/")
    a = ap.parse_args()
    tcache = os.path.join(os.path.dirname(__file__), "tuned_cache"); os.makedirs(tcache, exist_ok=True)
    dirs = sorted(glob.glob(os.path.join(BASE, "train_*")))
    if a.folds != "all":
        want = set(a.folds.split(",")); dirs = [d for d in dirs if os.path.basename(d) in want]
    rows = []; t0 = time.time()
    for d in dirs:
        name = os.path.basename(d); ft = time.time()
        oof, test_p, y, id_col, target, sample = run_variant(a.variant, d, a.splits)
        if a.save:
            np.savez(os.path.join(tcache, f"{name}__{a.variant}.npz"), oof=oof, test=test_p)
        au = true_aucs(d, id_col, target, sample, test_p)
        rows.append(dict(fold=name, oof=round(roc_auc_score(y, oof),4), full=round(au["full"],4),
                         private=round(au.get("private", float('nan')),4), sec=round(time.time()-ft,1)))
        print(f"  {name}: oof={rows[-1]['oof']:.4f} full={rows[-1]['full']:.4f} priv={rows[-1]['private']:.4f} ({rows[-1]['sec']}s)", flush=True)
    df = pd.DataFrame(rows)
    print("="*80); print(f"VARIANT {a.variant} splits={a.splits}")
    print(df.to_string(index=False))
    print(f"MEAN full={df.full.mean():.4f} private={df.private.mean():.4f} oof={df.oof.mean():.4f}  (baseline greedy full=0.7982)")
    print(f"time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
