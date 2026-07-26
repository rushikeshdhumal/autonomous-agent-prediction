"""Cheap stacking test (Phase 1b, bigger-lever exploration): does a meta-learner beat
greedy blending? Uses ONLY already-cached OOF/test arrays from this session
(dev/tuned_cache/ + dev/hpo_cache/) -- no new model training, just fitting a small
meta-learner on 9-dimensional feature vectors (near-instant).

To avoid the meta-learner overfitting to noise in its own training data, meta-level OOF
predictions are produced via cross_val_predict (a fresh K-fold on top of the base OOF
matrix), which is the standard way to fairly estimate stacking performance offline.

Usage: python stacking_test.py [--meta logreg|gbm]
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

import pipeline as P

BASE = os.path.join(os.path.dirname(__file__), "..", "data")
TUNED_CACHE = os.path.join(os.path.dirname(__file__), "tuned_cache")
HPO_CACHE = os.path.join(os.path.dirname(__file__), "hpo_cache")

BASE_VARIANTS = ["lgb_onehot_es", "cat_onehot_es", "xgb_onehot_es"]
HPO_VARIANTS = [f"{m}_hpo{i}" for m in ("lightgbm", "xgboost", "catboost") for i in range(2)]
ALL_VARIANTS = BASE_VARIANTS + HPO_VARIANTS


def _load(name: str, variant: str):
    cache_dir = TUNED_CACHE if variant in BASE_VARIANTS else HPO_CACHE
    fp = os.path.join(cache_dir, f"{name}__{variant}.npz")
    if not os.path.exists(fp):
        return None
    z = np.load(fp)
    return z["oof"], z["test"]


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


def _greedy_weights(oofs, y, n_iter=40):
    M = len(oofs)
    singles = [roc_auc_score(y, oofs[i]) for i in range(M)]
    picks = [int(np.argmax(singles))]; best = max(singles)
    for _ in range(n_iter):
        cur = oofs[picks].mean(axis=0)
        sc = [roc_auc_score(y, (cur * len(picks) + oofs[i]) / (len(picks) + 1)) for i in range(M)]
        j = int(np.argmax(sc))
        if sc[j] <= best + 1e-6:
            break
        picks.append(j); best = sc[j]
    return np.bincount(picks, minlength=M) / len(picks)


def make_meta(kind: str):
    if kind == "logreg":
        return LogisticRegression(max_iter=2000, C=1.0)
    if kind == "gbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=100, max_depth=2, learning_rate=0.05,
                              num_leaves=7, random_state=42, verbosity=-1)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="logreg", choices=["logreg", "gbm"])
    ap.add_argument("--splits", type=int, default=5)
    args = ap.parse_args()

    rows = []
    for d in sorted(glob.glob(os.path.join(BASE, "train_*"))):
        name = os.path.basename(d)
        train, test, sample, id_col, target, feat_cols = P.load_fold(d)
        classes = np.sort(train[target].unique())
        y = (train[target].to_numpy() == classes[1]).astype(int)

        oofs, tests, used = [], [], []
        for v in ALL_VARIANTS:
            r = _load(name, v)
            if r is None:
                continue
            oofs.append(r[0]); tests.append(r[1]); used.append(v)
        oofs, tests = np.array(oofs), np.array(tests)  # (M, n)
        Xmeta_tr = oofs.T   # (n_train, M)
        Xmeta_te = tests.T  # (n_test, M)

        # --- greedy blend (current deployed approach) ---
        w = _greedy_weights(oofs, y)
        greedy_test = (tests * w[:, None]).sum(0)
        greedy_oof = (oofs * w[:, None]).sum(0)
        au_greedy = _true_aucs(d, id_col, target, sample, greedy_test)

        # --- stacking meta-learner ---
        skf = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=42)
        meta_oof = cross_val_predict(make_meta(args.meta), Xmeta_tr, y, cv=skf,
                                     method="predict_proba")[:, 1]
        meta_model = make_meta(args.meta)
        meta_model.fit(Xmeta_tr, y)
        meta_test = meta_model.predict_proba(Xmeta_te)[:, 1]
        au_stack = _true_aucs(d, id_col, target, sample, meta_test)

        rows.append(dict(fold=name, n_members=len(used),
                         greedy_oof=round(roc_auc_score(y, greedy_oof), 4),
                         greedy_full=round(au_greedy["full"], 4),
                         greedy_priv=round(au_greedy.get("private", float("nan")), 4),
                         stack_oof=round(roc_auc_score(y, meta_oof), 4),
                         stack_full=round(au_stack["full"], 4),
                         stack_priv=round(au_stack.get("private", float("nan")), 4)))

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print("-" * 100)
    print(f"GREEDY   mean full={df.greedy_full.mean():.4f} private={df.greedy_priv.mean():.4f}")
    print(f"STACKING mean full={df.stack_full.mean():.4f} private={df.stack_priv.mean():.4f}  (meta={args.meta})")
    n_stack_wins = (df.stack_full > df.greedy_full).sum()
    print(f"Stacking beats greedy on {n_stack_wins}/{len(df)} folds")


if __name__ == "__main__":
    main()
