"""Cache per-model OOF + bagged-test probability arrays once per fold, then search
blends cheaply on the cached arrays (compute-OOF-once, reuse — cf. plan doc §13.7).

Pass 1 (expensive, ~minutes):   python experiment.py build --models lightgbm,xgboost,catboost,hgb
Pass 2 (cheap, seconds):        python experiment.py blend --models lightgbm,xgboost,catboost
                                python experiment.py greedy --models lightgbm,xgboost,catboost,hgb
Caches live in dev/cache/<fold>__<model>.npz  and  dev/cache/<fold>__meta.npz
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import pipeline as P

BASE = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE, exist_ok=True)


def fold_names(spec: str):
    dirs = sorted(glob.glob(os.path.join(BASE, "train_*")))
    if spec != "all":
        want = set(spec.split(","))
        dirs = [d for d in dirs if os.path.basename(d) in want]
    return dirs


def true_test_labels(data_dir, id_col, target, sample):
    sol = pd.read_csv(os.path.join(data_dir, "solution.csv"))
    keep = [id_col, target] + (["Usage"] if "Usage" in sol.columns else [])
    # left-merge onto sample[id] to preserve prediction row order; take sol's target only
    merged = sample[[id_col]].merge(sol[keep], on=id_col, how="left")
    y_test = merged[target].to_numpy()
    usage = merged["Usage"].to_numpy() if "Usage" in merged.columns else None
    return y_test, usage


def build(models, folds, splits, max_card):
    for d in fold_names(folds):
        name = os.path.basename(d)
        train, test, sample, id_col, target, feat_cols = P.load_fold(d)
        y = train[target].to_numpy()
        X_tr, X_te, kinds = P.build_matrices(train, test, feat_cols)
        y_test, usage = true_test_labels(d, id_col, target, sample)
        np.savez(os.path.join(CACHE, f"{name}__meta.npz"),
                 y=y, y_test=y_test, usage=usage if usage is not None else np.array([]))
        for m in models:
            fp = os.path.join(CACHE, f"{name}__{m}.npz")
            if os.path.exists(fp):
                print(f"  {name}/{m}: cached, skip")
                continue
            oof, tst, auc = P.cv_oof_test(X_tr, y, X_te, m, n_splits=splits)
            np.savez(fp, oof=oof, test=tst, oof_auc=auc)
            print(f"  {name}/{m}: oof={auc:.4f} test={roc_auc_score(y_test, tst):.4f}")


def _load(name, m):
    d = np.load(os.path.join(CACHE, f"{name}__{m}.npz"))
    return d["oof"], d["test"]


def _meta(name):
    d = np.load(os.path.join(CACHE, f"{name}__meta.npz"), allow_pickle=True)
    return d["y"], d["y_test"], (d["usage"] if d["usage"].size else None)


def _test_auc(y_test, pred, usage, split="private"):
    if usage is None or split == "full":
        return roc_auc_score(y_test, pred)
    mask = usage == split.capitalize()
    return roc_auc_score(y_test[mask], pred[mask])


def blend(models, folds):
    rows = []
    for d in fold_names(folds):
        name = os.path.basename(d)
        y, y_test, usage = _meta(name)
        oofs, tests = [], []
        for m in models:
            o, t = _load(name, m)
            oofs.append(o); tests.append(t)
        boof = np.mean(oofs, axis=0)
        btest = np.mean(tests, axis=0)
        rows.append(dict(fold=name, oof=roc_auc_score(y, boof),
                         full=_test_auc(y_test, btest, usage, "full"),
                         private=_test_auc(y_test, btest, usage, "private"),
                         **{f"t_{m}": _test_auc(y_test, _load(name, m)[1], usage, "full") for m in models}))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)
    print(df.round(4).to_string(index=False))
    print(f"\nBLEND {models}  MEAN oof={df.oof.mean():.4f} full={df.full.mean():.4f} private={df.private.mean():.4f}")
    for m in models:
        print(f"   single {m}: mean full={df[f't_{m}'].mean():.4f}")
    return df


def greedy(models, folds):
    """Per-fold Caruana greedy weight selection on OOF, then apply weights to test."""
    rows = []
    for d in fold_names(folds):
        name = os.path.basename(d)
        y, y_test, usage = _meta(name)
        oofs = np.array([_load(name, m)[0] for m in models])   # (M, n)
        tests = np.array([_load(name, m)[1] for m in models])  # (M, nt)
        M = len(models)
        picks = []
        # init with best single
        singles = [roc_auc_score(y, oofs[i]) for i in range(M)]
        picks.append(int(np.argmax(singles)))
        best = max(singles)
        for _ in range(30):
            cur = oofs[picks].mean(axis=0)
            cand_scores = []
            for i in range(M):
                trial = (cur * len(picks) + oofs[i]) / (len(picks) + 1)
                cand_scores.append(roc_auc_score(y, trial))
            j = int(np.argmax(cand_scores))
            if cand_scores[j] <= best + 1e-6:
                break
            picks.append(j); best = cand_scores[j]
        w = np.bincount(picks, minlength=M) / len(picks)
        btest = (tests * w[:, None]).sum(axis=0)
        boof = (oofs * w[:, None]).sum(axis=0)
        rows.append(dict(fold=name, oof=roc_auc_score(y, boof),
                         full=_test_auc(y_test, btest, usage, "full"),
                         private=_test_auc(y_test, btest, usage, "private"),
                         w=",".join(f"{models[i][:3]}:{w[i]:.2f}" for i in range(M) if w[i] > 0)))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)
    print(df.round(4).to_string(index=False))
    print(f"\nGREEDY {models}  MEAN oof={df.oof.mean():.4f} full={df.full.mean():.4f} private={df.private.mean():.4f}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "blend", "greedy"])
    ap.add_argument("--models", default="lightgbm,xgboost,catboost")
    ap.add_argument("--folds", default="all")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--max-card", type=int, default=20)
    a = ap.parse_args()
    ms = a.models.split(",")
    if a.cmd == "build":
        build(ms, a.folds, a.splits, a.max_card)
    elif a.cmd == "blend":
        blend(ms, a.folds)
    elif a.cmd == "greedy":
        greedy(ms, a.folds)
