"""Offline scorer: run pipeline.run_fold on all 16 folds, score test predictions
against the local solution.csv (ground truth), report per-fold OOF AUC (what the
agent would see internally) vs true TEST AUC (public/private/full).

No LLM, no Docker — the fast/free Phase-1 dev loop. Usage:
    python score_all_folds.py --models lightgbm
    python score_all_folds.py --models lightgbm,xgboost,catboost
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import pipeline as P

BASE = os.path.join(os.path.dirname(__file__), "..", "data")


def true_aucs(data_dir: str, id_col: str, target: str, sample: pd.DataFrame, pred: np.ndarray):
    """Score predictions vs solution.csv on public / private / full splits."""
    sol = pd.read_csv(os.path.join(data_dir, "solution.csv"))
    sub = sample.copy()
    sub[target] = pred
    merged = sol.merge(sub, on=id_col, suffixes=("_true", "_pred"))
    yt = merged[f"{target}_true"].to_numpy()
    yp = merged[f"{target}_pred"].to_numpy()
    out = {"full": roc_auc_score(yt, yp)}
    if "Usage" in merged.columns:
        for split in ("Public", "Private"):
            m = merged[merged["Usage"] == split]
            out[split.lower()] = roc_auc_score(m[f"{target}_true"], m[f"{target}_pred"]) if len(m) else float("nan")
    return out


def datamd_cats(data_dir: str, feat_cols: list[str]) -> set[str]:
    """True categorical columns per DATA.md (diagnostic only — not available at eval time)."""
    p = os.path.join(data_dir, "DATA.md")
    cats = set()
    if not os.path.exists(p):
        return cats
    for line in open(p, encoding="utf-8"):
        m = re.search(r"`(feature_\d+)`\s*:\s*([a-z ]+)", line, re.I)
        if m and m.group(2).strip().lower() == "categorical":
            cats.add(m.group(1))
    return cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="lightgbm")
    ap.add_argument("--folds", default="all", help="'all' or comma list e.g. train_01,train_08")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--max-card", type=int, default=20)
    args = ap.parse_args()
    models = tuple(args.models.split(","))

    fold_dirs = sorted(glob.glob(os.path.join(BASE, "train_*")))
    if args.folds != "all":
        want = set(args.folds.split(","))
        fold_dirs = [d for d in fold_dirs if os.path.basename(d) in want]

    rows = []
    t0 = time.time()
    for d in fold_dirs:
        name = os.path.basename(d)
        ft = time.time()
        res = P.run_fold(d, models=models, n_splits=args.splits, max_card=args.max_card)
        aucs = true_aucs(d, res["id_col"], res["target"], res["sample"], res["test_pred"])
        # diagnostic: how well did data-only inference match DATA.md categoricals?
        _, _, _, _, _, feat_cols = P.load_fold(d)
        true_cats = datamd_cats(d, feat_cols)
        inferred = set(res["cat_cols"])
        cat_match = "n/a" if not true_cats else f"{len(true_cats & inferred)}/{len(true_cats)}(+{len(inferred - true_cats)})"
        row = dict(fold=name, n=res["n_train"], oof=round(res["blend_oof_auc"], 4),
                   full=round(aucs["full"], 4), public=round(aucs.get("public", float('nan')), 4),
                   private=round(aucs.get("private", float('nan')), 4),
                   ncat=res["n_cat"], catmatch=cat_match, sec=round(time.time() - ft, 1))
        for m in models:
            row[f"oof_{m}"] = round(res["per_model_auc"][m], 4)
        rows.append(row)
        print(f"  {name}: oof={row['oof']:.4f} full={row['full']:.4f} "
              f"pub={row['public']:.4f} priv={row['private']:.4f} ({row['sec']}s)")

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    print("\n" + "=" * 100)
    print(f"MODELS: {models} | splits={args.splits} | max_card={args.max_card}")
    print(df.to_string(index=False))
    print("-" * 100)
    print(f"MEAN  oof={df.oof.mean():.4f}  full={df.full.mean():.4f}  "
          f"public={df.public.mean():.4f}  private={df.private.mean():.4f}")
    print(f"MIN   oof={df.oof.min():.4f}  full={df.full.min():.4f}  (worst fold: {df.loc[df.full.idxmin(),'fold']})")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
