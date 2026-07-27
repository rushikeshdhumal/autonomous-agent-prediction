"""Offline anytime-pipeline validator (Phase 1b, step 3).

Runs the ACTUAL deployed skill script
(submissions/01_gbm_blend/agent/skills/auto-ml/scripts/run_pipeline.py) as a subprocess
against each of the 16 folds with a chosen time budget, scoring its output against the
local solution.csv. This validates exactly what will be deployed -- no reimplementation,
no drift between "what we test" and "what we ship".

Usage:
    python eval_anytime.py --folds all --time-budget 300
    python eval_anytime.py --folds train_01,train_12 --time-budget 30   # tests the anytime floor
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = os.path.join(os.path.dirname(__file__), "..", "data")
SKILL = os.path.join(os.path.dirname(__file__), "..", "submissions", "01_gbm_blend", "agent",
                     "skills", "auto-ml", "scripts", "run_pipeline.py")


def true_aucs(data_dir: str, id_col: str, target: str, pred_df: pd.DataFrame) -> dict:
    sol = pd.read_csv(os.path.join(data_dir, "solution.csv"))
    merged = pred_df.merge(sol, on=id_col, suffixes=("_pred", "_true"))
    yt = merged[f"{target}_true"].to_numpy()
    yp = merged[f"{target}_pred"].to_numpy()
    out = {"full": roc_auc_score(yt, yp)}
    if "Usage" in merged.columns:
        for s in ("Public", "Private"):
            mask = merged["Usage"].to_numpy() == s
            out[s.lower()] = roc_auc_score(yt[mask], yp[mask]) if mask.any() else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="all")
    ap.add_argument("--time-budget", type=float, default=300.0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    dirs = sorted(glob.glob(os.path.join(BASE, "train_*")))
    if args.folds != "all":
        want = set(args.folds.split(","))
        dirs = [d for d in dirs if os.path.basename(d) in want]

    out_dir = args.out_dir or os.path.join(os.path.dirname(__file__), "_anytime_tmp")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    t0 = time.time()
    for d in dirs:
        name = os.path.basename(d)
        out_csv = os.path.join(out_dir, f"{name}_submission.csv")
        ft = time.time()
        proc = subprocess.run(
            [sys.executable, SKILL, "--data-dir", d, "--out", out_csv,
             "--time-budget", str(args.time_budget)],
            capture_output=True, text=True,
        )
        elapsed = time.time() - ft
        if proc.returncode != 0:
            print(f"  {name}: FAILED rc={proc.returncode}\nSTDERR (tail):\n{proc.stderr[-1500:]}")
            rows.append(dict(fold=name, status="error", full=float("nan"), private=float("nan"),
                             pool_size=0, blend_oof=None, sec=round(elapsed, 1)))
            continue

        summary = {}
        try:
            summary = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception as e:
            print(f"  {name}: could not parse summary JSON ({e}); stdout tail={proc.stdout[-500:]}")

        if not os.path.exists(out_csv):
            print(f"  {name}: NO SUBMISSION FILE WRITTEN (rc=0 but missing {out_csv})")
            rows.append(dict(fold=name, status="no_output", full=float("nan"), private=float("nan"),
                             pool_size=summary.get("pool_size"), blend_oof=summary.get("blend_oof_auc"),
                             sec=round(elapsed, 1)))
            continue

        sample = pd.read_csv(os.path.join(d, "sample_submission.csv"))
        id_col, target = sample.columns[0], sample.columns[1]
        pred_df = pd.read_csv(out_csv)
        au = true_aucs(d, id_col, target, pred_df)

        rows.append(dict(fold=name, status="ok", full=round(au["full"], 4),
                         private=round(au.get("private", float("nan")), 4),
                         pool_size=summary.get("pool_size"), blend_oof=summary.get("blend_oof_auc"),
                         sec=round(elapsed, 1)))
        print(f"  {name}: full={au['full']:.4f} priv={au.get('private', float('nan')):.4f} "
              f"pool={summary.get('pool_size')} oof={summary.get('blend_oof_auc')} ({elapsed:.1f}s)",
              flush=True)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("=" * 100)
    print(df.to_string(index=False))
    print("-" * 100)
    ok = df[df.status == "ok"]
    n_err = len(df) - len(ok)
    if len(ok):
        print(f"MEAN full={ok.full.mean():.4f} private={ok.private.mean():.4f}  "
              f"(ok={len(ok)}/{len(df)}, errors/no-output={n_err})  time={time.time()-t0:.0f}s")
    else:
        print(f"ALL {len(df)} FOLDS FAILED — see errors above.")
    print("(prior deployed tuned blend = 0.8025 full / 0.8037 private ; LB = 0.809)")


if __name__ == "__main__":
    main()
