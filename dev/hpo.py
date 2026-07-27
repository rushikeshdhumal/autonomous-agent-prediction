"""Offline random-search HPO (Phase 1b, step 1).

Searches each booster's hyperparameters via random sampling, evaluated on a representative
subset of folds (spanning small/large, categorical/numeric/mixed schemas) using reduced-cost
early stopping for speed. Ranks trials by mean OOF AUC, then greedily selects a DIVERSE
shortlist per model (trials far apart in normalized hyperparam space), so the blend gets
genuinely different members rather than near-duplicates.

Winning shortlist configs are then re-validated at FULL fidelity (matching the deployed
skill's ES_MAX_ITERS=3000 / ES_ROUNDS=150) across all 16 folds, cached to dev/hpo_cache/,
and combined with the existing tuned baseline (dev/tuned_cache/) via greedy blending to
measure the actual incremental gain — the number that matters.

Usage:
    python hpo.py search   --model lightgbm --trials 15   # search + shortlist (prints + saves JSON)
    python hpo.py validate                                 # full-fidelity validate the saved shortlist
    python hpo.py blend                                    # greedy-blend shortlist + baseline, compare to 0.8025
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import pipeline as P

warnings.filterwarnings("ignore")

BASE = os.path.join(os.path.dirname(__file__), "..", "data")
HPO_CACHE = os.path.join(os.path.dirname(__file__), "hpo_cache")
TUNED_CACHE = os.path.join(os.path.dirname(__file__), "tuned_cache")
SHORTLIST_PATH = os.path.join(os.path.dirname(__file__), "hpo_shortlist.json")
os.makedirs(HPO_CACHE, exist_ok=True)

# Representative folds for the search phase: small (13, 05), all-categorical (08),
# all-numeric (16), mixed (01). Deliberately excludes the largest folds (train_11/12) to
# keep search trials fast; winning configs are validated on ALL 16 folds afterward.
SEARCH_FOLDS = ["train_01", "train_05", "train_08", "train_13", "train_16"]
SEARCH_SPLITS = 3
SEARCH_MAX_ITERS = 1000
SEARCH_ES_ROUNDS = 75

FULL_SPLITS = 5
FULL_MAX_ITERS = 3000
FULL_ES_ROUNDS = 150

MODELS = ("lightgbm", "xgboost", "catboost")
SEED = 42


# --------------------------------------------------------------------- hyperparam spaces
def sample_params(model: str, rng: np.random.Generator) -> dict:
    """Draw one random hyperparameter set for `model`. Ranges centered on what already
    validated well (low LR + early stopping) but wide enough to find diverse configs."""
    lr = float(np.exp(rng.uniform(np.log(0.008), np.log(0.06))))
    if model == "lightgbm":
        return dict(
            learning_rate=round(lr, 4),
            num_leaves=int(rng.integers(15, 128)),
            min_child_samples=int(rng.integers(5, 60)),
            subsample=round(float(rng.uniform(0.6, 1.0)), 3),
            colsample_bytree=round(float(rng.uniform(0.6, 1.0)), 3),
            reg_lambda=round(float(np.exp(rng.uniform(np.log(0.1), np.log(10)))), 4),
        )
    if model == "xgboost":
        return dict(
            learning_rate=round(lr, 4),
            max_depth=int(rng.integers(3, 10)),
            min_child_weight=int(rng.integers(1, 21)),
            subsample=round(float(rng.uniform(0.6, 1.0)), 3),
            colsample_bytree=round(float(rng.uniform(0.6, 1.0)), 3),
            reg_lambda=round(float(np.exp(rng.uniform(np.log(0.1), np.log(10)))), 4),
        )
    if model == "catboost":
        return dict(
            learning_rate=round(lr, 4),
            depth=int(rng.integers(4, 10)),
            l2_leaf_reg=round(float(np.exp(rng.uniform(np.log(1), np.log(15)))), 4),
        )
    raise ValueError(model)


PARAM_RANGES = {
    "lightgbm": dict(learning_rate=(0.008, 0.06), num_leaves=(15, 127), min_child_samples=(5, 59),
                     subsample=(0.6, 1.0), colsample_bytree=(0.6, 1.0), reg_lambda=(0.1, 10)),
    "xgboost": dict(learning_rate=(0.008, 0.06), max_depth=(3, 9), min_child_weight=(1, 20),
                    subsample=(0.6, 1.0), colsample_bytree=(0.6, 1.0), reg_lambda=(0.1, 10)),
    "catboost": dict(learning_rate=(0.008, 0.06), depth=(4, 9), l2_leaf_reg=(1, 15)),
}


def normalize_params(model: str, params: dict) -> np.ndarray:
    """Map a param dict to [0,1]^d using PARAM_RANGES, for diversity distance calc."""
    ranges = PARAM_RANGES[model]
    return np.array([
        (params[k] - lo) / (hi - lo) if hi > lo else 0.0
        for k, (lo, hi) in ranges.items()
    ])


# --------------------------------------------------------------------------- model fit
def fit_predict(model: str, params: dict, X_tr, y_tr, X_va, y_va, X_test,
                max_iters: int, es_rounds: int):
    if model == "lightgbm":
        from lightgbm import LGBMClassifier, early_stopping
        m = LGBMClassifier(n_estimators=max_iters, random_state=SEED, n_jobs=-1, verbosity=-1,
                           subsample_freq=1, **params)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
              callbacks=[early_stopping(es_rounds, verbose=False)])
    elif model == "xgboost":
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=max_iters, tree_method="hist", eval_metric="auc",
                          early_stopping_rounds=es_rounds, random_state=SEED, n_jobs=-1,
                          verbosity=0, **params)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    elif model == "catboost":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(iterations=max_iters, eval_metric="AUC", random_seed=SEED,
                               thread_count=-1, verbose=False, allow_writing_files=False,
                               early_stopping_rounds=es_rounds, **params)
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    else:
        raise ValueError(model)
    return m.predict_proba(X_va)[:, 1], m.predict_proba(X_test)[:, 1]


# --------------------------------------------------------------------------- search
_FOLD_DATA_CACHE: dict[str, tuple] = {}


def _fold_data(name: str):
    if name not in _FOLD_DATA_CACHE:
        d = os.path.join(BASE, name)
        train, test, sample, id_col, target, feat_cols = P.load_fold(d)
        y = train[target].to_numpy()
        X_tr, X_te, _ = P.build_matrices(train, test, feat_cols)
        _FOLD_DATA_CACHE[name] = (X_tr.values, y, X_te.values)
    return _FOLD_DATA_CACHE[name]


def score_trial(model: str, params: dict, folds: list[str], splits: int,
                max_iters: int, es_rounds: int) -> float:
    """Mean OOF AUC for one hyperparam set, averaged across `folds` (each with its own CV)."""
    fold_aucs = []
    for name in folds:
        Xv, y, Xt = _fold_data(name)
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=SEED)
        oof = np.zeros(len(y))
        for tr, va in skf.split(Xv, y):
            va_p, _ = fit_predict(model, params, Xv[tr], y[tr], Xv[va], y[va], Xt,
                                  max_iters, es_rounds)
            oof[va] = va_p
        fold_aucs.append(roc_auc_score(y, oof))
    return float(np.mean(fold_aucs))


def search(model: str, n_trials: int, folds: list[str], splits: int,
          max_iters: int, es_rounds: int, seed: int = SEED):
    rng = np.random.default_rng(seed)
    trials = []
    t0 = time.time()
    for i in range(n_trials):
        params = sample_params(model, rng)
        tt = time.time()
        score = score_trial(model, params, folds, splits, max_iters, es_rounds)
        trials.append({"params": params, "score": score})
        print(f"  [{model}] trial {i+1}/{n_trials}: score={score:.4f} "
              f"params={params} ({time.time()-tt:.1f}s)", flush=True)
    trials.sort(key=lambda t: -t["score"])
    print(f"{model}: search done in {time.time()-t0:.0f}s. Best={trials[0]['score']:.4f}")
    return trials


def select_diverse(model: str, trials: list[dict], k: int, min_dist: float = 0.25,
                   score_floor: float = 0.005) -> list[dict]:
    """Greedily pick up to k trials: always take the best, then repeatedly take the
    next-best trial whose normalized-param distance to all picks exceeds min_dist, as
    long as its score is within score_floor of the best (avoid diverse-but-bad configs)."""
    if not trials:
        return []
    best_score = trials[0]["score"]
    picks = [trials[0]]
    vecs = [normalize_params(model, trials[0]["params"])]
    for t in trials[1:]:
        if len(picks) >= k:
            break
        if best_score - t["score"] > score_floor:
            continue
        v = normalize_params(model, t["params"])
        if min(np.linalg.norm(v - pv) for pv in vecs) >= min_dist:
            picks.append(t)
            vecs.append(v)
    return picks


# --------------------------------------------------------------------------- CLI actions
def cmd_search(args):
    folds = args.folds.split(",") if args.folds else SEARCH_FOLDS
    shortlist = {}
    if os.path.exists(SHORTLIST_PATH):
        shortlist = json.load(open(SHORTLIST_PATH))
    models = [args.model] if args.model != "all" else list(MODELS)
    for model in models:
        trials = search(model, args.trials, folds, args.splits, args.max_iters, args.es_rounds)
        picks = select_diverse(model, trials, args.keep)
        print(f"\n{model}: selected {len(picks)} diverse configs (extra, beyond baseline):")
        for p in picks:
            print(f"    score={p['score']:.4f}  {p['params']}")
        shortlist[model] = [p["params"] for p in picks]
    json.dump(shortlist, open(SHORTLIST_PATH, "w"), indent=2)
    print(f"\nSaved shortlist to {SHORTLIST_PATH}")


def cmd_validate(args):
    """Re-run shortlist configs at FULL fidelity across all 16 folds; cache OOF/test."""
    if not os.path.exists(SHORTLIST_PATH):
        raise SystemExit(f"No shortlist at {SHORTLIST_PATH}; run `search` first.")
    shortlist = json.load(open(SHORTLIST_PATH))
    dirs = sorted(glob.glob(os.path.join(BASE, "train_*")))
    for d in dirs:
        name = os.path.basename(d)
        train, test, sample, id_col, target, feat_cols = P.load_fold(d)
        y = train[target].to_numpy()
        X_tr, X_te, _ = P.build_matrices(train, test, feat_cols)
        Xv, Xt = X_tr.values, X_te.values
        for model, configs in shortlist.items():
            for idx, params in enumerate(configs):
                key = f"{model}_hpo{idx}"
                fp = os.path.join(HPO_CACHE, f"{name}__{key}.npz")
                if os.path.exists(fp):
                    continue
                skf = StratifiedKFold(n_splits=FULL_SPLITS, shuffle=True, random_state=SEED)
                oof = np.zeros(len(y)); test_p = np.zeros(len(Xt))
                ft = time.time()
                for tr, va in skf.split(Xv, y):
                    va_p, test_pred = fit_predict(model, params, Xv[tr], y[tr], Xv[va], y[va], Xt,
                                                  FULL_MAX_ITERS, FULL_ES_ROUNDS)
                    oof[va] = va_p
                    test_p += test_pred / FULL_SPLITS
                auc = roc_auc_score(y, oof)
                np.savez(fp, oof=oof, test=test_p)
                print(f"  {name}/{key}: oof={auc:.4f} ({time.time()-ft:.1f}s)", flush=True)


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


def cmd_blend(args):
    """Greedy-blend baseline (tuned_cache) + HPO shortlist (hpo_cache), compare to 0.8025."""
    shortlist = json.load(open(SHORTLIST_PATH)) if os.path.exists(SHORTLIST_PATH) else {}
    baseline_variants = ["lgb_onehot_es", "cat_onehot_es", "xgb_onehot_es"]
    hpo_variants = [f"{m}_hpo{i}" for m, cfgs in shortlist.items() for i in range(len(cfgs))]
    all_variants = baseline_variants + hpo_variants

    rows = []
    for d in sorted(glob.glob(os.path.join(BASE, "train_*"))):
        name = os.path.basename(d)
        train, test, sample, id_col, target, feat_cols = P.load_fold(d)
        classes = np.sort(train[target].unique())
        y = (train[target].to_numpy() == classes[1]).astype(int)

        oofs, tests, used = [], [], []
        for v in all_variants:
            cache_dir = TUNED_CACHE if v in baseline_variants else HPO_CACHE
            fp = os.path.join(cache_dir, f"{name}__{v}.npz")
            if not os.path.exists(fp):
                continue
            z = np.load(fp)
            oofs.append(z["oof"]); tests.append(z["test"]); used.append(v)
        oofs, tests = np.array(oofs), np.array(tests)

        w = _greedy_weights(oofs, y)
        blend_test = (tests * w[:, None]).sum(0)
        au = _true_aucs(d, id_col, target, sample, blend_test)
        rows.append(dict(fold=name, full=au["full"], private=au.get("private", float("nan")),
                         w=",".join(f"{used[i]}:{w[i]:.2f}" for i in range(len(used)) if w[i] > 0)))

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 80)
    print(df.round(4).to_string(index=False))
    print("-" * 90)
    print(f"BASELINE+HPO greedy blend: mean full={df.full.mean():.4f} private={df.private.mean():.4f}")
    print("(baseline-only tuned blend = 0.8025 full / 0.8037 private)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("search")
    sp.add_argument("--model", default="all", choices=[*MODELS, "all"])
    sp.add_argument("--trials", type=int, default=15)
    sp.add_argument("--keep", type=int, default=2, help="extra diverse configs to keep per model")
    sp.add_argument("--folds", default=None, help="comma list; default = SEARCH_FOLDS")
    sp.add_argument("--splits", type=int, default=SEARCH_SPLITS)
    sp.add_argument("--max-iters", type=int, default=SEARCH_MAX_ITERS)
    sp.add_argument("--es-rounds", type=int, default=SEARCH_ES_ROUNDS)

    sub.add_parser("validate")
    sub.add_parser("blend")

    args = ap.parse_args()
    if args.cmd == "search":
        cmd_search(args)
    elif args.cmd == "validate":
        cmd_validate(args)
    elif args.cmd == "blend":
        cmd_blend(args)
