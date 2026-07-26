"""Non-GBM model family test (Phase 1b, bigger-lever exploration): does adding a small MLP
(PyTorch, CPU) as a structurally different blend member improve the greedy blend, by having
errors less correlated with the 3 GBM families?

NNs can't handle NaN or unscaled features the way GBMs do, so this uses NN-specific
preprocessing (impute + standardize num/ord columns, add per-column missingness indicators)
kept SEPARATE from the GBMs' NaN-preserving one-hot/ordinal design matrix -- this is necessary
preprocessing for the NN to function, not an attempt at new engineered signal (which was
already tested and found harmful in dev/feature_eng.py).

Strong regularization (dropout, weight decay, early stopping) guards against overfitting on
the small 500-row folds.

Usage: python nn_test.py [--folds all] [--splits 5]
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import pipeline as P

BASE = os.path.join(os.path.dirname(__file__), "..", "data")
SEED = 42


def preprocess_for_nn(train, test, feat_cols, kinds):
    """NaN-free, standardized design matrix for a neural net. num/ord columns: impute with
    train median, standardize with train mean/std, add a missingness indicator. cat columns:
    one-hot (dummy_na=True), already NaN-free and roughly scaled in [0,1]."""
    num_ord_cols = [c for c in feat_cols if kinds[c] in ("num", "ord")]
    cat_cols = [c for c in feat_cols if kinds[c] == "cat"]
    n_tr = len(train)
    combined_raw = pd.concat([train[feat_cols], test[feat_cols]], axis=0, ignore_index=True)

    parts = []
    if num_ord_cols:
        vals = pd.DataFrame(index=combined_raw.index)
        miss = pd.DataFrame(index=combined_raw.index)
        for c in num_ord_cols:
            if kinds[c] == "ord":
                v = pd.to_numeric(combined_raw[c].astype("string").str.extract(r"(\d+)", expand=False),
                                  errors="coerce")
            else:
                v = combined_raw[c].astype(np.float64)
            train_v = v.iloc[:n_tr]
            median = train_v.median()
            std = train_v.std()
            std = std if std and std > 1e-6 else 1.0
            mean = train_v.mean()
            miss[f"{c}_isna"] = v.isna().astype(np.float32)
            filled = v.fillna(median)
            vals[c] = ((filled - mean) / std).astype(np.float32)
        parts.append(vals)
        parts.append(miss)
    if cat_cols:
        parts.append(pd.get_dummies(combined_raw[cat_cols].astype("object"),
                                    columns=cat_cols, dummy_na=True, dtype=np.float32))
    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=combined_raw.index)
    return X.iloc[:n_tr].reset_index(drop=True).values.astype(np.float32), \
        X.iloc[n_tr:].reset_index(drop=True).values.astype(np.float32)


def make_mlp(input_dim: int):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 1),
    )


def fit_nn_fold(X_tr, y_tr, X_va, y_va, X_test, max_epochs=150, patience=15, seed=SEED):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)

    Xtr_t = torch.from_numpy(X_tr)
    ytr_t = torch.from_numpy(y_tr.astype(np.float32))
    Xva_t = torch.from_numpy(X_va)
    Xte_t = torch.from_numpy(X_test)

    model = make_mlp(X_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    n = len(X_tr)
    batch_size = min(256, max(32, n // 8))
    best_auc = -1.0
    best_va_pred = None
    best_te_pred = None
    no_improve = 0

    for epoch in range(max_epochs):
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
            va_logits = model(Xva_t).squeeze(-1)
            va_pred = torch.sigmoid(va_logits).numpy()
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

    return best_va_pred, best_te_pred, best_auc


def cv_run_nn(X, y, X_test, n_splits, seed=SEED):
    oof = np.zeros(len(y))
    test = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        va_pred, te_pred, _ = fit_nn_fold(X[tr], y[tr], X[va], y[va], X_test, seed=seed)
        oof[va] = va_pred
        test += te_pred / n_splits
    return oof, test, roc_auc_score(y, oof)


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


TUNED_CACHE = os.path.join(os.path.dirname(__file__), "tuned_cache")
NN_CACHE = os.path.join(os.path.dirname(__file__), "nn_cache")
BASE_VARIANTS = ["lgb_onehot_es", "cat_onehot_es", "xgb_onehot_es"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="all")
    ap.add_argument("--splits", type=int, default=None)
    ap.add_argument("--save", action="store_true", help="cache nn oof/test arrays to dev/nn_cache/")
    args = ap.parse_args()
    if args.save:
        os.makedirs(NN_CACHE, exist_ok=True)

    dirs = sorted(glob.glob(os.path.join(BASE, "train_*")))
    if args.folds != "all":
        want = set(args.folds.split(","))
        dirs = [d for d in dirs if os.path.basename(d) in want]

    rows = []
    t0 = time.time()
    for d in dirs:
        name = os.path.basename(d)
        train, test, sample, id_col, target, feat_cols = P.load_fold(d)
        classes = np.sort(train[target].unique())
        y = (train[target].to_numpy() == classes[1]).astype(int)
        kinds = P.analyze_columns(train, feat_cols)
        n_splits = args.splits if args.splits else (3 if len(train) > 20000 else 5)

        X_nn_tr, X_nn_te = preprocess_for_nn(train, test, feat_cols, kinds)
        ft = time.time()
        nn_oof, nn_test, nn_auc = cv_run_nn(X_nn_tr, y, X_nn_te, n_splits)
        t_nn = time.time() - ft
        au_nn = _true_aucs(d, id_col, target, sample, nn_test)
        if args.save:
            np.savez(os.path.join(NN_CACHE, f"{name}__mlp.npz"), oof=nn_oof, test=nn_test)

        # load cached GBM members (base tuned configs) to test blend impact
        oofs, tests, used = [], [], []
        for v in BASE_VARIANTS:
            fp = os.path.join(TUNED_CACHE, f"{name}__{v}.npz")
            if os.path.exists(fp):
                z = np.load(fp)
                oofs.append(z["oof"]); tests.append(z["test"]); used.append(v)

        w_no_nn = _greedy_weights(np.array(oofs), y)
        test_no_nn = (np.array(tests) * w_no_nn[:, None]).sum(0)
        au_no_nn = _true_aucs(d, id_col, target, sample, test_no_nn)

        oofs_with_nn = oofs + [nn_oof]
        tests_with_nn = tests + [nn_test]
        w_with_nn = _greedy_weights(np.array(oofs_with_nn), y)
        test_with_nn = (np.array(tests_with_nn) * w_with_nn[:, None]).sum(0)
        au_with_nn = _true_aucs(d, id_col, target, sample, test_with_nn)

        rows.append(dict(fold=name, nn_oof=round(nn_auc, 4), nn_full=round(au_nn["full"], 4),
                         gbm_only_full=round(au_no_nn["full"], 4),
                         with_nn_full=round(au_with_nn["full"], 4),
                         delta=round(au_with_nn["full"] - au_no_nn["full"], 4),
                         nn_weight=round(float(w_with_nn[-1]), 3), t_nn=round(t_nn, 1)))
        print(f"  {name}: nn_alone={au_nn['full']:.4f} gbm_only={au_no_nn['full']:.4f} "
              f"with_nn={au_with_nn['full']:.4f} delta={au_with_nn['full']-au_no_nn['full']:+.4f} "
              f"nn_w={w_with_nn[-1]:.2f} ({t_nn:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("=" * 100)
    print(df.to_string(index=False))
    print("-" * 100)
    print(f"GBM-only mean full={df.gbm_only_full.mean():.4f}  "
          f"WITH-NN mean full={df.with_nn_full.mean():.4f}  mean delta={df.delta.mean():+.4f}")
    print(f"NN helps on {(df.delta > 0).sum()}/{len(df)} folds. time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
