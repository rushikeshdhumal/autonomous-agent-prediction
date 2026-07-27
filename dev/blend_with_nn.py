"""Does the NN's gain survive when blended with the FULL GBM pool (3 base + 6 HPO shortlist
= 9 members), not just the 3 base models tested in nn_test.py? Uses cached OOF/test arrays
from dev/tuned_cache/, dev/hpo_cache/, and dev/nn_cache/ -- no retraining needed.

Usage: python blend_with_nn.py
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import pipeline as P

BASE = os.path.join(os.path.dirname(__file__), "..", "data")
TUNED_CACHE = os.path.join(os.path.dirname(__file__), "tuned_cache")
HPO_CACHE = os.path.join(os.path.dirname(__file__), "hpo_cache")
NN_CACHE = os.path.join(os.path.dirname(__file__), "nn_cache")

BASE_VARIANTS = ["lgb_onehot_es", "cat_onehot_es", "xgb_onehot_es"]
HPO_VARIANTS = [f"{m}_hpo{i}" for m in ("lightgbm", "xgboost", "catboost") for i in range(2)]
GBM_VARIANTS = BASE_VARIANTS + HPO_VARIANTS


def _load(cache_dir, name, variant):
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


def main():
    rows = []
    for d in sorted(glob.glob(os.path.join(BASE, "train_*"))):
        name = os.path.basename(d)
        train, test, sample, id_col, target, feat_cols = P.load_fold(d)
        classes = np.sort(train[target].unique())
        y = (train[target].to_numpy() == classes[1]).astype(int)

        gbm_oofs, gbm_tests, used = [], [], []
        for v in GBM_VARIANTS:
            cache_dir = TUNED_CACHE if v in BASE_VARIANTS else HPO_CACHE
            r = _load(cache_dir, name, v)
            if r is None:
                continue
            gbm_oofs.append(r[0]); gbm_tests.append(r[1]); used.append(v)

        # GBM-pool-only blend (9 members)
        w_gbm = _greedy_weights(np.array(gbm_oofs), y)
        test_gbm = (np.array(gbm_tests) * w_gbm[:, None]).sum(0)
        au_gbm = _true_aucs(d, id_col, target, sample, test_gbm)

        # + NN
        nn = _load(NN_CACHE, name, "mlp")
        if nn is None:
            rows.append(dict(fold=name, gbm_pool_full=round(au_gbm["full"], 4),
                             with_nn_full=None, delta=None, nn_weight=None))
            print(f"  {name}: NO NN CACHE FOUND, skipping")
            continue
        oofs_all = gbm_oofs + [nn[0]]
        tests_all = gbm_tests + [nn[1]]
        w_all = _greedy_weights(np.array(oofs_all), y)
        test_all = (np.array(tests_all) * w_all[:, None]).sum(0)
        au_all = _true_aucs(d, id_col, target, sample, test_all)

        rows.append(dict(fold=name, gbm_pool_full=round(au_gbm["full"], 4),
                         with_nn_full=round(au_all["full"], 4),
                         delta=round(au_all["full"] - au_gbm["full"], 4),
                         nn_weight=round(float(w_all[-1]), 3)))
        print(f"  {name}: gbm_pool={au_gbm['full']:.4f} with_nn={au_all['full']:.4f} "
              f"delta={au_all['full']-au_gbm['full']:+.4f} nn_w={w_all[-1]:.2f}", flush=True)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("=" * 100)
    print(df.to_string(index=False))
    valid = df.dropna()
    print("-" * 100)
    print(f"GBM-9-pool mean full={valid.gbm_pool_full.mean():.4f}  "
          f"WITH-NN mean full={valid.with_nn_full.mean():.4f}  mean delta={valid.delta.mean():+.4f}")
    print(f"(reference: deployed 3-GBM baseline = 0.8025 full ; anytime-pipeline 9-pool = 0.8031/0.8033)")


if __name__ == "__main__":
    main()
