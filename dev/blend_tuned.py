"""Blend the cached tuned (early-stopping) models and compare to the deployed baseline.
Reads dev/tuned_cache/<fold>__<variant>.npz (oof,test) + scores vs solution.csv."""
import glob, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

BASE = os.path.join(os.path.dirname(__file__), "..", "data")
TC = os.path.join(os.path.dirname(__file__), "tuned_cache")
import sys
VARIANTS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["lgb_native_es", "cat_onehot_es", "xgb_onehot_es"]


def greedy_weights(oofs, y, n_iter=30):
    M = len(oofs)
    singles = [roc_auc_score(y, oofs[i]) for i in range(M)]
    picks = [int(np.argmax(singles))]; best = max(singles)
    for _ in range(n_iter):
        cur = oofs[picks].mean(axis=0)
        sc = [roc_auc_score(y, (cur*len(picks)+oofs[i])/(len(picks)+1)) for i in range(M)]
        j = int(np.argmax(sc))
        if sc[j] <= best + 1e-6: break
        picks.append(j); best = sc[j]
    return np.bincount(picks, minlength=M)/len(picks)


rows = []
for d in sorted(glob.glob(os.path.join(BASE, "train_*"))):
    name = os.path.basename(d)
    tr = pd.read_csv(os.path.join(d, "train.csv"))
    sample = pd.read_csv(os.path.join(d, "sample_submission.csv"))
    id_col, target = sample.columns[0], sample.columns[1]
    classes = np.sort(tr[target].unique())
    y = (tr[target].to_numpy() == classes[1]).astype(int)
    sol = pd.read_csv(os.path.join(d, "solution.csv"))
    keep = [id_col, target] + (["Usage"] if "Usage" in sol.columns else [])
    m = sample[[id_col]].merge(sol[keep], on=id_col, how="left")
    y_test = m[target].to_numpy(); usage = m["Usage"].to_numpy() if "Usage" in m.columns else None

    oofs, tests = [], []
    for v in VARIANTS:
        z = np.load(os.path.join(TC, f"{name}__{v}.npz"))
        oofs.append(z["oof"]); tests.append(z["test"])
    oofs, tests = np.array(oofs), np.array(tests)

    def score(pred, split):
        if usage is None or split == "full": return roc_auc_score(y_test, pred)
        mask = usage == split.capitalize(); return roc_auc_score(y_test[mask], pred[mask])

    w = greedy_weights(oofs, y)
    gt = (tests * w[:, None]).sum(0)
    eq = tests.mean(0)
    singles = {v: score(tests[i], "full") for i, v in enumerate(VARIANTS)}
    best_single = max(singles.values())
    rows.append(dict(fold=name, greedy_full=score(gt, "full"), greedy_priv=score(gt, "private"),
                     equal_full=score(eq, "full"), best_single_full=best_single,
                     w=",".join(f"{VARIANTS[i][:3]}:{w[i]:.2f}" for i in range(len(VARIANTS)) if w[i] > 0)))

df = pd.DataFrame(rows)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
print(df.round(4).to_string(index=False))
print("-" * 90)
print(f"TUNED greedy   full={df.greedy_full.mean():.4f}  private={df.greedy_priv.mean():.4f}")
print(f"TUNED equal    full={df.equal_full.mean():.4f}")
print(f"TUNED best-single-per-fold full={df.best_single_full.mean():.4f}")
print(f"(deployed untuned greedy blend = 0.7982 full / 0.7995 private ; LB = 0.805)")
