"""Schema-adaptive tabular binary-classification pipeline for kaggle-kaggle folds.

Dependency-light (pandas, numpy, scikit-learn, lightgbm, xgboost, catboost) so the
same logic can later be bundled as an agent skill script that runs inside
gcr.io/kaggle-images/python.

IMPORTANT — faithfulness to the eval sandbox:
    The agent's sandbox contains ONLY train.csv / test.csv / sample_submission.csv.
    There is NO DATA.md at eval time, so schema (which columns are categorical) is
    INFERRED FROM THE DATA, never read from DATA.md. The pipeline below does the same.

Given the data profile (all features integer/float-coded, categoricals low-cardinality
<=8, missing values everywhere, balanced 0/1 target), the approach is:
    - infer categorical columns = low-cardinality integer-valued columns
    - one-hot encode categoricals (dummy_na to capture missingness), keep numeric as-is
      with NaN preserved (GBMs handle NaN natively)
    - StratifiedKFold OOF probabilities + bagged test probabilities
    - submit positive-class probability (metric is ROC AUC on the target column)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED = 42


# ----------------------------------------------------------------------------- schema
_ORD_RE = r"^\s*ord_?\d+\s*$"  # string ordinal like 'ord_3'


def analyze_columns(train: pd.DataFrame, feat_cols: list[str]) -> dict[str, str]:
    """Classify each feature from data alone (no DATA.md at eval time) into:
        'num' — numeric (float/int; includes integer-coded counts/ordinals, kept ordered)
        'ord' — string ordinal 'ord_<int>' -> ordinal-encode by the integer (preserve order)
        'cat' — any other string -> nominal, one-hot encoded
    """
    kinds: dict[str, str] = {}
    for c in feat_cols:
        s = train[c]
        if s.dtype.kind in "iufb":
            kinds[c] = "num"
            continue
        nn = s.dropna().astype("string")
        if len(nn) and nn.str.match(_ORD_RE, case=False).all():
            kinds[c] = "ord"
        else:
            kinds[c] = "cat"
    return kinds


def build_matrices(train: pd.DataFrame, test: pd.DataFrame, feat_cols: list[str],
                   kinds: dict[str, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Encode per column kind; align across train/test; keep numeric NaN for GBMs."""
    if kinds is None:
        kinds = analyze_columns(train, feat_cols)
    ord_cols = [c for c in feat_cols if kinds[c] == "ord"]
    cat_cols = [c for c in feat_cols if kinds[c] == "cat"]
    num_cols = [c for c in feat_cols if kinds[c] == "num"]

    n_tr = len(train)
    combined = pd.concat([train[feat_cols], test[feat_cols]], axis=0, ignore_index=True)
    parts = []
    if ord_cols:
        ordmat = pd.DataFrame(index=combined.index)
        for c in ord_cols:
            codes = combined[c].astype("string").str.extract(r"(\d+)", expand=False)
            ordmat[c] = pd.to_numeric(codes, errors="coerce")  # NaN preserved
        parts.append(ordmat.astype(np.float32))
    if cat_cols:
        dummies = pd.get_dummies(combined[cat_cols].astype("object"),
                                 columns=cat_cols, dummy_na=True, dtype=np.float32)
        parts.append(dummies)
    if num_cols:
        parts.append(combined[num_cols].astype(np.float32))
    X = pd.concat(parts, axis=1) if parts else combined.astype(np.float32)
    return X.iloc[:n_tr].reset_index(drop=True), X.iloc[n_tr:].reset_index(drop=True), kinds


# ----------------------------------------------------------------------------- models
def make_model(name: str, n_train: int):
    """Factory of regularized, NaN-tolerant classifiers with sane defaults."""
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=600, learning_rate=0.03, num_leaves=31, max_depth=-1,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=1.0, min_child_samples=max(5, min(20, n_train // 50)),
            random_state=SEED, n_jobs=-1, verbosity=-1,
        )
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=600, learning_rate=0.03, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            tree_method="hist", eval_metric="auc",
            random_state=SEED, n_jobs=-1, verbosity=0,
        )
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=600, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
            loss_function="Logloss", eval_metric="AUC",
            random_seed=SEED, thread_count=-1, verbose=False, allow_writing_files=False,
        )
    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=600, learning_rate=0.03, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=SEED,
        )
    raise ValueError(name)


# ----------------------------------------------------------------------------- CV
def cv_oof_test(X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame, model_name: str,
                n_splits: int = 5) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (oof_proba, bagged_test_proba, oof_auc) for one model via StratifiedKFold."""
    n = len(y)
    Xv = X.values
    Xt = X_test.values
    oof = np.zeros(n)
    test = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr_idx, va_idx in skf.split(Xv, y):
        model = make_model(model_name, len(tr_idx))
        model.fit(Xv[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict_proba(Xv[va_idx])[:, 1]
        test += model.predict_proba(Xt)[:, 1] / n_splits
    return oof, test, roc_auc_score(y, oof)


# ----------------------------------------------------------------------------- fold
def load_fold(data_dir: str):
    import os
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    sample = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
    id_col = sample.columns[0]
    target = sample.columns[1]
    feat_cols = [c for c in train.columns if c not in (id_col, target)]
    return train, test, sample, id_col, target, feat_cols


def run_fold(data_dir: str, models=("lightgbm",), n_splits: int = 5, max_card: int = 20):
    """Run the pipeline on one fold; return predictions + diagnostics. (max_card kept for
    CLI compatibility; no longer used — encoding is value-driven via analyze_columns.)"""
    train, test, sample, id_col, target, feat_cols = load_fold(data_dir)
    y = train[target].to_numpy()
    X_tr, X_te, kinds = build_matrices(train, test, feat_cols)
    cat_cols = [c for c in feat_cols if kinds[c] == "cat"]

    per_model_oof, per_model_test, per_model_auc = {}, {}, {}
    for m in models:
        oof, tst, auc = cv_oof_test(X_tr, y, X_te, m, n_splits=n_splits)
        per_model_oof[m] = oof
        per_model_test[m] = tst
        per_model_auc[m] = auc

    # simple equal-weight blend of member OOF/test probabilities
    blend_oof = np.mean([per_model_oof[m] for m in models], axis=0)
    blend_test = np.mean([per_model_test[m] for m in models], axis=0)
    blend_auc = roc_auc_score(y, blend_oof)

    return {
        "id_col": id_col, "target": target, "sample": sample,
        "n_train": len(train), "n_feat": len(feat_cols), "n_cat": len(cat_cols),
        "per_model_auc": per_model_auc, "blend_oof_auc": blend_auc,
        "test_pred": blend_test, "per_model_test": per_model_test,
        "cat_cols": cat_cols,
    }


def write_submission(result: dict, out_path: str):
    sub = result["sample"].copy()
    sub[result["target"]] = result["test_pred"]
    sub.to_csv(out_path, index=False)
    return out_path
