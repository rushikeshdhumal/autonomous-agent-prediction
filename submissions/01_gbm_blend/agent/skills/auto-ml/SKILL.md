---
name: auto-ml
description: >
  Trains a robust, schema-adaptive gradient-boosting blend (LightGBM + XGBoost + CatBoost,
  5-fold cross-validated and greedy-weighted) on the competition's train.csv and writes
  positive-class probabilities to submission.csv. Use this for any tabular binary-classification
  task in this competition — it handles mixed numeric/categorical/ordinal columns and missing
  values automatically and needs no configuration.
---

# auto_ml — one-shot tabular binary classification

This skill contains a pre-tested, deterministic pipeline that does the entire modeling job.
Prefer it over writing your own training code: it is faster, cheaper, and more reliable.

## What the pipeline does
- Auto-locates `train.csv` / `test.csv` / `sample_submission.csv` in the working directory.
- Infers each feature's type **from the data** (numeric, string-ordinal `ord_*`, string-nominal
  `cat_*`); one-hot encodes nominal categoricals, ordinal-encodes ordinals, keeps numeric with
  NaN preserved (the gradient boosters handle missing values natively).
- Trains LightGBM + XGBoost + CatBoost with 5-fold stratified CV, blends them with greedy
  (Caruana) weights chosen on out-of-fold predictions, and bags the test predictions.
- Writes `submission.csv` (`row_id,target` with **positive-class probabilities** — the ROC AUC
  metric needs probabilities, not 0/1 labels).

## Scripts

### `scripts/run_pipeline.py`
Runs the full pipeline. **No arguments are required.**

**Instructions:**
1. Run it with:
   `run_skill_script(skill_name="auto-ml", file_path="scripts/run_pipeline.py")`
2. It prints a one-line JSON summary: `cv_oof_auc_per_model`, `blend_weights`, `blend_oof_auc`,
   and the output path. A `blend_oof_auc` around 0.6–0.97 (dataset-dependent) is expected.
3. It writes `submission.csv` to the working directory. Submit it with
   `submit_predictions("submission.csv")`.

Optional arguments (only if you have a reason to change defaults):
`{"splits": 5, "models": "lightgbm,xgboost,catboost"}` passed as the `args` object.
Typical runtime is under a few minutes; it fits comfortably in the session budget.
