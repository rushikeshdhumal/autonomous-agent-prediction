# Autonomous Agent Prediction (Beta) — Project Documentation

This folder documents both the **competition** and **our solution** in depth. If you read it
top to bottom you will understand what the competition is, how it is evaluated, how our agent and
modeling pipeline work, why we made the choices we did, and how to reproduce and extend everything.

> **One-line summary.** This is a *"Kaggle-in-Kaggle"* competition: instead of uploading a CSV of
> predictions, we upload a small **autonomous AI agent** that Kaggle runs — with no human in the
> loop — inside a locked-down sandbox, where it must explore data, train models, and submit
> predictions for **16 tabular binary-classification datasets**, scored on **ROC AUC**.

---

## Current status (living summary)

| Item | Value |
| :--- | :--- |
| Competition | [Autonomous Agent Prediction (Beta)](https://www.kaggle.com/competitions/autonomous-agent-prediction-beta) |
| Deadline | **2026-08-06 23:59 UTC** |
| Metric | ROC AUC (higher is better) |
| Our approach | Thin LLM agent + one deterministic **`auto-ml` skill** — anytime member-pool blend of LightGBM+XGBoost+CatBoost+a small PyTorch MLP |
| Offline mean AUC (16 folds) | **0.8049 full / 0.8056 private** (current deployed pipeline) |
| Kaggle submission history | v1 0.805 → v2 0.809 → **v3 0.815** — calibration with offline holds across all three |
| Leaderboard top-10 cluster | 0.824–0.830 (gap to #1 now **0.015**) |
| Agent LLM | `gemini-3.1-flash-lite` (free tier for local dev) |
| Cost per fold at eval | ~$0.01, ~6 LLM calls (runs on Kaggle's budget, not ours) |

The single most important empirical result so far: **our offline score predicts the leaderboard
direction reliably, across three submissions** (v1 ~0.798→0.805, v2 +0.0043 offline→+0.004 LB, v3
+0.0024 offline→+0.006 LB — the real gain even exceeded the offline estimate). That lets us iterate
the model mostly offline — for free, with no LLM and no submission slots — and submit once we have
a *measured* improvement. The single biggest lever found was adding a structurally different model
family (a small MLP) alongside the GBMs — more GBM tuning alone had saturated. See
[08-results-and-roadmap.md](08-results-and-roadmap.md) and [`GOTCHAS.md`](../GOTCHAS.md).

---

## How the whole thing fits together

```mermaid
flowchart LR
    subgraph our["What WE build (this repo)"]
        skill["auto-ml skill<br/>(deterministic GBM pipeline)"]
        agent["thin LlmAgent<br/>(agent.yaml + system prompt)"]
        agent --> skill
    end
    zip["submission.zip"]
    our --> zip
    zip -->|kaggle submit| kaggle["Kaggle backend"]
    subgraph eval["Kaggle runs it autonomously, per dataset"]
        compiler["adk-submission<br/>compiles YAML → ADK agent"]
        sandbox["Docker sandbox<br/>gcr.io/kaggle-images/python<br/>train.csv / test.csv"]
        lb["Scoring engine<br/>public + private ROC AUC"]
        compiler --> sandbox --> lb
    end
    kaggle --> eval
    lb --> score["Aggregate leaderboard score<br/>(mean AUC across datasets)"]
```

---

## Reading order

| # | Document | What it covers |
| :-- | :--- | :--- |
| 1 | [01-competition-overview.md](01-competition-overview.md) | What "Kaggle-in-Kaggle" is, what you submit, the rules, budgets |
| 2 | [02-the-data.md](02-the-data.md) | The 16 datasets: schema, encodings, class balance, our diagnostics |
| 3 | [03-evaluation-harness.md](03-evaluation-harness.md) | How the harness runs your agent: tools, scoring, skill execution internals |
| 4 | [04-strategy-and-decisions.md](04-strategy-and-decisions.md) | The strategic reads and the decisions we committed to |
| 5 | [05-modeling-pipeline.md](05-modeling-pipeline.md) | The ML pipeline: encoding, models, cross-validation, blending, experiments |
| 6 | [06-agent-architecture.md](06-agent-architecture.md) | The agent: `agent.yaml`, the skill, the system prompt, the thin-agent design |
| 7 | [07-development-workflow.md](07-development-workflow.md) | Offline dev loop, local eval, submission, environment setup, reproduction |
| 8 | [08-results-and-roadmap.md](08-results-and-roadmap.md) | Results, calibration, current state, and what's next |

---

## Repository map

```
autonomous-agent-prediction/
├── docs/                          # ← you are here
├── KAGGLE_COMPETITION_PLAN.md     # original planning doc (competition mechanics + ported knowledge)
├── data/train_01 … train_16/      # the 16 datasets (train/test/sample_submission/solution/DATA.md)
├── models.yaml                    # competition LLM registry + pricing
├── run_local_eval.py              # local evaluation harness (runs the agent in Docker)
├── validate_submission.py         # pre-flight linter (compiles the agent, no Docker)
├── wheels/                        # adk_submission + kaggle_kaggle wheels
├── sample_submission/             # trivial baseline (immutable reference — never edit)
├── dev/                           # OUR offline modeling dev loop (no LLM/Docker)
│   ├── pipeline.py                #   schema-adaptive pipeline (library form)
│   ├── score_all_folds.py         #   run + score all 16 folds vs solution.csv
│   ├── experiment.py              #   cache per-model OOF/test arrays, search blends
│   ├── tune.py                    #   early-stopping tuning experiments (v2 lever)
│   ├── hpo.py                     #   offline random-search HPO + diverse shortlist selection
│   ├── eval_anytime.py            #   runs the ACTUAL deployed skill script, validates end-to-end
│   ├── stacking_test.py           #   stacking meta-learner experiment (rejected)
│   ├── feature_eng.py             #   feature engineering experiment (rejected — hurts)
│   ├── nn_test.py                 #   MLP blend member experiment (adopted — v3 lever)
│   └── blend_with_nn.py           #   validates MLP gain survives combination with full GBM pool
├── GOTCHAS.md                     # bugs found + fixed this project, in STAR format
└── submissions/
    └── 01_gbm_blend/
        ├── agent/                 # THE SUBMISSION (compiled by adk-submission)
        │   ├── agent.yaml         #   agent definition (model, tools, skill)
        │   ├── prompts/system.md  #   system prompt (thin orchestration)
        │   ├── configs/sampling.yaml
        │   └── skills/auto-ml/    #   the bundled deterministic pipeline
        │       ├── SKILL.md
        │       └── scripts/run_pipeline.py
        ├── output/                # eval traces
        └── submission.zip         # packaged archive uploaded to Kaggle
```
