# Agent Skills Reference

> Table of Contents
> 1. [Archive Structure](#1-archive-structure)
> 2. [Authoring SKILL.md](#2-authoring-skillmd)
> 3. [Writing Skill Scripts](#3-writing-skill-scripts)
> 4. [Declaring Skills in agent.yaml](#4-declaring-skills-in-agentyaml)
> 5. [Limits & Security Rules](#5-limits--security-rules)
> 6. [Complete Example](#6-complete-example)

Skills let competitors bundle pre-tested Python scripts and specialized instructions directly into their submission archive. The evaluation harness automatically converts bundled skills into callable tools for the agent, saving token budget and reducing the risk of LLM-generated code errors during evaluation.

---

## 1. Archive Structure

Each skill resides in its own subdirectory under `skills/`:

```
my_submission/
├── agent.yaml
└── skills/
    └── feature_engineer/
        ├── SKILL.md             # Mandatory manifest & LLM instructions
        ├── scripts/
        │   └── generate_fe.py   # Executable Python script
        └── resources/
            └── encodings.json   # Optional static assets
```

---

## 2. Authoring `SKILL.md`

The `SKILL.md` file provides metadata (via YAML frontmatter) and serves as the instruction manual for the agent at runtime.

```markdown
---
name: feature_engineer
description: Generates advanced statistical and non-linear features from tabular datasets.
---

# Feature Engineer Skill

Use this skill to enrich training and test datasets with engineered features.

## Available Scripts

### `scripts/generate_fe.py`
Reads `train.csv` and `test.csv`, applies feature transformations, and writes `train_fe.csv` and `test_fe.csv`.

**Instructions**:
1. Ensure `train.csv` and `test.csv` are in the working directory.
2. Run this script before training models.
3. Verify output with `run_command("head -n 5 train_fe.csv")`.
```

**Key fields:**
- **`name`** (required): Must match the directory name.
- **`description`** (required): Concise summary — the LLM uses this to decide whether to engage the skill.
- **Markdown body**: Clear, step-by-step instructions specifying what scripts are available, what inputs they expect, and what outputs they produce.

---

## 3. Writing Skill Scripts

Scripts execute in the same sandboxed environment as `run_command` calls.

### Execution Environment
- **Working directory**: The Docker sandbox root, alongside `train.csv`, `test.csv`, `sample_submission.csv`, and any files from previous tool calls.
- **Pre-installed packages**: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `torch`, `tensorflow`, `scipy`, etc. (from `gcr.io/kaggle-images/python`).
- **No internet**: Scripts cannot make external API calls, download weights, or `pip install`. Bundle custom weights as model adapters instead.

### Output Communication
Whatever scripts print to `stdout`/`stderr` is returned directly to the agent's reasoning loop. Print structured summaries (e.g., `"Added 42 new features. Saved to train_fe.csv"`) rather than raw dataframes — large outputs consume context window space and inflate token costs.

---

## 4. Declaring Skills in `agent.yaml`

```yaml
name: kaggle_agent
model: strong
instruction: !include prompts/system.md
tools:
  - run_command
  - write_file
  - submit_predictions
  - select_submission
  - get_status
skills:
  - skills/feature_engineer
  - skills/model_tuner
generate_content_config:
  temperature: 0.2
```

---

## 5. Limits & Security Rules

| Rule | Limit | Notes |
| :--- | :--- | :--- |
| **Max skills** | 10 | Per submission archive |
| **Script timeout** | 3600s | Shared with global execution timeout |
| **No path traversal** | Enforced | All paths must resolve within submission root |
| **No symlinks** | Enforced | Anywhere in the archive |
| **Unique names** | Required | Skill directory names must be distinct |

---

## 6. Complete Example

### `skills/feature_engineer/SKILL.md`
```markdown
---
name: feature_engineer
description: Automated log-transform and polynomial feature synthesis for tabular datasets.
---

# Feature Engineer Skill

Use this skill immediately after dataset exploration.

## Scripts

### `scripts/run_fe.py`
Reads `train.csv` and `test.csv`, applies log1p to skewed columns, writes `train_fe.csv` and `test_fe.csv`.

**Instructions**:
1. Call `scripts/run_fe.py` before training.
2. Update training scripts to read from `train_fe.csv` and `test_fe.csv`.
```

### `skills/feature_engineer/scripts/run_fe.py`
```python
import pandas as pd
import numpy as np

print("Starting automated feature engineering...")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
initial_cols = len(train.columns)

for col in train.select_dtypes(include=[np.number]).columns:
    if col not in ["id", "target"]:
        train[f"{col}_log"] = np.log1p(train[col].clip(lower=0))
        test[f"{col}_log"] = np.log1p(test[col].clip(lower=0))

train.to_csv("train_fe.csv", index=False)
test.to_csv("test_fe.csv", index=False)

final_cols = len(train.columns)
print(f"SUCCESS: Added {final_cols - initial_cols} features. Saved to train_fe.csv, test_fe.csv.")
```
