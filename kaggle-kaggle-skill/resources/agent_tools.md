# Agent Tools Reference

> Table of Contents
> 1. [Environment Overview](#1-environment-overview)
> 2. [Tool Summary](#2-tool-summary)
> 3. [run_command](#3-run_command)
> 4. [write_file](#4-write_file)
> 5. [edit_file](#5-edit_file)
> 6. [submit_predictions](#6-submit_predictions)
> 7. [select_submission](#7-select_submission)
> 8. [get_status](#8-get_status)
> 9. [Declaring Tools](#9-declaring-tools)

The agent operates inside a fully isolated evaluation sandbox with six built-in tools for dataset exploration, code execution, prediction submission, and budget management.

---

## 1. Environment Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   Competitor ADK Agent                          │
│         ┌───────────────────────┴───────────────────────┐       │
│         ▼                                               ▼       │
│ ┌───────────────────────────────┐       ┌─────────────────────┐ │
│ │     The Docker Sandbox        │       │   The Leaderboard   │ │
│ │ (run_command, write, edit)    │       │  (submit, select,   │ │
│ │  ├── train.csv / test.csv     │       │     get_status)     │ │
│ │  └── pre-installed ML libs    │       │  ├── Public LB      │ │
│ └───────────────────────────────┘       │  └── Private LB     │ │
│         ▲                               └──────────┬──────────┘ │
│         └─────────────── Budget Check ─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

- **Docker Sandbox**: Code execution in an offline Linux container (default: `gcr.io/kaggle-images/python`) pre-populated with competition data. No internet access.
- **Leaderboard**: Submission tools interact with the scoring engine for evaluation, budget enforcement, and private leaderboard selections.

---

## 2. Tool Summary

| Tool | Signature | Purpose |
| :--- | :--- | :--- |
| `run_command` | `run_command(command: str) -> str` | Execute shell commands in the sandbox. |
| `write_file` | `write_file(filepath: str, content: str) -> str` | Create or overwrite files. |
| `edit_file` | `edit_file(filepath: str, old_string: str, new_string: str, allow_multiple: bool = False) -> str` | Replace text blocks in existing files. |
| `submit_predictions` | `submit_predictions(filepath: str) -> str` | Submit predictions for public LB scoring. |
| `select_submission` | `select_submission(submission_ids: list[str]) -> str` | Select submissions for private LB scoring. |
| `get_status` | `get_status() -> str` | Check remaining budgets and submission history. |

---

## 3. `run_command`

Executes a shell command inside the Docker sandbox.

**Working directory** contains: `train.csv`, `test.csv`, `sample_submission.csv`, plus any files created by previous tool calls.

**Pre-installed libraries**: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `torch`, `tensorflow`, `scipy`, etc.

### Returns

**Success** (`status: "ok"`):
```json
{
  "status": "ok",
  "stdout": "Model trained successfully.\n",
  "exit_code": 0,
  "duration_seconds": 14.23
}
```
`stdout`/`stderr` are truncated to `max_stdout_chars` (default 5000).

**Error** (`status: "error"`):
```json
{
  "status": "error",
  "error_type": "CommandError",
  "error_message": "ModuleNotFoundError: No module named 'unknown_package'",
  "details": { "exit_code": 1, "duration_seconds": 0.45 }
}
```

---

## 4. `write_file`

Creates or overwrites a file. Parent directories are created automatically.

### Arguments
- `filepath`: Relative path within sandbox (e.g., `train.py`, `src/models/xgb.py`).
- `content`: Full file content.

### Returns
```json
{ "status": "ok", "filepath": "train.py", "size": 1420 }
```

---

## 5. `edit_file`

Replaces a contiguous text block in an existing file. More token-efficient than rewriting the entire file with `write_file`.

### Arguments
- `filepath`: Relative path to target file.
- `old_string`: Exact text to replace (flexible indentation matching).
- `new_string`: Replacement text.
- `allow_multiple` (optional, default `False`): Replace all occurrences.

### Returns
```json
{
  "status": "ok",
  "filepath": "train.py",
  "occurrences": 1,
  "strategy": "exact",
  "diff_snippet": "--- Current\n+++ Proposed\n@@ -15,3 +15,3 @@\n-model = RandomForestClassifier(n_estimators=100)\n+model = RandomForestClassifier(n_estimators=200)"
}
```

---

## 6. `submit_predictions`

Submits a CSV for public leaderboard scoring. The CSV must exactly match `sample_submission.csv` in columns, row count, and IDs.

### Validations
The scoring engine rejects submissions for:
- **Submission limit exceeded**: Agent has reached `max_submissions`.
- **File too large**: Exceeds 50 MB.
- **Column mismatch**: Columns don't match `sample_submission.csv`.
- **Row count mismatch**: Different number of rows.
- **Duplicate IDs**: ID column contains duplicates.
- **ID mismatch**: IDs don't match the expected test set.
- **File not found**: The filepath doesn't exist in the sandbox.

### Returns
```json
{
  "status": "ok",
  "submission_id": "sub_1",
  "score": 0.845123,
  "metric": "ROC AUC",
  "submission_number": 1,
  "remaining_submissions": 29,
  "best_score": 0.845123,
  "all_submissions": [
    {"id": "sub_1", "score": 0.845123}
  ]
}
```

The harness evaluates against Public, Private, and Holdout splits. Only the **Public** score is returned to the agent.

---

## 7. `select_submission`

Selects submissions for final private leaderboard scoring. If never called, the harness defaults to the best public score.

### Arguments
- `submission_ids`: List of submission ID strings (e.g., `["sub_1", "sub_3"]`).

### Validations
- Empty list → error.
- Exceeds `max_selections` → error.
- Unknown IDs → error.

### Returns
```json
{
  "status": "ok",
  "selected": [
    {"id": "sub_1", "public_score": 0.845123},
    {"id": "sub_3", "public_score": 0.861204}
  ],
  "message": "2 submission(s) selected for final scoring."
}
```

---

## 8. `get_status`

Returns real-time session status including budgets, submission history, and token consumption.

### Returns
```json
{
  "tool_calls_used": 42,
  "tool_calls_remaining": 958,
  "submissions_used": 3,
  "submissions_remaining": 27,
  "time_minutes_used": 15.45,
  "time_minutes_remaining": 44.55,
  "all_submissions": [
    {"id": "sub_1", "score": 0.812034},
    {"id": "sub_2", "score": 0.834102}
  ],
  "best_score": 0.850123,
  "metric": "ROC AUC",
  "selected_submission_ids": ["sub_2"],
  "token_budget": {
    "total_cost_usd": 0.341250,
    "remaining_usd": 9.658750,
    "max_budget_usd": 10.0,
    "total_input_tokens": 245102,
    "total_output_tokens": 18450,
    "total_tokens": 263552,
    "llm_calls": 35
  }
}
```

---

## 9. Declaring Tools

```yaml
name: kaggle_agent
model: strong
instruction: !include prompts/system.md
tools:
  - run_command
  - write_file
  - edit_file
  - submit_predictions
  - select_submission
  - get_status
generate_content_config:
  temperature: 0.2
  max_output_tokens: 8192
```
