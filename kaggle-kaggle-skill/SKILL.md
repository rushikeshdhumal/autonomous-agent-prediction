---
name: kaggle-kaggle-competitor
description: >
  Guides competitors in designing, authoring, validating, evaluating, debugging, and
  submitting autonomous ML agents to kaggle-kaggle competitions.
  Make sure to use this skill whenever the user mentions kaggle-kaggle, autonomous ML agents,
  building an agent submission, running local evaluation, debugging agent traces, or submitting
  to Kaggle, even if they don't explicitly ask for the competitor guide.
  Don't use for building the evaluation harness itself or for non-kaggle-kaggle competitions.
---

# Kaggle-in-Kaggle Competitor Guide

You are helping a competitor participate in a **Kaggle-in-Kaggle (`kaggle-kaggle`) Autonomous ML Agent Competition**. In these competitions, participants engineer an autonomous AI agent that explores datasets, trains models, submits predictions, and selects its best submission — all without human intervention during evaluation.

Your role is to assist the competitor with structural design, sandbox compliance, validation, evaluation, trace debugging, and submission mechanics. Stay unopinionated about ML modeling strategies, feature engineering, or hyperparameter choices — focus on the mechanics that determine whether a submission runs correctly and doesn't get rejected.

---

## Critical Rules

These constraints are non-negotiable. Violating any of them causes the submission to fail validation or be rejected by the evaluation harness.

1. **`agent.yaml` must be at the archive root.** The harness expects it at the top level of the submission zip — not in a subdirectory.
2. **No symlinks anywhere in the archive.** The validator rejects them outright.
3. **No path traversal (`../`) in `!include` or `config_path`.** All references must resolve within the submission directory, because the sandbox blocks escape attempts.
4. **No `importlib` or dynamic Python imports in YAML configs.** Only the closed set of registered tools and models is available — this is how the sandbox prevents arbitrary code execution.
5. **Maximum 10 bundled skills per submission.** Each skill must have a `SKILL.md` manifest.
6. **Models must exist in the competition's `models.yaml`.** The compiler resolves model aliases against the competition's registry. Run `validate_submission.py` to verify before submitting.
7. **The agent runs in a network-isolated sandbox.** No package installation, no internet access, no GPUs unless the harness explicitly provides them. All dependencies must come from the pre-installed data science container.
8. **Budget limits are hard limits.** Exceeding any budget dimension (time, tool calls, submissions, USD spend) terminates the session immediately — there is no grace period. The agent should call `get_status()` periodically to pace itself.
9. **Never edit `sample_submission/`.** E.g., `sample_submission/` is an immutable reference template. Always create a fresh experiment directory under `submissions/<experiment_name>/` for new submissions or iterations.

---

## Interaction Model

The competitor drives the workflow. Help with whichever step they are focused on. You may suggest the natural next step when the current one is complete, but **do not execute further steps unless the competitor asks you to**. The competitor may want to iterate within a single step, pause, or do something else entirely.

---

## Environment Setup

Before assisting with validation, local evaluation, or Kaggle submission, verify that the necessary prerequisites (virtual environment, container runtime, API keys, Kaggle CLI) are configured. If setup or verification is required, consult `resources/environment_setup.md`.

---

## Workflow Reference

The typical progression through a submission lifecycle is:

```
Design → Validate → Evaluate Locally → Analyze Traces → Submit to Kaggle
```

### Design & Architecture

> [!IMPORTANT]
> **IMMUTABLE TEMPLATE & FRESH EXPERIMENTS**: The `sample_submission/` directory is an immutable reference template. **NEVER edit or run evaluations directly against `sample_submission/`**. For every new experiment or iteration, you MUST create a fresh, dedicated experiment directory (e.g., `submissions/01_baseline/`, `submissions/02_improved_prompts/`) by copying `sample_submission/` or branching from a previous experiment. This guarantees strict reproducibility and prevents overwriting previous work.

Help the competitor structure their submission archive and write valid YAML configurations using a structured experiment hierarchy. The archive must follow this encapsulated layout:

```
submissions/
└── 01_baseline/
    ├── agent/                   # Root config and agent definition
    │   ├── agent.yaml           # MANDATORY at agent root
    │   ├── prompts/
    │   │   └── system.md        # Modular system prompts
    │   ├── tools/
    │   │   └── sub_agent.yaml   # Sub-agent tool wrappers
    │   └── skills/
    │       └── feature_engineer/ # Custom bundled skills
    │           ├── SKILL.md
    │           └── scripts/
    │               └── run_fe.py
    ├── output/                  # Evaluation traces and logs specific to this run
    └── submission.zip           # Packaged archive for Kaggle
```

For YAML syntax, agent classes (`LlmAgent`, `SequentialAgent`, `ParallelAgent`, `LoopAgent`), `!include` rules, generation config constraints, and sub-agent/tool references, consult `resources/agent_config.md`.

For authoring bundled skills (the `SKILL.md` manifest, script execution environment, sandbox constraints), consult `resources/agent_skills.md`.

For the 6 built-in tools available to the competitor's agent (`run_command`, `write_file`, `edit_file`, `submit_predictions`, `select_submission`, `get_status`) — their signatures, return formats, and error handling — consult `resources/agent_tools.md`.

For system prompt authoring, dynamic state injection placeholders (`{problem_description}`, `{metric_name}`, `{max_budget_usd}`, etc.), and prompt modularization, consult `resources/agent_instructions.md`.

### Validate, Evaluate, Analyze & Submit

To eliminate root directory clutter and ensure strict reproducibility, all execution commands should target the encapsulated experiment directory (`submissions/<experiment_name>/`). 

Consult the **Quick Reference** table below for the exact CLI commands to validate agent definitions, run local evaluation (with dynamic trace routing to the experiment's `output/` subdirectory), parse execution traces, package the archive, and submit to Kaggle.

---

## Quick Reference

| Task | Command |
| :--- | :--- |
| **Validate submission** | `uv run python validate_submission.py --agent-dir submissions/01_baseline/agent` |
| **Run local eval** | `uv run python run_local_eval.py --submission-dir submissions/01_baseline/agent --dataset train_01 --metric roc_auc` |
| **Parse trace** | `uv run python scripts/parse_eval_trace.py --experiment-dir submissions/01_baseline` |
| **Package archive** | `(cd submissions/01_baseline/agent && zip -r ../submission.zip .)` |
| **Submit to Kaggle** | `kaggle competitions submit <slug> -f submissions/01_baseline/submission.zip -m "<message>"` |
| **Check submissions** | `kaggle competitions submissions <slug>` |
| **Check leaderboard** | `kaggle competitions leaderboard <slug> -s` |
| **Check agent budget** | Agent calls `get_status()` — remind competitor to include this in their agent's strategy |

---

## Resources & Scripts Manifest

### Reference Documents
Read these on-demand when the competitor needs detailed syntax, API, or setup information:

- `resources/environment_setup.md` — One-time prerequisite configuration guide: venv, container runtime, LLM API keys, Kaggle CLI
- `resources/agent_config.md` — YAML configuration reference: agent classes, field tables, `!include` rules, generation constraints, sub-agent and tool references
- `resources/agent_skills.md` — Bundling custom skills: `SKILL.md` manifests, script execution environment, sandbox limits
- `resources/agent_tools.md` — Built-in tool reference: signatures, return formats, error codes for all 6 competition tools
- `resources/agent_instructions.md` — System instruction reference: dynamic state injection placeholders, prompt modularization, budgeting placeholders

### Project Root Scripts
These scripts exist directly at the root of the competitor's project repository (`starter_kit/`) and must be run from there:

- `validate_submission.py` — Pre-flight linter for YAML syntax, `!include` resolution, and ADK compilation
- `run_local_eval.py` — Local evaluation harness

*(Note: Fallback/reference copies of these scripts are also bundled inside the system skill directory's `scripts/` folder for project seeding).*

### Skill-Bundled Helper Scripts
This standalone helper script is bundled inside the skill's `scripts/` directory:

- `parse_eval_trace.py` — Trace log parser for debugging agent behavior
