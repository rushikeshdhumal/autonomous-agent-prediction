# Agent Instructions Reference (`agent_instructions.md`)

> Table of Contents
> 1. [Core Mechanics](#1-core-mechanics)
> 2. [Dynamic State Injection](#2-dynamic-state-injection)
> 3. [Prompt Modularization](#3-prompt-modularization)
> 4. [Instruction Structure](#4-instruction-structure)
> 5. [Complete Example](#5-complete-example)

When competing in Kaggle-in-Kaggle (`kaggle-kaggle`), your autonomous agent operates inside an isolated evaluation sandbox. It relies entirely on its system instruction to understand the competition task, manage its execution budget, orchestrate built-in tools, and execute its data science workflow.

---

## 1. Core Mechanics

In `kaggle-kaggle`, your agent is compiled using `adk-submission` from a declarative YAML configuration file (`agent.yaml`). The `instruction` field defines your agent's system prompt.

- **Persistence**: The underlying Google ADK injects this instruction as the system prompt before every LLM call. Unlike conversation turns which can be summarized or compacted away over long evaluation sessions, the system instruction remains permanently active in the context window.
- **Runtime Injection**: While the instruction template is authored statically in your submission archive, the evaluation harness dynamically injects competition-specific context and budget limits into it at runtime.

---

## 2. Dynamic State Injection

Before your agent executes, `kaggle-kaggle` populates `session.state` with granular competition context and budget constraints. You can pull this information directly into your `agent.yaml` instruction (or included markdown prompts) using standard Python format string placeholders:

| Placeholder | Description | Example Value |
| :--- | :--- | :--- |
| **`{problem_description}`** | Full markdown description of the competition task and dataset. | `"Predict loan default risk using customer financial history..."` |
| **`{metric_name}`** | The official evaluation metric used by the scoring engine. | `roc_auc_score` |
| **`{metric_direction}`** | The optimization direction for the metric. | `higher is better` (or `lower is better`) |
| **`{max_tool_calls}`** | Maximum allowed tool invocations across the session. | `150` |
| **`{max_submissions}`** | Maximum allowed public leaderboard prediction submissions. | `30` |
| **`{max_selections}`** | Maximum allowed submissions selected for the private leaderboard. | `2` |
| **`{max_exec_seconds}`** | Timeout per code execution command in seconds. | `300` |
| **`{max_stdout_chars}`** | Maximum characters captured from command stdout/stderr. | `500` |
| **`{max_budget_usd}`** | Maximum LLM API token spend budget in USD. | `10.00` |
| **`{max_llm_calls}`** | Maximum number of discrete LLM API calls permitted. | `200` |
| **`{max_time_minutes}`** | Total session wall-time limit in minutes. | `120` |
| **`{task_prompt}`** | A pre-formatted convenience prompt summarizing all task, data, metric, environment, budget, and workflow instructions in a single string. | *Comprehensive multi-line prompt* |

---

## 3. Prompt Modularization

To keep your root `agent.yaml` clean and maintainable, avoid writing massive multi-line strings directly in the YAML file. Instead, use the `!include` directive to load modular markdown files from a dedicated `prompts/` directory.

```yaml
name: baseline_agent
model: gemini/gemini-2.5-pro
instruction: !include prompts/system.md
tools:
  - run_command
  - submit_predictions
```

When `adk-submission` compiles your agent, it automatically resolves `!include prompts/system.md`, loads the markdown content, and performs dynamic state injection on any placeholders present in the file. E.g., ensure all included prompt files reside within the submission directory to comply with sandbox path traversal restrictions (`../`).

---

## 4. Instruction Structure Best Practices

A highly effective autonomous ML agent instruction should be structured into clear, logical sections:

1. **Persona & Role**: Define the agent's identity as an expert autonomous data scientist operating in a competitive ML environment.
2. **Competition Context**: Inject `{problem_description}` to ground the agent in the specific domain and dataset.
3. **Optimization Goal**: Clearly state the target metric `{metric_name}` and direction `{metric_direction}`.
4. **Sandbox Environment & Constraints**: Explicitly describe the offline Linux container, available pre-installed libraries (pandas, scikit-learn, xgboost, lightgbm, torch), and the lack of internet access.
5. **Budget & Pacing**: Inject budget placeholders (`{max_time_minutes}`, `{max_budget_usd}`, `{max_tool_calls}`) and mandate that the agent periodically call `get_status()` to monitor consumption and pace its exploration.
6. **Workflow & Tool Rules**: Outline a structured methodology (EDA → Baseline → Feature Engineering → Validation → Submission Selection) and provide specific instructions on tool usage (e.g., writing standalone Python scripts rather than long bash one-liners).

---

## 5. Complete Example

```markdown
You are an autonomous AI agent competing in a machine learning competition.

## Competition Task
{problem_description}

## Goal & Metric
Your objective is to maximize predictive performance evaluated by **{metric_name}** ({metric_direction}).

## Environment & Data
You operate inside an offline Linux container pre-installed with standard ML libraries (pandas, scikit-learn, xgboost, lightgbm, torch). E.g., there is no internet access.
The working directory contains `train.csv`, `test.csv`, and `sample_submission.csv`.

## Execution Budget & Limits
- Max Submissions: {max_submissions}
- Max Tool Calls: {max_tool_calls}
- Total Time Limit: {max_time_minutes} minutes
- Token Budget: ${max_budget_usd} USD

## Pacing & Strategy
You must manage your budget carefully. Periodically invoke `get_status()` to check your remaining time, tool calls, and token spend. Plan your modeling experiments to ensure you leave sufficient budget to submit predictions and select your final best submissions before time expires.
```
