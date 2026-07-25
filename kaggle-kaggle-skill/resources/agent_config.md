# Agent Config Reference (`agent.yaml`)

> Table of Contents
> 1. [Security Model](#1-security-model)
> 2. [Agent Classes](#2-agent-classes)
> 3. [LlmAgent Fields](#3-llmagent-fields)
> 4. [Workflow Agents](#4-workflow-agents)
> 5. [Sub-Agent & Tool References](#5-sub-agent--tool-references)
> 6. [Generation Configuration](#6-generation-configuration)
> 7. [Advanced Features](#7-advanced-features)

`adk-submission` is a sandboxed compiler that builds ADK agents from declarative YAML configurations. Arbitrary code execution is prohibited — all tools, models, and callbacks resolve against closed registries controlled by the competition organizer.

---

## 1. Security Model

The key constraints to internalize:

- **No `importlib` / No `CodeConfig`**: All Python paths are replaced by plain string lookups against registries.
- **Closed Registries**: Tools, models, skills, and callbacks resolve against `ToolRegistry`, `ModelRegistry`, `SkillRegistry`, `CallbackRegistry`.
- **Strict Schema (`extra="forbid"`)**: Unknown or unmapped YAML fields are rejected at parse time.
- **Sandboxed `!include`**: Path traversal (`../`) and symlinks are blocked.

---

## 2. Agent Classes

Every submission must provide a root `agent.yaml`. The `agent_class` key selects the agent type:

| `agent_class` | ADK Class | Purpose |
| :--- | :--- | :--- |
| `LlmAgent` (or omitted) | `google.adk.agents.Agent` | LLM-driven agent with tools, instructions, and reasoning. |
| `SequentialAgent` | `google.adk.agents.SequentialAgent` | Executes sub-agents sequentially. |
| `ParallelAgent` | `google.adk.agents.ParallelAgent` | Executes sub-agents concurrently. |
| `LoopAgent` | `google.adk.agents.LoopAgent` | Repeats sub-agents in a loop. |

---

## 3. LlmAgent Fields

`LlmAgent` is the default (used when `agent_class` is omitted).

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **`name`** | `str` | *Required* | Unique agent identifier. |
| **`description`** | `str` | `""` | Agent role description, used by parent agents for delegation. |
| **`model`** | `str` | `None` | Alias from `ModelRegistry` (e.g., `strong`, `fast`), or `adapter:my_lora`. |
| **`instruction`** | `str` | *Required* | System prompt. Supports `!include file.md`. |
| **`output_key`** | `str` | `None` | Key in `session.state` for the agent's final response. |
| **`include_contents`** | `str` | `"default"` | `"default"` or `"none"` — controls conversation history. |
| **`disallow_transfer_to_parent`** | `bool` | `None` | Prevents delegation back to parent. |
| **`disallow_transfer_to_peers`** | `bool` | `None` | Prevents delegation to peer sub-agents. |
| **`tools`** | `list` | `None` | Tool names from `ToolRegistry` or `AgentToolEntry` sub-agents. |
| **`skills`** | `list[str]` | `None` | Relative paths to bundled skill directories. |
| **`sub_agents`** | `list[SubAgentRef]` | `None` | Sub-agents available for LLM delegation. |
| **`generate_content_config`** | `dict` | `None` | LLM generation parameters (validated against organizer constraints). |
| **`before_agent_callbacks`** | `list[str]` | `None` | Pre-approved callbacks before agent runs. |
| **`after_agent_callbacks`** | `list[str]` | `None` | Pre-approved callbacks after agent finishes. |
| **`before_model_callbacks`** | `list[str]` | `None` | Pre-approved callbacks before each LLM call. |
| **`after_model_callbacks`** | `list[str]` | `None` | Pre-approved callbacks after each LLM call. |
| **`before_tool_callbacks`** | `list[str]` | `None` | Pre-approved callbacks before tool invocation. |
| **`after_tool_callbacks`** | `list[str]` | `None` | Pre-approved callbacks after tool finishes. |

---

## 4. Workflow Agents

Workflow agents provide deterministic orchestration without LLM delegation.

### `SequentialAgent`
Executes sub-agents one after another. State changes propagate to subsequent agents.
- `agent_class`: `SequentialAgent`
- `name`: `str` (Required)
- `sub_agents`: `list[SubAgentRef]` (Required, at least one)
- `description`, `before_agent_callbacks`, `after_agent_callbacks`: Optional

### `ParallelAgent`
Executes sub-agents concurrently. Sub-agents should write to distinct `output_key`s to avoid races.
- `agent_class`: `ParallelAgent`
- `name`: `str` (Required)
- `sub_agents`: `list[SubAgentRef]` (Required)

### `LoopAgent`
Repeats sub-agents until `max_iterations` or escalation (`escalate=True`).
- `agent_class`: `LoopAgent`
- `name`: `str` (Required)
- `sub_agents`: `list[SubAgentRef]` (Required)
- `max_iterations`: `int` (Optional, defaults to organizer's limit)

---

## 5. Sub-Agent & Tool References

References must point to YAML files within the submission directory — no Python code imports.

### `SubAgentRef`
```yaml
sub_agents:
  - config_path: sub_agents/researcher.yaml
  - config_path: sub_agents/critic.yaml
```

### `AgentToolEntry`
Wraps another agent as a callable tool:
```yaml
tools:
  - run_command
  - agent_tool:
      config_path: sub_agents/summarizer.yaml
      skip_summarization: true  # pass raw output instead of summarizing
```

---

## 6. Generation Configuration

Configure LLM parameters via `generate_content_config`. Values are validated against organizer-defined `GenerationConstraints`.

### Settable Fields
```yaml
generate_content_config:
  temperature: 0.5
  top_p: 0.95
  top_k: 40
  max_output_tokens: 4096
  stop_sequences: ["END", "FINAL_ANSWER"]
  presence_penalty: 0.1
  frequency_penalty: 0.2
  response_mime_type: text/plain
  seed: 0
  thinking_config:
    thinking_budget: 1024
    include_thoughts: true
```

### Forbidden Fields
These are hard-excluded to prevent safety bypass: `safety_settings`, `system_instruction`, `response_schema`, `tools`.

Check competition guidelines for which settable fields are permitted and their allowable ranges.

---

## 7. Advanced Features

### Sandboxed `!include`
Modularize prompts and configs:
```yaml
instruction: !include prompts/expert_prompt.md
generate_content_config: !include configs/sampling.yaml
```
Supported extensions: `.md`, `.txt`, `.yaml`, `.yml`. Paths must be relative to submission root. No absolute paths, no `../`, no symlinks.

### Model Adapters
Bundle fine-tuned weights (LoRA) in the submission:
```yaml
model: adapter:my_custom_lora
```
The compiler matches against the adapter manifest and resolves via the organizer's serving infrastructure.

### Custom Skills
Bundle skill directories with `SKILL.md` manifests:
```yaml
skills:
  - skills/sql_analyst
```
See `resources/agent_skills.md` for authoring details.
