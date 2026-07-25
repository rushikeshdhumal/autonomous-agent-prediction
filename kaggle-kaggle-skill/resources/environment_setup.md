# Kaggle-in-Kaggle Environment Setup Guide

Different workflow steps in a Kaggle-in-Kaggle competition require specific prerequisites. Consult this guide to verify or configure the competitor's environment before executing validation, local evaluation, or Kaggle submission tasks. Do not install packages or modify environment files without the competitor's explicit approval.

---

## 1. For Validation & Local Evaluation

The pre-flight linter (`validate_submission.py`) and local evaluation harness (`run_local_eval.py`) require the `adk-submission` and `kaggle-kaggle` packages, along with their dependencies.

### Check Importability
Verify that the required packages are importable within the project environment (prefer `uv run` if `uv` is installed, otherwise `python3`):
```bash
uv run python -c "import adk_submission; import kaggle_kaggle"
```

### Set Up Virtual Environment
If the packages are missing or import checks fail, suggest creating a virtual environment with `uv` and installing dependencies from `requirements.txt`:
```bash
# Install uv if not already available:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install dependencies:
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

> **Prefer `uv` over `pip`**: Using `uv` allows prefixing Python commands with `uv run python` instead of manually activating the virtual environment. This guarantees scripts execute within the correct isolated project environment. The agent can install `uv` and set up the venv itself if requested by the competitor.

---

## 2. For Local Evaluation (Container Runtime & LLM API)

Local evaluation executes the participant's agent end-to-end inside a sandboxed container and requires direct LLM API access.

### Configure LLM API Keys
Copy the environment template and add direct provider API keys (`GEMINI_API_KEY`, `OPENAI_API_KEY`, etc.) to `.env`:
```bash
cp .env.example .env
```

### Container Runtime Verification
Check that a container runtime is available and the official Kaggle Python sandbox image has been pulled:
```bash
# Check for Podman (preferred) or Docker:
podman --version || docker --version

# Pull the sandbox image:
podman pull gcr.io/kaggle-images/python
```
If neither runtime is installed, suggest installing Podman (preferred over Docker for rootless operation).

> **Podman Configuration**: When using Podman instead of Docker, `kaggle-kaggle` requires `DOCKER_HOST` to be set in the environment so the Python Docker SDK can locate the Podman socket (e.g., `export DOCKER_HOST=unix:///tmp/podman.sock` or `export DOCKER_HOST=unix:///run/user/1000/podman/podman.sock`).

---

## 3. For Kaggle Submission

Submitting an agent archive requires accepting the competition rules and configuring the Kaggle CLI utility.

### Accept Competition Rules
Before submitting, the competitor **must** accept the rules on the Kaggle website. Navigate to `https://www.kaggle.com/competitions/<competition-slug>` and click **"Join Competition"**.

### Install Kaggle CLI
Verify that the `kaggle` CLI utility is installed. If not, suggest installing it into the local virtual environment:
```bash
uv pip install kaggle
```

### Authenticate & Verify Entry
Credentials must be saved to `~/.kaggle/kaggle.json` (downloaded from https://www.kaggle.com/settings/api). Verify authentication and competition entry status:
```bash
# Verify CLI connection:
kaggle competitions list -s "<competition-slug>"

# Verify you have entered the competition:
kaggle competitions list --group entered
```
