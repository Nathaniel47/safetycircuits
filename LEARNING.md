# LEARNING.md: SafetyCircuits

Plain-English notes on what was built and why it matters. One section per step.

## Step 2: Repo setup

**What was built.** An empty but working project: a Python package `safetycircuits/` with one folder per part of the study, a `config.py` that holds the model and SAE choices, dependency files, a Dockerfile, formatting and secret checks, and a GitHub Actions workflow that runs the tests.

**Why the dependencies are split in three.**
- `requirements.txt` is what the feature browser needs (Streamlit, Plotly, numpy, pandas). It has no PyTorch, so the app deploys on free hosting and CI stays fast.
- `requirements-gpu.txt` adds PyTorch, Transformers and SAELens. You only install it in Kaggle or Colab, where the GPU is.
- `requirements-dev.txt` adds pytest, ruff and pre-commit.

This mirrors the plan: the heavy model work happens on a free GPU and saves files, and the laptop and the web app only read those files.

**Secrets.** The only secret is the Hugging Face token (`HF_TOKEN`), needed because Gemma is gated. It lives in `.env` (ignored by git) or in Kaggle/Colab Secrets. The `.env.example` file lists the name with an empty value, and a test checks that stays true. The gitleaks pre-commit hook blocks commits that look like they contain a key.

**Responsible handling.** CLAUDE.md records the rule that the study stores refusal flags, scores and activations, never harmful model text.

**How to check it works.** Activate a virtual environment, run `pip install -r requirements-dev.txt`, then `pytest` and `ruff check .`. All should pass.

**Reading.** The pre-commit documentation (pre-commit.com) intro page: it explains hooks in five minutes.
