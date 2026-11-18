# SafetyCircuits: Interpreting Refusal and Jailbreaks in a Small LLM

An open study of how Gemma-2-2B-it decides to refuse, using a refusal direction, Gemma Scope sparse autoencoders and activation steering, and whether that mechanism carries over to Twi, French and Swahili. It measures refusal behaviour only and publishes no harmful completions.

Status: setup done (step 2). See [PLAN.md](PLAN.md) for the design and [BUILD_STEPS.md](BUILD_STEPS.md) for progress.

## Run locally (laptop: tests and the feature browser, no GPU)

Windows (PowerShell):
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest
streamlit run app/browser.py
```

macOS / Linux:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
streamlit run app/browser.py
```

Docker: `docker compose up` serves the browser on http://localhost:8501.

## GPU work (Kaggle or Colab)

Model runs need a 16 GB GPU, so they run in notebooks under `notebooks/` and save results to files.

1. On huggingface.co: log in, open `google/gemma-2-2b-it`, click **Acknowledge license**.
2. Create a read token (Settings → Access Tokens). Locally put it in `.env` as `HF_TOKEN=...` (gitignored). On Kaggle use Add-ons → Secrets; on Colab use Secrets.
3. In the notebook: `pip install -r requirements-gpu.txt`.

## Data and licences

Filled in during step 3 (data pipeline). Sources and licences checked so far are in [PLAN.md](PLAN.md) section 5.

## Responsible use

Public benchmark prompts only. The code stores refusal flags, scores and activations, never harmful model output.
