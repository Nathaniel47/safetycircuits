# SafetyCircuits: Interpreting Refusal and Jailbreaks in a Small LLM

An open study of how Gemma-2-2B-it decides to refuse, using a refusal direction, Gemma Scope sparse autoencoders and activation steering, and whether that mechanism carries over to Twi, French and Swahili. It measures refusal behaviour only and publishes no harmful completions.

Status: data pipeline done (step 3). See [PLAN.md](PLAN.md) for the design and [BUILD_STEPS.md](BUILD_STEPS.md) for progress.

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

Build the data (needs internet, about 70 MB): `python -m safetycircuits.data.fetch`, then `python -m safetycircuits.data.prepare`, then `python -m safetycircuits.data.report`. Raw files and prompt text stay in `data/raw/` and `data/interim/` (gitignored). The repo holds only ids (`data/splits.json`), checksums (`data/manifest.json`) and the report ([results/data_report.md](results/data_report.md)).

| Data | Licence (checked 2026-11-18) | Use here |
|---|---|---|
| AdvBench (llm-attacks) | MIT | 520 harmful behaviours, from the GitHub CSV (the Hugging Face copy is gated) |
| HarmBench | MIT | 200 standard text behaviours |
| XSTest | CC-BY-4.0 | 450 prompts (250 safe, 200 unsafe) for over-refusal |
| Alpaca-cleaned | CC-BY-4.0 on the dataset card; disputed because the text came from OpenAI models | 800 prompts sampled as harmless examples; prompts only, not redistributed |
| FLORES-200 | CC-BY-SA 4.0 | Twi, French, Swahili and English devtest sentences; anything derived from it keeps the same licence |
| Gemma-2-2B-it | Gemma Terms of Use + Prohibited Use Policy (gated) | the model under study |
| Gemma Scope SAEs | CC-BY-4.0 | sparse autoencoders (trained on the base model) |

## Responsible use

Public benchmark prompts only. The code stores refusal flags, scores and activations, never harmful model output.
