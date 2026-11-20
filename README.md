# SafetyCircuits: Interpreting Refusal and Jailbreaks in a Small LLM

An open study of how Gemma-2-2B-it decides to refuse, using a refusal direction, Gemma Scope sparse autoencoders and activation steering, and whether that mechanism carries over to Twi, French and Swahili. It measures refusal behaviour only and publishes no harmful completions.

Status: steps 2-5 done, with results in results/ (see results/step5_results_summary.md). Step 6 evaluation done (results/results_table.md, results/limitations.md). Next: step 7 (demo and deploy). See [PLAN.md](PLAN.md) for the design and [BUILD_STEPS.md](BUILD_STEPS.md) for progress.

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
4. Step 4 runner: [notebooks/01_refusal_direction.ipynb](notebooks/01_refusal_direction.ipynb) (clones the repo, builds the data, runs `python -m safetycircuits.refusal.run`). Run it with `--limit 16` first as a smoke test.

## Feature browser

```bash
streamlit run app/browser.py
```

It reads `results/browser_cache.json` (237 prompts, committed; no GPU needed). To rebuild it, run notebooks 02 to 05 in `notebooks/` on Kaggle (`02` SAE features, `03` jailbreaks, `04` cross-lingual, `05` browser cache). Without the cache the app shows setup instructions. Typing a prompt that is not in the cache needs a live model, which is not wired in: pick a cached prompt instead.

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
