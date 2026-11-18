"""Single place for model ids, SAE choice, seeds and paths. PLAN.md decides the values."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

MODEL_ID = "google/gemma-2-2b-it"
# Gemma Scope 2B SAEs are trained on the BASE model, not on 2B-it (see PLAN.md).
SAE_REPO = "google/gemma-scope-2b-pt-res"
SAE_WIDTH = "16k"

LANGUAGES = {"twi": "twi_Latn", "french": "fra_Latn", "swahili": "swh_Latn"}

SEEDS = (0, 1, 2)


def hf_token() -> str | None:
    """Hugging Face read token, from the environment only (never from a file in the repo)."""
    return os.environ.get("HF_TOKEN") or None
