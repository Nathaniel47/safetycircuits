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

# Residual-stream layers whose Gemma Scope SAE we use: the layer the refusal direction was found
# at (24, from results/refusal_direction.json) and a middle layer for comparison.
SAE_LAYERS = (17, 24)
SAE_TARGET_L0 = 100  # pick the SAE whose average L0 is closest to this

TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"  # CC-BY-NC-4.0
BASE_MODEL_ID = "google/gemma-2-2b"  # only for the SAE-transfer comparison

LANGUAGES = {"twi": "twi_Latn", "french": "fra_Latn", "swahili": "swh_Latn"}

SEEDS = (0, 1, 2)


def hf_token() -> str | None:
    """Hugging Face read token, from the environment only (never from a file in the repo)."""
    return os.environ.get("HF_TOKEN") or None
