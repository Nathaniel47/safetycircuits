"""Logic behind the feature browser, kept out of the Streamlit file so it can be tested.

The browser runs from results/browser_cache.json, built on a GPU by
`python -m safetycircuits.sae.build_cache`. No model is needed to browse it.
"""

import difflib
import json
import os
import re
from pathlib import Path

from safetycircuits.config import RESULTS_DIR

CACHE_ENV = "SAFETYCIRCUITS_CACHE"
RESULT_FILES = {
    "direction": "refusal_direction.json",
    "sae": "sae_features.json",
    "jailbreaks": "jailbreaks.json",
    "crosslingual": "crosslingual.json",
}
LOW, HIGH = 0.3, 0.7  # score bands: arbitrary, chosen only to give the number a readable label


def cache_path() -> Path:
    return Path(os.environ.get(CACHE_ENV) or RESULTS_DIR / "browser_cache.json")


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def load_cache(path: Path | None = None) -> dict | None:
    path = path or cache_path()
    if not path.exists():
        return None
    cache = json.loads(path.read_text(encoding="utf-8"))
    cache["_index"] = {normalise(e["text"]): e for e in cache["prompts"]}
    return cache


def find_prompt(cache: dict, text: str) -> dict | None:
    return cache["_index"].get(normalise(text))


def examples(cache: dict, per_group: int = 2) -> list[dict]:
    """A few prompts from each group, for the one-click buttons."""
    seen: dict[str, int] = {}
    out = []
    for entry in cache["prompts"]:
        if seen.get(entry["group"], 0) < per_group:
            seen[entry["group"]] = seen.get(entry["group"], 0) + 1
            out.append(entry)
    return out


def suggest(cache: dict, text: str, n: int = 5) -> list[dict]:
    """The cached prompts closest to what was typed, for a "did you mean" list."""
    keys = list(cache["_index"])
    close = difflib.get_close_matches(normalise(text), keys, n=n, cutoff=0.3)
    return [cache["_index"][k] for k in close]


def group_names(cache: dict) -> list[str]:
    """Group names in the order they first appear."""
    return list(dict.fromkeys(e["group"] for e in cache["prompts"]))


def prompts_in_group(cache: dict, group: str) -> list[dict]:
    return [e for e in cache["prompts"] if e["group"] == group]


def score_band(score: float) -> str:
    if score < LOW:
        return "closer to harmless prompts"
    if score > HIGH:
        return "closer to harmful prompts"
    return "in between"


def feature_rows(entry: dict, cache: dict) -> list[dict]:
    """Top features of a prompt joined with what is known about each feature."""
    table = cache.get("features", {})
    rows = []
    for feature, activation, *rest in entry["top_features"]:
        info = table.get(str(feature), {})
        rows.append(
            {
                "feature": feature,
                "activation": activation,
                "lift": rest[0] if rest else None,
                "refusal_feature": bool(info.get("is_refusal_feature")),
                "url": info.get("url") or "",
            }
        )
    return rows


def load_results(directory: Path | None = None) -> dict[str, dict]:
    """Whichever result files exist, parsed. Missing ones are simply absent."""
    directory = directory or RESULTS_DIR
    out = {}
    for key, name in RESULT_FILES.items():
        path = directory / name
        if path.exists():
            out[key] = json.loads(path.read_text(encoding="utf-8"))
    return out
