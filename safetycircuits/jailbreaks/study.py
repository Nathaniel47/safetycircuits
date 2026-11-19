"""Compare jailbreaks by outcome and by how much they suppress the refusal signals.

Pure numpy, tested in CI. A "success" is a reply that is not a refusal and is not degenerate.
"""

import numpy as np

from safetycircuits.jailbreaks.suffixes import SUFFIXES
from safetycircuits.jailbreaks.templates import TEMPLATES
from safetycircuits.refusal.direction import auroc
from safetycircuits.refusal.stats import wilson

BASELINE = "baseline"
CONDITIONS = [BASELINE, *TEMPLATES, *SUFFIXES]
KIND = {
    BASELINE: "baseline",
    **dict.fromkeys(TEMPLATES, "template"),
    **dict.fromkeys(SUFFIXES, "suffix"),
}


def apply(condition: str, prompt: str) -> str:
    if condition == BASELINE:
        return prompt
    if condition in TEMPLATES:
        return TEMPLATES[condition].format(prompt=prompt)
    return prompt + SUFFIXES[condition]


def summarise(
    refused: np.ndarray,
    degenerate: np.ndarray,
    score: np.ndarray,
    feature_score: np.ndarray | None,
    base_score: float,
    harmless_score: float = 0.0,
) -> dict:
    """One condition, one row per harmful test prompt (arrays of equal length)."""
    n = len(refused)
    success = ~refused & ~degenerate
    rate, lo, hi = wilson(int(success.sum()), n)
    row = {
        "n": n,
        "refusal_rate": float(refused.mean()),
        "degenerate_rate": float(degenerate.mean()),
        "success_rate": rate,
        "success_ci95": [lo, hi],
        "mean_refusal_score": float(score.mean()),
        # 1 means the score fell all the way to the harmless level, 0 means no change
        "score_suppression": float((base_score - score.mean()) / (base_score - harmless_score)),
    }
    if feature_score is not None:
        row["mean_feature_score"] = float(feature_score.mean())
    if refused.any() and (~refused).any():
        # Does a lower refusal score go with not refusing, inside this condition?
        row["score_predicts_refusal_auroc"] = auroc(score[refused], score[~refused])
    return row


def pooled_success_vs_refused_features(
    feats: np.ndarray, success: np.ndarray, k: int = 20
) -> list[dict]:
    """Features most active when a jailbreak failed (refused) compared with when it succeeded.

    feats (n_rows, n_features) pools rows from all jailbreak conditions; success is a bool per row.
    """
    from safetycircuits.sae.features import feature_auroc

    refused_rows, success_rows = feats[~success], feats[success]
    if len(refused_rows) == 0 or len(success_rows) == 0:
        return []
    auc = feature_auroc(refused_rows, success_rows)
    order = np.argsort(-auc)[:k]
    return [
        {
            "feature": int(f),
            "auroc_refused_vs_success": float(auc[f]),
            "mean_refused": float(refused_rows[:, f].mean()),
            "mean_success": float(success_rows[:, f].mean()),
        }
        for f in order
    ]


def rank_conditions(summary: dict[str, dict]) -> list[str]:
    """Jailbreak conditions, most successful first (baseline excluded)."""
    names = [c for c in summary if c != BASELINE]
    return sorted(names, key=lambda c: -summary[c]["success_rate"])
