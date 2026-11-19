"""Find the refusal direction: the difference between mean harmful and mean harmless activations.

Pure numpy, so it is tested in CI without torch. Activation arrays have shape
(n_prompts, n_layers, d_model): the residual stream at the last prompt token, after every layer.
"""

import numpy as np


def diff_in_means(harmful: np.ndarray, harmless: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per layer: unit direction (n_layers, d) and the length of the raw difference (n_layers,).

    The length is the typical gap between harmful and harmless along the direction, so it is the
    natural size for adding the direction to a prompt.
    """
    diff = harmful.mean(axis=0) - harmless.mean(axis=0)
    norm = np.linalg.norm(diff, axis=-1)
    return diff / np.maximum(norm[:, None], 1e-12), norm


def project(acts: np.ndarray, unit: np.ndarray) -> np.ndarray:
    """Scalar coordinate of each prompt along the direction. acts (n, d), unit (d,) -> (n,)."""
    return acts @ unit


def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Chance that a random positive scores above a random negative (ties count half)."""
    scores = np.concatenate([positive, negative])
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    for value in np.unique(scores):  # average the ranks of tied scores
        tied = scores == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    n_pos, n_neg = len(positive), len(negative)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def cohens_d(positive: np.ndarray, negative: np.ndarray) -> float:
    pooled = np.sqrt((positive.var(ddof=1) + negative.var(ddof=1)) / 2)
    return float((positive.mean() - negative.mean()) / max(pooled, 1e-12))


def rank_layers(
    harmful_train: np.ndarray,
    harmless_train: np.ndarray,
    harmful_val: np.ndarray,
    harmless_val: np.ndarray,
) -> list[dict]:
    """Fit the direction on train, measure how well it separates val prompts, best layer first.

    AUROC saturates near 1.0 in many layers, so Cohen's d breaks ties. Separation alone does not
    prove a layer *controls* refusal: the run script also tests ablation on the top layers.
    """
    unit, norm = diff_in_means(harmful_train, harmless_train)
    rows = []
    for layer in range(unit.shape[0]):
        pos = project(harmful_val[:, layer], unit[layer])
        neg = project(harmless_val[:, layer], unit[layer])
        rows.append(
            {
                "layer": layer,
                "auroc": auroc(pos, neg),
                "cohens_d": cohens_d(pos, neg),
                "norm": float(norm[layer]),
            }
        )
    return sorted(rows, key=lambda r: (-round(r["auroc"], 3), -r["cohens_d"]))


def random_unit(d_model: int, seed: int) -> np.ndarray:
    """Control: a random direction. Removing it should barely change refusal."""
    v = np.random.default_rng(seed).standard_normal(d_model)
    return v / np.linalg.norm(v)


def calibrated_score(
    acts: np.ndarray, unit: np.ndarray, harmless_mean: float, harmful_mean: float
) -> np.ndarray:
    """Refusal score: 0 at the typical harmless prompt, 1 at the typical harmful one.

    acts (n, d) at the layer the direction was found at; harmless_mean and harmful_mean are the
    mean projections of the train prompts of each class.
    """
    return (project(acts, unit) - harmless_mean) / (harmful_mean - harmless_mean)
