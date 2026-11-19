"""Which SAE features separate harmful from harmless prompts. Pure numpy, tested in CI.

Feature arrays have shape (n_prompts, n_features): SAE activations at the last prompt token.
"""

import numpy as np

NEURONPEDIA = "https://www.neuronpedia.org/gemma-2-2b/{layer}-gemmascope-res-16k/{feature}"
MIN_FIRE_RATE = 0.05  # ignore features that fire on under 5% of the positive prompts


def feature_auroc(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """AUROC of every feature (column) for telling pos rows from neg rows. Ties count half.

    SAE activations are mostly exact zeros, so ties are the normal case and are handled.
    """
    n_neg = neg.shape[0]
    out = np.empty(pos.shape[1])
    for j in range(pos.shape[1]):
        ref = np.sort(neg[:, j])
        less = np.searchsorted(ref, pos[:, j], side="left")
        leq = np.searchsorted(ref, pos[:, j], side="right")
        out[j] = ((less + 0.5 * (leq - less)) / n_neg).mean()
    return out


def top_features(
    train_pos: np.ndarray,
    train_neg: np.ndarray,
    val_pos: np.ndarray,
    val_neg: np.ndarray,
    k: int = 20,
) -> list[dict]:
    """The k features most specific to the positive class on train, rechecked on val.

    Ranking on train and reporting val keeps the list honest: a feature that only looked
    good by chance on train will show a weak val AUROC.
    """
    train_auc = feature_auroc(train_pos, train_neg)
    fire = (train_pos > 0).mean(axis=0)
    train_auc = np.where(fire >= MIN_FIRE_RATE, train_auc, 0.5)
    order = np.argsort(-train_auc)[:k]
    val_auc = feature_auroc(val_pos[:, order], val_neg[:, order])
    return [
        {
            "feature": int(f),
            "train_auroc": float(train_auc[f]),
            "val_auroc": float(val_auc[i]),
            "mean_pos": float(train_pos[:, f].mean()),
            "mean_neg": float(train_neg[:, f].mean()),
            "fire_rate_pos": float((train_pos[:, f] > 0).mean()),
            "fire_rate_neg": float((train_neg[:, f] > 0).mean()),
        }
        for i, f in enumerate(order)
    ]


def decoder_cosines(w_dec: np.ndarray, features: list[int], direction: np.ndarray) -> np.ndarray:
    """Cosine between each feature's decoder vector and the refusal direction."""
    vecs = w_dec[features]
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    return vecs @ (direction / np.linalg.norm(direction))


def direction_overlap(w_dec: np.ndarray, rows: list[dict], direction: np.ndarray) -> float:
    """Cosine between the refusal direction and the mean-activation-weighted sum of the top
    features' decoder vectors: how much of the direction these features rebuild."""
    ids = [r["feature"] for r in rows]
    weights = np.array([max(r["mean_pos"] - r["mean_neg"], 0.0) for r in rows])
    combo = (weights[:, None] * w_dec[ids]).sum(axis=0)
    return float(combo @ direction / max(np.linalg.norm(combo) * np.linalg.norm(direction), 1e-12))


def refusal_feature_score(feats: np.ndarray, ids: list[int], scale: np.ndarray) -> np.ndarray:
    """Per-prompt activation of the refusal features, each divided by its mean on harmful prompts
    so the score is about 1 on typical harmful prompts and about 0 on harmless ones."""
    return (feats[:, ids] / np.maximum(scale, 1e-12)).mean(axis=1)


def neuronpedia_url(layer: int, feature: int) -> str:
    return NEURONPEDIA.format(layer=layer, feature=feature)
