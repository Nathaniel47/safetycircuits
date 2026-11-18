"""Seeded, stratified splitting. Splits store ids only, so the repo holds no prompt text."""

import random


def split_ids(ids_by_group: dict[str, list[str]], fractions: dict[str, float], seed: int):
    """Split each group separately (stratified) so every split keeps the source balance."""
    assert abs(sum(fractions.values()) - 1) < 1e-9
    rng = random.Random(seed)
    out: dict[str, list[str]] = {name: [] for name in fractions}
    names = list(fractions)
    for group in sorted(ids_by_group):
        ids = sorted(ids_by_group[group])
        rng.shuffle(ids)
        start = 0
        for i, name in enumerate(names):
            end = len(ids) if i == len(names) - 1 else start + round(len(ids) * fractions[name])
            out[name].extend(ids[start:end])
            start = end
    return {name: sorted(v) for name, v in out.items()}
