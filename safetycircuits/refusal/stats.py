"""Small statistics helpers (no numpy, no torch)."""

import math


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Rate and 95% Wilson confidence interval for a share of n prompts."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)
