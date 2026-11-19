import numpy as np
import pytest

from safetycircuits.crosslingual.metrics import (
    CHRF_GOOD,
    bootstrap_auroc_ci,
    chrf,
    summarise_language,
)


def test_chrf_identical_is_100_and_unrelated_is_low():
    assert chrf("The cat sat on the mat", "The cat sat on the mat") == pytest.approx(100.0)
    assert chrf("xqzv wjk", "The cat sat on the mat") < 5
    assert chrf("", "something") == 0.0


def test_chrf_orders_closer_translations_higher():
    ref = "How do I reset my email password?"
    close = chrf("How can I reset my email password?", ref)
    far = chrf("What is the capital of France?", ref)
    assert close > far and close > CHRF_GOOD > far


def test_chrf_ignores_spaces():
    assert chrf("ab cd", "abcd") == pytest.approx(100.0)


def test_bootstrap_auroc_ci_contains_point_and_is_seeded():
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(1, 1, 60), rng.normal(0, 1, 60)
    point, lo, hi = bootstrap_auroc_ci(pos, neg, n_boot=200, seed=1)
    assert lo <= point <= hi and 0.5 < point < 1
    assert bootstrap_auroc_ci(pos, neg, n_boot=200, seed=1) == (point, lo, hi)


def test_summarise_language_fields():
    pos, neg = np.array([0.9, 1.1, 1.0, 0.8]), np.array([0.0, 0.1, -0.1, 0.05])
    row = summarise_language(
        pos,
        neg,
        np.array([0.0, 0.1]),
        np.array([True, True, True, False]),
        np.array([False, False, False, False]),
        np.array([False, False, False, False]),
    )
    assert row["direction_auroc"] == 1.0
    assert row["harmful_refusal_rate"] == 0.75 and row["harmful_refusal_rate_ablated"] == 0.0
    assert row["mean_score_flores_benign_text"] == pytest.approx(0.05)
    assert row["n_harmful"] == 4 and row["n_harmless"] == 4


def test_summarise_language_skips_missing_parts():
    row = summarise_language(np.array([1.0, 2.0]), np.array([0.0, 0.5]), None, None, None, None)
    assert "harmful_refusal_rate" not in row and "mean_score_flores_benign_text" not in row
