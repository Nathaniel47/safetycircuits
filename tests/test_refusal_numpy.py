import numpy as np
import pytest

from safetycircuits.refusal.classify import is_refusal
from safetycircuits.refusal.direction import (
    auroc,
    diff_in_means,
    project,
    random_unit,
    rank_layers,
)
from safetycircuits.refusal.stats import wilson


def synthetic(n=60, layers=4, d=16, signal_layers=(2, 3), seed=0):
    """Harmless ~ noise; harmful = noise + a fixed vector, only in some layers."""
    rng = np.random.default_rng(seed)
    true = rng.standard_normal(d)
    true /= np.linalg.norm(true)
    harmless = rng.standard_normal((n, layers, d))
    harmful = rng.standard_normal((n, layers, d))
    for layer in signal_layers:
        harmful[:, layer] += 6 * true
    return harmful, harmless, true


def test_diff_in_means_recovers_the_planted_direction():
    harmful, harmless, true = synthetic(n=400)
    unit, norm = diff_in_means(harmful, harmless)
    assert np.allclose(np.linalg.norm(unit, axis=-1), 1)
    assert abs(unit[3] @ true) > 0.95
    assert norm[3] > norm[0]


def test_auroc_known_values():
    assert auroc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert auroc(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 0.0
    assert auroc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5


def test_rank_layers_puts_signal_layers_first():
    tr_h, tr_l, _ = synthetic(seed=1)
    va_h, va_l, _ = synthetic(seed=1, n=40)
    ranking = rank_layers(tr_h, tr_l, va_h, va_l)
    assert {r["layer"] for r in ranking[:2]} == {2, 3}
    assert ranking[0]["auroc"] > 0.95
    assert ranking[-1]["auroc"] < 0.75


def test_projection_separates_classes():
    harmful, harmless, true = synthetic()
    pos, neg = project(harmful[:, 3], true), project(harmless[:, 3], true)
    assert pos.mean() - neg.mean() > 4


def test_random_unit_is_unit_and_seeded():
    a = random_unit(32, 0)
    assert np.isclose(np.linalg.norm(a), 1)
    assert np.array_equal(a, random_unit(32, 0))
    assert not np.array_equal(a, random_unit(32, 1))


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("I cannot fulfill your request. I'm just an AI.", True),
        ("I’m sorry, but I can’t help with that.", True),
        ("As an AI language model, I must decline.", True),
        ("Sure! Here is a poem about autumn.", False),
        ("To terminate a Python process, use kill. I cannot stress enough how handy it is", False),
        ("", False),
    ],
)
def test_is_refusal(reply, expected):
    assert is_refusal(reply) is expected


def test_wilson_interval():
    p, lo, hi = wilson(50, 100)
    assert p == 0.5 and 0.40 < lo < 0.41 and 0.59 < hi < 0.60
    assert wilson(0, 10)[1] == 0.0
    assert wilson(10, 10)[2] == 1.0
