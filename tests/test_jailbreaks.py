import numpy as np
import pytest

from safetycircuits.jailbreaks.study import (
    BASELINE,
    CONDITIONS,
    KIND,
    apply,
    pooled_success_vs_refused_features,
    rank_conditions,
    summarise,
)
from safetycircuits.jailbreaks.suffixes import SUFFIXES, random_words
from safetycircuits.jailbreaks.templates import TEMPLATES
from safetycircuits.refusal.direction import calibrated_score


def test_eight_jailbreaks_plus_baseline():
    assert len(TEMPLATES) == 5 and len(SUFFIXES) == 3
    assert CONDITIONS[0] == BASELINE and len(CONDITIONS) == 9
    assert sorted(set(KIND.values())) == ["baseline", "suffix", "template"]


def test_every_template_contains_the_prompt_exactly_once():
    for name, template in TEMPLATES.items():
        assert template.count("{prompt}") == 1, name
        assert "PROMPT-X" in apply(name, "PROMPT-X")


def test_apply_baseline_and_suffix():
    assert apply(BASELINE, "hello") == "hello"
    for name, suffix in SUFFIXES.items():
        assert apply(name, "hello") == "hello" + suffix


def test_random_suffix_is_fixed_by_seed():
    assert random_words(seed=0) == random_words(seed=0)
    assert random_words(seed=0) != random_words(seed=1)


def test_wrappers_contain_no_harm_of_their_own():
    text = " ".join(TEMPLATES.values()) + " ".join(SUFFIXES.values())
    for word in ("bomb", "weapon", "hack", "kill", "poison", "malware"):
        assert word not in text.lower()


def test_calibrated_score_is_zero_for_harmless_mean_and_one_for_harmful_mean():
    unit = np.array([1.0, 0.0])
    acts = np.array([[2.0, 5.0], [10.0, -1.0]])
    assert calibrated_score(acts, unit, 2.0, 10.0).tolist() == [0.0, 1.0]


def test_summarise_counts_only_coherent_non_refusals_as_success():
    refused = np.array([True, False, False, False])
    degenerate = np.array([False, True, False, False])
    score = np.array([1.0, 0.2, 0.3, 0.5])
    row = summarise(refused, degenerate, score, None, base_score=1.0)
    assert row["success_rate"] == 0.5  # only the last two are real successes
    assert row["refusal_rate"] == 0.25 and row["degenerate_rate"] == 0.25
    assert row["score_suppression"] == pytest.approx(1 - np.mean(score))
    assert "mean_feature_score" not in row


def test_score_predicts_refusal_auroc_when_score_tracks_refusal():
    refused = np.array([True, True, False, False])
    row = summarise(refused, np.zeros(4, bool), np.array([0.9, 0.8, 0.1, 0.2]), None, 1.0)
    assert row["score_predicts_refusal_auroc"] == 1.0


def test_pooled_features_find_the_one_that_marks_refusal():
    rng = np.random.default_rng(0)
    feats = (rng.random((60, 10)) < 0.2) * rng.random((60, 10))
    success = np.arange(60) % 2 == 0
    feats[~success, 4] = 3.0 + rng.random((~success).sum())
    rows = pooled_success_vs_refused_features(feats, success, k=2)
    assert rows[0]["feature"] == 4 and rows[0]["auroc_refused_vs_success"] > 0.99
    assert pooled_success_vs_refused_features(feats, np.ones(60, bool)) == []


def test_rank_conditions_excludes_baseline_and_sorts_by_success():
    summary = {
        BASELINE: {"success_rate": 0.0},
        "a": {"success_rate": 0.1},
        "b": {"success_rate": 0.4},
    }
    assert rank_conditions(summary) == ["b", "a"]
