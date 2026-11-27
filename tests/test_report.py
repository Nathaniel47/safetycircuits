import json

import pytest

from safetycircuits.evaluation import report


def rate(r, n=100):
    return {"n": n, "refusals": int(r * n), "rate": r, "ci95": [max(0, r - 0.05), min(1, r + 0.05)]}


def fake_results():
    sweep = {
        str(layer): {
            "val_refusal_after_ablation": rate(0.9 - layer / 40),
            "val_auroc": 0.99,
        }
        for layer in (10, 17, 24)
    }
    agree = {
        "n": 10,
        "refused": 3,
        "not_refused": 7,
        "clear_compliance": 5,
        "unclear": 2,
        "refused_and_affirmative": 0,
        "degenerate": 0,
        "clear_share_of_not_refused": 5 / 7,
    }
    sd = {"mean": 0.1, "sd": 0.02}
    return {
        "direction": {
            "chosen_layer": 24,
            "results": {"harmful_test": {"baseline": rate(0.98), "ablated": rate(0.08)}},
        },
        "baselines": {
            "baselines_auroc_test": {
                "bag_of_words_naive_bayes": 0.99,
                "first_word_only": 0.87,
                "prompt_length_in_words": 0.72,
            },
            "xstest_unsafe_vs_safe_auroc": {
                "bag_of_words_naive_bayes": 0.61,
                "first_word_only": 0.53,
                "prompt_length_in_words": 0.49,
            },
            "style_matched": {
                "n_harmless_matched_val_test": 69,
                "bag_of_words_naive_bayes": 0.98,
                "prompt_length_in_words": 0.70,
            },
        },
        "evaluation": {
            "meta": {"layer_chosen_on_validation": 17},
            "layer_sweep_validation": sweep,
            "test": {
                "baseline_refusal": rate(0.98),
                "17": {
                    "refusal_after_ablation": rate(0.05),
                    "auroc_harmful_vs_harmless": 0.999,
                    "auroc_xstest_unsafe_vs_safe": 0.9,
                    "auroc_harmful_vs_style_matched_harmless": 0.99,
                },
                "24": {
                    "refusal_after_ablation": rate(0.08),
                    "auroc_harmful_vs_harmless": 0.999,
                    "auroc_xstest_unsafe_vs_safe": 0.85,
                    "auroc_harmful_vs_style_matched_harmless": 0.99,
                },
            },
            "seed_variance_80pct_subsamples": {
                "summary": {
                    layer: {
                        "refusal_after_ablation": sd,
                        "auroc": {"mean": 0.999, "sd": 0.0001},
                        "cosine_with_full_direction": {"mean": 0.99, "sd": 0.001},
                    }
                    for layer in ("17", "24")
                }
            },
            "random_directions_test_refusal": {
                "rates": [0.97, 0.98, 0.98],
                "mean": 0.977,
                "sd": 0.006,
            },
            "classifier_checks": {"harmful_baseline": agree},
        },
        "jailbreaks": {
            "meta": {"layer": 24},
            "summary": {
                "baseline": {
                    "success_rate": 0.02,
                    "success_ci95": [0.0, 0.06],
                    "mean_refusal_score": 0.98,
                    "score_suppression": 0.0,
                },
                "fiction_frame": {
                    "success_rate": 0.4,
                    "success_ci95": [0.3, 0.5],
                    "mean_refusal_score": 0.3,
                    "score_suppression": 0.7,
                },
            },
        },
        "crosslingual": {
            "languages": {
                "english": {
                    "all": {
                        "direction_auroc": 0.99,
                        "direction_auroc_ci95": [0.98, 1.0],
                        "harmful_refusal_rate": 0.97,
                        "harmful_refusal_rate_ablated": 0.05,
                    }
                },
                "twi": {
                    "all": {
                        "direction_auroc": 0.56,
                        "direction_auroc_ci95": [0.48, 0.64],
                        "harmful_refusal_rate": 0.06,
                        "harmful_refusal_rate_ablated": 0.0,
                    },
                    "gemma_comprehension_chrf_harmless": 18.1,
                },
            }
        },
    }


def test_human_review_counts_and_table_section():
    review = json.loads((report.RESULTS_DIR / "human_review.json").read_text(encoding="utf-8"))
    summary = report.review_summary(review)
    assert {s["n"] for s in summary.values()} == {review["meta"]["n_prompts"]}
    removed = summary["removed"]
    assert (removed["judge_refused"], removed["refused"], removed["complied"]) == (3, 15, 5)
    assert removed["agree"] == 6
    assert summary["fiction"]["complied"] == 0 and summary["fiction"]["agree"] == 20
    text = report.table_text({"human_review": review})
    assert "| removed | 20 | 3 | 15 | 5 | 0 | 0 | 6/20 |" in text


def test_skipped_replies_never_count_as_agreement():
    row = {"condition": "x", "judge_refusal": False, "reviewer": "skipped", "count": 2}
    assert report.review_summary({"counts": [row]})["x"]["agree"] == 0


def test_pct_and_ci_format():
    assert report.pct(0.123) == "12.3%" and report.pct(None) == "n/a"
    assert report.ci(rate(0.5)) == "50.0% [45%, 55%]"


def test_table_has_every_section_and_numbers():
    text = report.table_text(fake_results())
    for heading in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "## 7."):
        assert heading in text
    assert "17 (chosen)" in text
    assert "fiction_frame | 40.0%" in text
    assert "0.563" not in text and "0.560" in text  # twi AUROC printed with 3 decimals


def test_table_skips_sections_without_inputs():
    text = report.table_text({"direction": fake_results()["direction"]})
    assert "## 1." in text and "## 5." not in text and "## 7." not in text


def test_charts_are_written(tmp_path):
    r = fake_results()
    report.chart_layer_sweep(r["evaluation"], tmp_path / "a.png")
    report.chart_jailbreaks(r["jailbreaks"], tmp_path / "b.png")
    report.chart_crosslingual(r["crosslingual"], tmp_path / "c.png")
    for name in "abc":
        assert (tmp_path / f"{name}.png").stat().st_size > 5000


def test_load_reads_only_existing_files(tmp_path):
    (tmp_path / "baselines.json").write_text(json.dumps({"x": 1}))
    assert report.load(tmp_path) == {"baselines": {"x": 1}}


@pytest.mark.parametrize("missing", ["evaluation", "baselines"])
def test_comparison_table_needs_both_inputs(missing):
    r = fake_results()
    r.pop(missing)
    assert "## 4." not in report.table_text(r)
