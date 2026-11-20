"""Evaluation on a GPU: layer sweep, seed variance, random controls, classifier checks.

Run after the earlier steps:  python -m safetycircuits.evaluation.run
Saves numbers only (results/evaluation.json). The one text file it writes with model replies,
results/classifier_review_benign.csv, holds replies to harmless prompts only.
"""

import argparse
import csv
import json

import numpy as np
import torch

from safetycircuits.config import DATA_DIR, RESULTS_DIR, SEEDS
from safetycircuits.evaluation.baselines import style_matched, style_words, tokens
from safetycircuits.evaluation.checks import agreement, mean_sd, starts_affirmative, subsample
from safetycircuits.jailbreaks.study import apply
from safetycircuits.model.hooks import ablating
from safetycircuits.model.load import generate_replies, load_model
from safetycircuits.refusal.classify import is_refusal, looks_degenerate
from safetycircuits.refusal.direction import auroc, diff_in_means, random_unit
from safetycircuits.refusal.run import BATCH, git_commit, load_prompts, residuals
from safetycircuits.refusal.stats import wilson

LAYERS = [8, 10, 12, 14, 16, 17, 18, 20, 22, 24, 25]
HEADLINE_LAYER = 24
N_RANDOM = 5
SUBSAMPLE_FRACTION = 0.8
N_REVIEW = 25


def rate(flags) -> dict:
    flags = np.asarray(flags, dtype=bool)
    p, lo, hi = wilson(int(flags.sum()), len(flags))
    return {"n": len(flags), "refusals": int(flags.sum()), "rate": p, "ci95": [lo, hi]}


def gen(model, tok, prompts: list[str], direction=None) -> dict:
    """Generate replies (optionally with a direction removed) and keep only flags, not text."""
    refused, degenerate, affirm = [], [], []
    for i in range(0, len(prompts), BATCH):
        batch = prompts[i : i + BATCH]
        if direction is None:
            replies = generate_replies(model, tok, batch)
        else:
            with ablating(model, direction):
                replies = generate_replies(model, tok, batch)
        for r in replies:
            refused.append(is_refusal(r))
            degenerate.append(looks_degenerate(r))
            affirm.append(starts_affirmative(r))
    return {k: np.array(v) for k, v in (("r", refused), ("d", degenerate), ("a", affirm))}


def scores(acts: np.ndarray, unit: np.ndarray, layer: int) -> np.ndarray:
    return acts[:, layer] @ unit[layer]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    lim = ap.parse_args().limit
    torch.manual_seed(SEEDS[0])

    data = load_prompts()
    splits = json.loads((DATA_DIR / "splits.json").read_text())

    def texts(kind: str, split: str) -> list[str]:
        pool = data[kind]
        return [pool[i] for i in splits[kind][split]][:lim]

    model, tok = load_model()
    R = lambda prompts: residuals(model, tok, prompts)  # noqa: E731
    h = {s: texts("harmful", s) for s in ("train", "val", "test")}
    la = {s: texts("harmless", s) for s in ("train", "val", "test")}
    acts = {("h", s): R(h[s]) for s in h} | {("l", s): R(la[s]) for s in la}

    xs = {
        k: [t for i, t in data["xstest"].items() if data["xstest_labels"][i] == k][:lim]
        for k in ("safe", "unsafe")
    }
    xs_acts = {k: R(v) for k, v in xs.items()}
    words = style_words(h["train"])
    harmless_rows = [{"id": i, "text": t} for i, t in data["harmless"].items()]
    non_train = set(splits["harmless"]["val"]) | set(splits["harmless"]["test"])
    matched = [
        r["text"] for r in style_matched([r for r in harmless_rows if r["id"] in non_train], words)
    ][:lim]
    matched_acts = R(matched)

    unit_full, norm_full = diff_in_means(acts["h", "train"], acts["l", "train"])
    unit_t = lambda layer: torch.tensor(unit_full[layer])  # noqa: E731

    # 1. Layer sweep. Layers are chosen on the validation prompts, never on the test prompts.
    sweep = {}
    for layer in LAYERS:
        flags = gen(model, tok, h["val"], unit_t(layer))["r"]
        sweep[layer] = {
            "val_refusal_after_ablation": rate(flags),
            "val_auroc": auroc(
                scores(acts["h", "val"], unit_full, layer),
                scores(acts["l", "val"], unit_full, layer),
            ),
        }
    best = min(LAYERS, key=lambda li: (sweep[li]["val_refusal_after_ablation"]["rate"], -li))
    layers_to_report = sorted({HEADLINE_LAYER, best})

    # 2. Test numbers for the headline and best layers, plus harder discrimination tests.
    base = gen(model, tok, h["test"])
    test = {"baseline_refusal": rate(base["r"])}
    for layer in layers_to_report:
        abl = gen(model, tok, h["test"], unit_t(layer))
        test[str(layer)] = {
            "refusal_after_ablation": rate(abl["r"]),
            "auroc_harmful_vs_harmless": auroc(
                scores(acts["h", "test"], unit_full, layer),
                scores(acts["l", "test"], unit_full, layer),
            ),
            "auroc_xstest_unsafe_vs_safe": auroc(
                scores(xs_acts["unsafe"], unit_full, layer),
                scores(xs_acts["safe"], unit_full, layer),
            ),
            "auroc_harmful_vs_style_matched_harmless": auroc(
                scores(acts["h", "test"], unit_full, layer), scores(matched_acts, unit_full, layer)
            ),
            "n_style_matched": len(matched),
        }
        if layer == HEADLINE_LAYER:
            ablated_headline = abl

    # 3. Seed variance: refit the direction on random 80% subsets of the training prompts.
    seeds_out = {str(layer): [] for layer in layers_to_report}
    for seed in range(3):
        idx_h = subsample(list(range(len(h["train"]))), SUBSAMPLE_FRACTION, seed)
        idx_l = subsample(list(range(len(la["train"]))), SUBSAMPLE_FRACTION, seed)
        unit_s, _ = diff_in_means(acts["h", "train"][idx_h], acts["l", "train"][idx_l])
        for layer in layers_to_report:
            flags = gen(model, tok, h["test"], torch.tensor(unit_s[layer]))["r"]
            seeds_out[str(layer)].append(
                {
                    "seed": seed,
                    "refusal_after_ablation": float(flags.mean()),
                    "auroc": auroc(
                        scores(acts["h", "test"], unit_s, layer),
                        scores(acts["l", "test"], unit_s, layer),
                    ),
                    "cosine_with_full_direction": float(unit_s[layer] @ unit_full[layer]),
                }
            )
    seed_summary = {
        layer: {
            k: dict(zip(("mean", "sd"), mean_sd([r[k] for r in rows]), strict=True))
            for k in ("refusal_after_ablation", "auroc", "cosine_with_full_direction")
        }
        for layer, rows in seeds_out.items()
    }

    # 4. Several random directions (not one), removed at every layer.
    random_rates = [
        float(
            gen(model, tok, h["test"], torch.tensor(random_unit(unit_full.shape[1], 100 + s)))[
                "r"
            ].mean()
        )
        for s in range(N_RANDOM)
    ]

    # 5. Classifier checks. Replies become flags; only harmless-prompt replies are written out.
    jb = gen(model, tok, [apply("fiction_frame", p) for p in h["test"]])
    checks = {
        "harmful_baseline": agreement(base["r"], base["a"], base["d"]),
        f"harmful_ablated_layer_{HEADLINE_LAYER}": agreement(
            ablated_headline["r"], ablated_headline["a"], ablated_headline["d"]
        ),
        "harmful_fiction_frame_jailbreak": agreement(jb["r"], jb["a"], jb["d"]),
    }
    review_prompts = la["test"][:N_REVIEW] + xs["safe"][:N_REVIEW]
    review_replies = []
    for i in range(0, len(review_prompts), BATCH):
        review_replies += generate_replies(model, tok, review_prompts[i : i + BATCH])
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "classifier_review_benign.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["prompt", "reply_start", "classifier_says_refusal", "your_verdict_refusal_yes_no"]
        )
        for p, r in zip(review_prompts, review_replies, strict=True):
            w.writerow([p, " ".join(r.split())[:110], is_refusal(r), ""])

    # 6. A file for a Twi reader (harmless prompts only), written outside the repo's tracked files.
    tr_path = DATA_DIR / "interim" / "translations.jsonl"
    if tr_path.exists():
        rows = [json.loads(line) for line in open(tr_path, encoding="utf-8")]
        twi = [r for r in rows if r["lang"] == "twi" and r["kind"] == "harmless"][:30]
        english = {i: data["harmless"][i] for i in splits["harmless"]["test"]}
        with open(DATA_DIR / "interim" / "twi_review.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "english_original",
                    "twi_translation",
                    "rating_good_ok_wrong",
                    "better_twi_if_wrong",
                ]
            )
            for r in twi:
                w.writerow([english.get(r["id"], ""), r["text"], "", ""])

    out = {
        "meta": {
            "seed": SEEDS[0],
            "git_commit": git_commit(),
            "torch": torch.__version__,
            "limit": lim,
            "layers_swept": LAYERS,
            "layer_chosen_on_validation": best,
            "headline_layer": HEADLINE_LAYER,
            "subsample_fraction": SUBSAMPLE_FRACTION,
            "style_words": words,
            "n_tokens_check": sum(len(tokens(t)) for t in matched),
        },
        "layer_sweep_validation": {str(k): v for k, v in sweep.items()},
        "test": test,
        "seed_variance_80pct_subsamples": {"runs": seeds_out, "summary": seed_summary},
        "random_directions_test_refusal": {
            "rates": random_rates,
            **dict(zip(("mean", "sd"), mean_sd(random_rates), strict=True)),
        },
        "classifier_checks": checks,
        "xstest_n": {k: len(v) for k, v in xs.items()},
    }
    (RESULTS_DIR / "evaluation.json").write_text(json.dumps(out, indent=1) + "\n")
    print(
        json.dumps(
            {
                "best": best,
                "test": test,
                "seed_summary": seed_summary,
                "random": out["random_directions_test_refusal"],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
