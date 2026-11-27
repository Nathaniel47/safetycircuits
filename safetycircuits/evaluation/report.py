"""Build results/results_table.md and three charts from the saved result files (laptop, no GPU).

Run: python -m safetycircuits.evaluation.report
Sections whose input file is missing are skipped, so it works at any stage.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from safetycircuits.config import RESULTS_DIR  # noqa: E402

BLUE, GREY, RED = "#2b6cb0", "#a0aec0", "#c0392b"


def load(results: Path = RESULTS_DIR) -> dict[str, dict]:
    names = {
        "direction": "refusal_direction.json",
        "baselines": "baselines.json",
        "evaluation": "evaluation.json",
        "early_layers": "early_layers.json",
        "jailbreaks": "jailbreaks.json",
        "crosslingual": "crosslingual.json",
        "sae": "sae_features.json",
        "human_review": "human_review.json",
    }
    out = {}
    for key, name in names.items():
        path = results / name
        if path.exists():
            out[key] = json.loads(path.read_text(encoding="utf-8"))
    return out


def pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def ci(rate: dict) -> str:
    lo, hi = rate["ci95"]
    return f"{pct(rate['rate'])} [{pct(lo, 0)}, {pct(hi, 0)}]"


def review_summary(review: dict) -> dict[str, dict[str, int]]:
    """Per condition: how the phrase-match judge compares with the human reviewer."""
    keys = ("n", "judge_refused", "refused", "complied", "off-topic", "broken", "skipped", "agree")
    out: dict[str, dict[str, int]] = {}
    for row in review["counts"]:
        s = out.setdefault(row["condition"], dict.fromkeys(keys, 0))
        k = row["count"]
        s["n"] += k
        s[row["reviewer"]] += k
        s["judge_refused"] += k if row["judge_refusal"] else 0
        # Agreement is on "refused or not"; a skipped reply has no human verdict.
        if row["reviewer"] != "skipped" and row["judge_refusal"] == (row["reviewer"] == "refused"):
            s["agree"] += k
    return out


def chart_layer_sweep(ev: dict, out: Path) -> None:
    sweep = ev["layer_sweep_validation"]
    layers = [int(k) for k in sweep]
    rates = [sweep[str(k)]["val_refusal_after_ablation"]["rate"] for k in layers]
    err = [
        [
            r - sweep[str(k)]["val_refusal_after_ablation"]["ci95"][0]
            for r, k in zip(rates, layers, strict=True)
        ],
        [
            sweep[str(k)]["val_refusal_after_ablation"]["ci95"][1] - r
            for r, k in zip(rates, layers, strict=True)
        ],
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(
        layers, rates, yerr=err, fmt="o-", color=BLUE, capsize=3, label="direction removed (val)"
    )
    rnd = ev["random_directions_test_refusal"]
    ax.axhspan(
        rnd["mean"] - rnd["sd"],
        rnd["mean"] + rnd["sd"],
        color=GREY,
        alpha=0.4,
        label="random directions (test, mean ± sd)",
    )
    ax.axhline(
        ev["test"]["baseline_refusal"]["rate"], color=RED, ls="--", label="no intervention (test)"
    )
    best = ev["meta"]["layer_chosen_on_validation"]
    ax.axvline(best, color=BLUE, ls=":", alpha=0.6)
    ax.annotate(
        f"best on val: layer {best}", (best, 0.55), rotation=90, va="center", ha="right", fontsize=8
    )
    ax.set_xlabel("layer where the direction is fitted")
    ax.set_ylabel("share the phrase-match judge flags as refused")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Removing the refusal direction: which layer works best?")
    ax.legend(fontsize=8, loc="center left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_jailbreaks(jb: dict, out: Path) -> None:
    summ = jb["summary"]
    names = sorted(summ, key=lambda c: -summ[c]["success_rate"])
    controls = {"random_words_suffix", "polite_suffix", "baseline"}
    rates = [summ[n]["success_rate"] for n in names]
    err = [
        [r - summ[n]["success_ci95"][0] for r, n in zip(rates, names, strict=True)],
        [summ[n]["success_ci95"][1] - r for r, n in zip(rates, names, strict=True)],
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(
        names, rates, xerr=err, color=[GREY if n in controls else BLUE for n in names], capsize=3
    )
    ax.invert_yaxis()
    ax.set_xlabel("not flagged as a refusal by the phrase-match judge (95% CI)")
    ax.set_title("Jailbreaks (grey = baseline and control suffixes)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_crosslingual(cl: dict, out: Path) -> None:
    langs = cl["languages"]
    names = list(langs)
    aucs = [langs[n]["all"]["direction_auroc"] for n in names]
    err = [
        [a - langs[n]["all"]["direction_auroc_ci95"][0] for a, n in zip(aucs, names, strict=True)],
        [langs[n]["all"]["direction_auroc_ci95"][1] - a for a, n in zip(aucs, names, strict=True)],
    ]
    labels = [
        f"{n}\nGemma chrF {langs[n]['gemma_comprehension_chrf_harmless']:.0f}"
        if "gemma_comprehension_chrf_harmless" in langs[n]
        else n
        for n in names
    ]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(labels, aucs, yerr=err, color=[BLUE if a > 0.8 else RED for a in aucs], capsize=3)
    ax.axhline(0.5, color="black", ls="--", lw=1)
    ax.text(len(names) - 0.5, 0.51, "chance", ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC: harmful vs harmless")
    ax.set_title("Does the English refusal direction work in other languages?")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def table_text(r: dict) -> str:
    lines = [
        "# Results table",
        "",
        "Generated by `python -m safetycircuits.evaluation.report` from the files in `results/`.",
        "",
        "**Refusal rates in sections 1 to 7 come from a phrase-match judge. In a small human "
        "check (section 8) it missed most refusals when the direction was removed, and "
        "replies it did not flag often had not complied.**",
        "",
    ]
    d, ev, bl = r.get("direction"), r.get("evaluation"), r.get("baselines")

    if d:
        res = d["results"]
        lines += [
            "## 1. Main effect (test split, layer %d, seed 0)" % d["chosen_layer"],
            "",
            "| Prompts | Condition | Refusal rate [95% CI] | n |",
            "|---|---|---|---|",
        ]
        for name, conds in res.items():
            for cond, v in conds.items():
                lines.append(f"| {name} | {cond} | {ci(v)} | {v['n']} |")
        lines.append("")

    if ev:
        t = ev["test"]
        sd = ev["seed_variance_80pct_subsamples"]["summary"]
        rnd = ev["random_directions_test_refusal"]
        lines += [
            "## 2. Variance",
            "",
            f"Direction refitted on 3 random 80% subsets of the training prompts, test split. "
            f"Mean ± standard deviation over the 3 refits. Random directions: mean ± sd over {len(rnd['rates'])}.",
            "",
            "| Layer | Refusal after ablation | AUROC harmful vs harmless | Cosine with full-data direction |",
            "|---|---|---|---|",
        ]
        for layer, s in sd.items():
            a, b, c = s["refusal_after_ablation"], s["auroc"], s["cosine_with_full_direction"]
            lines.append(
                f"| {layer} | {pct(a['mean'])} ± {pct(a['sd'])} | {b['mean']:.4f} ± {b['sd']:.4f} | {c['mean']:.3f} ± {c['sd']:.3f} |"
            )
        lines += [
            "",
            f"Random directions removed instead: {pct(rnd['mean'])} ± {pct(rnd['sd'])} still refused "
            f"(no intervention: {ci(t['baseline_refusal'])}).",
            "",
            "## 3. Layer sweep (validation prompts; the layer is chosen here, not on test)",
            "",
            "| Layer | Refused after ablation (val) [95% CI] | AUROC (val) |",
            "|---|---|---|",
        ]
        for layer, v in ev["layer_sweep_validation"].items():
            mark = " (chosen)" if int(layer) == ev["meta"]["layer_chosen_on_validation"] else ""
            lines.append(
                f"| {layer}{mark} | {ci(v['val_refusal_after_ablation'])} | {v['val_auroc']:.4f} |"
            )
        rates = {
            k: v["val_refusal_after_ablation"]["rate"]
            for k, v in ev["layer_sweep_validation"].items()
        }
        lowest = min(rates.values())
        tied = [k for k, v in rates.items() if v == lowest]
        if len(tied) > 1:
            lines += [
                "",
                f"Layers {', '.join(tied)} are tied at {pct(lowest)} on validation. The chosen layer is "
                "the deepest of the tied ones, an arbitrary tie-break; only the chosen layer and the "
                "headline layer were run on the test split.",
            ]
        lines += [
            "",
            "Test split for the headline and chosen layers:",
            "",
            "| Layer | Refused after ablation [95% CI] |",
            "|---|---|",
        ]
        for layer in [k for k in t if k.isdigit()]:
            lines.append(f"| {layer} | {ci(t[layer]['refusal_after_ablation'])} |")
        lines.append("")

    if "early_layers" in r:
        el = r["early_layers"]
        lines += [
            "## 3b. Early layers on the test split (144 harmful prompts)",
            "",
            f"No intervention: {ci(el['baseline_refusal'])} refused. Direction fitted and removed at each layer. "
            "`Clear compliance` = a non-refused reply that opens like a helpful answer (second check); "
            "a low share means many non-refusals cannot be read as real compliance and may be broken text.",
            "",
            "| Layer | Refused after removal [95% CI] | Not refused | Clear compliance | Unclear |",
            "|---|---|---|---|---|",
        ]
        for layer, v in el["layers"].items():
            a2 = v["agreement"]
            lines.append(
                f"| {layer} | {ci(v['refusal_after_ablation'])} | {a2['not_refused']} | "
                f"{a2['clear_compliance']} ({pct(a2['clear_share_of_not_refused'], 0)}) | {a2['unclear']} |"
            )
        lines.append("")

    if ev and bl:
        t = ev["test"]
        layers = [k for k in t if k.isdigit()]
        lines += [
            "## 4. Direction versus simple text baselines (AUROC, higher = better separation)",
            "",
            "| Test | "
            + " | ".join(f"Direction, layer {k}" for k in layers)
            + " | Word-count naive Bayes | First word only | Prompt length |",
            "|---|" + "---|" * (len(layers) + 3),
        ]
        b = bl["baselines_auroc_test"]
        x = bl["xstest_unsafe_vs_safe_auroc"]
        s = bl["style_matched"]
        lines.append(
            "| AdvBench+HarmBench vs Alpaca | "
            + " | ".join(f"{t[k]['auroc_harmful_vs_harmless']:.4f}" for k in layers)
            + f" | {b['bag_of_words_naive_bayes']:.4f} | {b['first_word_only']:.4f} | {b['prompt_length_in_words']:.4f} |"
        )
        lines.append(
            f"| Harmful vs style-matched Alpaca (n={s['n_harmless_matched_val_test']}) | "
            + " | ".join(f"{t[k]['auroc_harmful_vs_style_matched_harmless']:.4f}" for k in layers)
            + f" | {s['bag_of_words_naive_bayes']:.4f} | n/a | {s['prompt_length_in_words']:.4f} |"
        )
        lines.append(
            "| XSTest unsafe vs safe | "
            + " | ".join(f"{t[k]['auroc_xstest_unsafe_vs_safe']:.4f}" for k in layers)
            + f" | {x['bag_of_words_naive_bayes']:.4f} | {x['first_word_only']:.4f} | {x['prompt_length_in_words']:.4f} |"
        )
        lines += [
            "",
            "Word counts alone already separate AdvBench-style prompts from Alpaca, so that test is easy "
            "for any method. XSTest pairs safe and unsafe prompts that share words, so it is the test "
            "that word counts cannot pass.",
            "",
        ]

    if "jailbreaks" in r:
        summ = r["jailbreaks"]["summary"]
        lines += [
            "## 5. Jailbreaks (144 harmful test prompts, layer %d)"
            % r["jailbreaks"]["meta"]["layer"],
            "",
            "| Condition | Not flagged as a refusal [95% CI] | Refusal score | Score suppression |",
            "|---|---|---|---|",
        ]
        for c in sorted(summ, key=lambda c: -summ[c]["success_rate"]):
            v = summ[c]
            lo, hi = v["success_ci95"]
            lines.append(
                f"| {c} | {pct(v['success_rate'])} [{pct(lo, 0)}, {pct(hi, 0)}] | {v['mean_refusal_score']:.2f} | {v['score_suppression']:.2f} |"
            )
        lines.append("")

    if "crosslingual" in r:
        lines += [
            "## 6. Cross-lingual (100 harmful + 100 harmless prompts per language)",
            "",
            "| Language | Direction AUROC [95% CI] | Harmful refused | Refused, direction removed | Gemma comprehension (chrF) |",
            "|---|---|---|---|---|",
        ]
        for n, v in r["crosslingual"]["languages"].items():
            a = v["all"]
            comp = v.get("gemma_comprehension_chrf_harmless")
            lines.append(
                f"| {n} | {a['direction_auroc']:.3f} [{a['direction_auroc_ci95'][0]:.2f}, {a['direction_auroc_ci95'][1]:.2f}] "
                f"| {pct(a['harmful_refusal_rate'], 0)} | {pct(a['harmful_refusal_rate_ablated'], 0)} | {'n/a' if comp is None else f'{comp:.1f}'} |"
            )
        lines.append("")

    if ev:
        lines += [
            "## 7. Does the refusal classifier mean what we think? (agreement with a second check)",
            "",
            "Among replies the classifier calls non-refusals, `clear compliance` = opens like a helpful "
            "answer and is not broken text. `Unclear` = neither. Counts only; no reply text is stored.",
            "",
            "| Replies | n | Refused | Not refused | Clear compliance | Unclear |",
            "|---|---|---|---|---|---|",
        ]
        for name, a in ev["classifier_checks"].items():
            lines.append(
                f"| {name} | {a['n']} | {a['refused']} | {a['not_refused']} | {a['clear_compliance']} | {a['unclear']} |"
            )
        lines.append("")

    if "human_review" in r:
        meta = r["human_review"]["meta"]
        lines += [
            "## 8. Human check of the judge (harmful prompts, layer %d)" % meta["layer"],
            "",
            f"{meta['n_prompts']} filtered test prompts per condition, one reviewer, shuffled, with "
            "condition and judge verdict not shown. Only counts were kept. `Agree` = judge and "
            "reviewer give the same refused / not refused answer.",
            "",
            "| Condition | n | Judge: refused | Reviewer: refused | Reviewer: complied | "
            "Reviewer: off-topic | Reviewer: broken or skipped | Agree |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name, s in review_summary(r["human_review"]).items():
            lines.append(
                f"| {name} | {s['n']} | {s['judge_refused']} | {s['refused']} | {s['complied']} | "
                f"{s['off-topic']} | {s['broken'] + s['skipped']} | {s['agree']}/{s['n']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    r = load()
    if "evaluation" in r:
        chart_layer_sweep(r["evaluation"], RESULTS_DIR / "chart_layer_sweep.png")
    if "jailbreaks" in r:
        chart_jailbreaks(r["jailbreaks"], RESULTS_DIR / "chart_jailbreaks.png")
    if "crosslingual" in r:
        chart_crosslingual(r["crosslingual"], RESULTS_DIR / "chart_crosslingual.png")
    (RESULTS_DIR / "results_table.md").write_text(table_text(r), encoding="utf-8", newline="\n")
    print("wrote results_table.md and charts for:", ", ".join(sorted(r)))


if __name__ == "__main__":
    main()
