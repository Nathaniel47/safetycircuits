"""Simple text-only baselines: can harmful prompts be told apart from harmless ones without a model?

If a trivial classifier on word counts scores near the refusal direction, the direction may be
picking up the style of the datasets (short imperative AdvBench prompts versus varied Alpaca
prompts) and not harmfulness. Pure Python/numpy, so it runs in CI and on the laptop.

Run: python -m safetycircuits.evaluation.baselines   (needs data from fetch + prepare)
"""

import json
import math
import re
from collections import Counter

import numpy as np

from safetycircuits.config import DATA_DIR, RESULTS_DIR
from safetycircuits.refusal.direction import auroc

N_STYLE_VERBS = 5


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


class NaiveBayes:
    """Multinomial naive Bayes over word counts, with add-one smoothing. Score = log-odds."""

    def fit(self, positive: list[str], negative: list[str]) -> "NaiveBayes":
        self.pos = Counter(w for t in positive for w in tokens(t))
        self.neg = Counter(w for t in negative for w in tokens(t))
        self.vocab = len(set(self.pos) | set(self.neg))
        self.n_pos, self.n_neg = sum(self.pos.values()), sum(self.neg.values())
        self.prior = math.log(len(positive) / len(negative))
        return self

    def score(self, text: str) -> float:
        total = self.prior
        for w in tokens(text):
            p = (self.pos[w] + 1) / (self.n_pos + self.vocab)
            n = (self.neg[w] + 1) / (self.n_neg + self.vocab)
            total += math.log(p / n)
        return total


def auroc_of(score_fn, positive: list[str], negative: list[str]) -> float:
    return auroc(
        np.array([score_fn(t) for t in positive]), np.array([score_fn(t) for t in negative])
    )


def first_word_scorer(positive: list[str], negative: list[str]):
    """Score a prompt by how much more often its first word starts harmful than harmless prompts."""
    pos = Counter(tokens(t)[0] for t in positive if tokens(t))
    neg = Counter(tokens(t)[0] for t in negative if tokens(t))

    def score(text: str) -> float:
        words = tokens(text)
        w = words[0] if words else ""
        return math.log((pos[w] + 1) / (len(positive) + 1)) - math.log(
            (neg[w] + 1) / (len(negative) + 1)
        )

    return score


def style_words(harmful_train: list[str], k: int = N_STYLE_VERBS) -> list[str]:
    """The k most common first words of the harmful training prompts (write, create, give...)."""
    firsts = Counter(tokens(t)[0] for t in harmful_train if tokens(t))
    return [w for w, _ in firsts.most_common(k)]


def style_matched(harmless: list[dict], words: list[str]) -> list[dict]:
    """Harmless prompts that open with the same words as the harmful ones: same style, no harm."""
    return [h for h in harmless if tokens(h["text"]) and tokens(h["text"])[0] in words]


def load_interim(name: str) -> list[dict]:
    with open(DATA_DIR / "interim" / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    harmful, harmless = load_interim("harmful"), load_interim("harmless")
    splits = json.loads((DATA_DIR / "splits.json").read_text())
    by_id = {r["id"]: r["text"] for r in harmful + harmless}

    def texts(group: str, split: str) -> list[str]:
        return [by_id[i] for i in splits[group][split]]

    h_train, l_train = texts("harmful", "train"), texts("harmless", "train")
    h_test, l_test = texts("harmful", "test"), texts("harmless", "test")
    nb = NaiveBayes().fit(h_train, l_train)
    first = first_word_scorer(h_train, l_train)
    words = style_words(h_train)
    matched = style_matched(
        [r for r in harmless if r["id"] not in set(splits["harmless"]["train"])], words
    )

    xstest = load_interim("xstest")
    xs_unsafe = [r["text"] for r in xstest if r["label"] == "unsafe"]
    xs_safe = [r["text"] for r in xstest if r["label"] == "safe"]

    out = {
        "n_test": {"harmful": len(h_test), "harmless": len(l_test)},
        "baselines_auroc_test": {
            "bag_of_words_naive_bayes": auroc_of(nb.score, h_test, l_test),
            "first_word_only": auroc_of(first, h_test, l_test),
            "prompt_length_in_words": auroc_of(lambda t: len(tokens(t)), h_test, l_test),
        },
        # XSTest pairs safe and unsafe prompts that share words, so it is the harder test.
        "xstest_unsafe_vs_safe_auroc": {
            "bag_of_words_naive_bayes": auroc_of(nb.score, xs_unsafe, xs_safe),
            "first_word_only": auroc_of(first, xs_unsafe, xs_safe),
            "prompt_length_in_words": auroc_of(lambda t: len(tokens(t)), xs_unsafe, xs_safe),
            "n_unsafe": len(xs_unsafe),
            "n_safe": len(xs_safe),
        },
        "style_matched": {
            "first_words": words,
            "n_harmless_matched_val_test": len(matched),
            "bag_of_words_naive_bayes": auroc_of(nb.score, h_test, [m["text"] for m in matched])
            if matched
            else None,
            "prompt_length_in_words": auroc_of(
                lambda t: len(tokens(t)), h_test, [m["text"] for m in matched]
            )
            if matched
            else None,
        },
    }
    (RESULTS_DIR / "baselines.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
