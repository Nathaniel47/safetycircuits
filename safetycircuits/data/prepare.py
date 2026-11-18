"""Clean the raw datasets and write splits.

Writes:
  data/interim/*.jsonl   prompt text (gitignored: rebuild with fetch + prepare)
  data/splits.json       ids only (committed)
  data/quality.json      counts of problems found (used by the data report)

Run: python -m safetycircuits.data.prepare
"""

import csv
import json
import random
import re

from safetycircuits.config import DATA_DIR, LANGUAGES, SEEDS
from safetycircuits.data.splits import split_ids

RAW = DATA_DIR / "raw"
INTERIM = DATA_DIR / "interim"
SEED = SEEDS[0]
N_HARMLESS = 800  # harmless pool, about the size of the harmful pool
N_TRANSLATE = 100  # harmful test prompts that get translated
N_FLORES_CONTROL = 100
SPLIT_FRACTIONS = {"train": 0.6, "val": 0.2, "test": 0.2}


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def key(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", normalise(text).lower())


def load_advbench() -> list[dict]:
    with open(RAW / "advbench.csv", encoding="utf-8", newline="") as f:
        return [
            {"id": f"advbench-{i:04d}", "source": "advbench", "text": normalise(r["goal"])}
            for i, r in enumerate(csv.DictReader(f))
        ]


def load_harmbench() -> tuple[list[dict], int]:
    """Standard behaviours only: the others need a context string or are copyright tasks."""
    with open(RAW / "harmbench.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    standard = [r for r in rows if r["FunctionalCategory"] == "standard"]
    items = [
        {
            "id": f"harmbench-{r['BehaviorID']}",
            "source": "harmbench",
            "category": r["SemanticCategory"],
            "text": normalise(r["Behavior"]),
        }
        for r in standard
    ]
    return items, len(rows) - len(standard)


def load_xstest() -> list[dict]:
    with open(RAW / "xstest.csv", encoding="utf-8", newline="") as f:
        return [
            {
                "id": f"xstest-{int(r['id']):04d}",
                "source": "xstest",
                "label": r["label"],
                "type": r["type"],
                "text": normalise(r["prompt"]),
            }
            for r in csv.DictReader(f)
        ]


def load_alpaca(rng: random.Random) -> tuple[list[dict], dict]:
    data = json.loads((RAW / "alpaca.json").read_text(encoding="utf-8"))
    stats = {"total": len(data), "with_input": 0, "too_short_or_long": 0, "duplicates": 0}
    seen, pool = set(), []
    for i, r in enumerate(data):
        text = normalise(r["instruction"])
        if r["input"].strip():
            stats["with_input"] += 1  # needs extra context, so not a standalone prompt
        elif not 20 <= len(text) <= 200:
            stats["too_short_or_long"] += 1
        elif key(text) in seen:
            stats["duplicates"] += 1
        else:
            seen.add(key(text))
            pool.append({"id": f"alpaca-{i:05d}", "source": "alpaca", "text": text})
    stats["eligible"] = len(pool)
    rng.shuffle(pool)
    return sorted(pool[:N_HARMLESS], key=lambda r: r["id"]), stats


def dedupe(items: list[dict], seen: set[str]) -> tuple[list[dict], int]:
    kept, dropped = [], 0
    for item in items:
        k = key(item["text"])
        if k in seen:
            dropped += 1
        else:
            seen.add(k)
            kept.append(item)
    return kept, dropped


def read_lines(path) -> list[str]:
    return path.read_text(encoding="utf-8").split("\n")[:-1] if path.exists() else []


def flores(rng: random.Random) -> dict:
    out: dict = {"lines": {}, "empty": {}}
    for code in ["eng_Latn", *LANGUAGES.values()]:
        lines = read_lines(RAW / "flores" / f"{code}.devtest")
        out["lines"][code] = len(lines)
        out["empty"][code] = sum(1 for line in lines if not line.strip())
    n = min(out["lines"].values())
    out["control_indices"] = sorted(rng.sample(range(n), min(N_FLORES_CONTROL, n)))
    return out


def write_jsonl(name: str, items: list[dict]) -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    with open(INTERIM / name, "w", encoding="utf-8", newline="\n") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    rng = random.Random(SEED)
    adv = load_advbench()
    harm, harmbench_non_standard = load_harmbench()
    seen: set[str] = set()
    adv, adv_dup = dedupe(adv, seen)
    harm, harm_dup = dedupe(harm, seen)  # includes overlap with AdvBench
    harmful = adv + harm
    harmless, alpaca_stats = load_alpaca(rng)
    xstest = load_xstest()

    harmful_splits = split_ids(
        {s: [i["id"] for i in harmful if i["source"] == s] for s in ("advbench", "harmbench")},
        SPLIT_FRACTIONS,
        SEED,
    )
    harmless_splits = split_ids({"alpaca": [i["id"] for i in harmless]}, SPLIT_FRACTIONS, SEED)
    test_ids = sorted(harmful_splits["test"])
    translate_ids = sorted(rng.sample(test_ids, min(N_TRANSLATE, len(test_ids))))
    fl = flores(rng)

    write_jsonl("harmful.jsonl", harmful)
    write_jsonl("harmless.jsonl", harmless)
    write_jsonl("xstest.jsonl", xstest)
    splits = {
        "seed": SEED,
        "fractions": SPLIT_FRACTIONS,
        "harmful": harmful_splits,
        "harmless": harmless_splits,
        "translate_ids": translate_ids,
        "xstest_ids": sorted(i["id"] for i in xstest),
        "flores_control_indices": fl.pop("control_indices"),
    }
    (DATA_DIR / "splits.json").write_text(json.dumps(splits, indent=1) + "\n")
    quality = {
        "advbench_duplicates_removed": adv_dup,
        "harmbench_duplicates_or_advbench_overlap_removed": harm_dup,
        "harmbench_non_standard_excluded": harmbench_non_standard,
        "alpaca": alpaca_stats,
        "flores": fl,
    }
    (DATA_DIR / "quality.json").write_text(json.dumps(quality, indent=1) + "\n")
    print(
        f"harmful={len(harmful)} harmless={len(harmless)} xstest={len(xstest)} "
        f"translate={len(translate_ids)}"
    )


if __name__ == "__main__":
    main()
