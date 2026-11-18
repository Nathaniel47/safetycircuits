"""Find the refusal direction in Gemma-2-2B-it and test it by ablating and adding it.

Run on a GPU (Kaggle/Colab): python -m safetycircuits.refusal.run
Needs HF_TOKEN, and data from `python -m safetycircuits.data.fetch` + `prepare`.

Saves only numbers: refusal verdicts and rates, never model replies.
"""

import argparse
import json
import platform
import subprocess
from collections import defaultdict

import numpy as np
import torch

from safetycircuits.config import DATA_DIR, MODEL_ID, RESULTS_DIR, SEEDS
from safetycircuits.model.hooks import ablating, adding, last_token_residuals
from safetycircuits.model.load import encode, generate_replies, load_model
from safetycircuits.refusal.classify import is_refusal
from safetycircuits.refusal.direction import diff_in_means, random_unit, rank_layers
from safetycircuits.refusal.stats import wilson

BATCH = 8


def load_prompts() -> dict[str, dict[str, str]]:
    """{'harmful'|'harmless'|'xstest': {id: text}} from data/interim/."""
    out = {}
    for name in ("harmful", "harmless", "xstest"):
        with open(DATA_DIR / "interim" / f"{name}.jsonl", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        out[name] = {r["id"]: r["text"] for r in rows}
        if name == "xstest":
            out["xstest_labels"] = {r["id"]: r["label"] for r in rows}
    return out


def residuals(model, tokenizer, prompts: list[str]) -> np.ndarray:
    chunks = []
    for i in range(0, len(prompts), BATCH):
        ids, mask = encode(tokenizer, prompts[i : i + BATCH], model.device)
        chunks.append(last_token_residuals(model, ids, mask).cpu().numpy())
    return np.concatenate(chunks)


def refusal_flags(model, tokenizer, prompts: list[str]) -> list[bool]:
    flags = []
    for i in range(0, len(prompts), BATCH):
        flags += [is_refusal(r) for r in generate_replies(model, tokenizer, prompts[i : i + BATCH])]
    return flags


def rate(flags: list[bool]) -> dict:
    p, lo, hi = wilson(sum(flags), len(flags))
    return {"n": len(flags), "refusals": sum(flags), "rate": p, "ci95": [lo, hi]}


def texts(pool: dict[str, str], ids: list[str], limit: int | None) -> list[str]:
    return [pool[i] for i in ids[:limit]]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="max prompts per set (smoke test)")
    ap.add_argument("--top-layers", type=int, default=3, help="layers tried for ablation on val")
    args = ap.parse_args()
    torch.manual_seed(SEEDS[0])

    data = load_prompts()
    splits = json.loads((DATA_DIR / "splits.json").read_text())
    s_harm, s_less = splits["harmful"], splits["harmless"]
    harm, less = data["harmful"], data["harmless"]
    lim = args.limit

    model, tok = load_model()
    n_layers = len(model.model.layers)

    # 1. Fit the direction on train, rank layers on val.
    acts = {
        (kind, split): residuals(model, tok, texts(pool, ids[split], lim))
        for kind, pool, ids in (("harm", harm, s_harm), ("less", less, s_less))
        for split in ("train", "val")
    }
    ranking = rank_layers(
        acts["harm", "train"], acts["less", "train"], acts["harm", "val"], acts["less", "val"]
    )
    candidates = [r["layer"] for r in ranking[: args.top_layers]]

    # 2. Of the top layers, keep the one whose direction removes the most refusal on val.
    val_harm = texts(harm, s_harm["val"], lim)
    unit_all, norm_all = diff_in_means(acts["harm", "train"], acts["less", "train"])
    val_effect = {}
    for layer in candidates:
        with ablating(model, torch.tensor(unit_all[layer])):
            val_effect[layer] = rate(refusal_flags(model, tok, val_harm))["rate"]
    layer = min(candidates, key=lambda li: val_effect[li])
    direction = torch.tensor(unit_all[layer])
    coefficient = float(norm_all[layer])
    np.save(RESULTS_DIR / "refusal_direction.npy", unit_all[layer])

    # 3. Test split numbers.
    test_harm = texts(harm, s_harm["test"], lim)
    test_less = texts(less, s_less["test"], lim)
    xs_ids = splits["xstest_ids"][:lim]
    xs_safe = [data["xstest"][i] for i in xs_ids if data["xstest_labels"][i] == "safe"]
    xs_unsafe = [data["xstest"][i] for i in xs_ids if data["xstest_labels"][i] == "unsafe"]
    control = torch.tensor(random_unit(unit_all.shape[1], SEEDS[0]), dtype=torch.float32)

    results = defaultdict(dict)
    results["harmful_test"]["baseline"] = rate(refusal_flags(model, tok, test_harm))
    with ablating(model, direction):
        results["harmful_test"]["ablated"] = rate(refusal_flags(model, tok, test_harm))
    with ablating(model, control):
        results["harmful_test"]["ablated_random_direction"] = rate(
            refusal_flags(model, tok, test_harm)
        )
    results["harmless_test"]["baseline"] = rate(refusal_flags(model, tok, test_less))
    with adding(model, layer, direction, coefficient):
        results["harmless_test"]["added"] = rate(refusal_flags(model, tok, test_less))
    with adding(model, layer, control, coefficient):
        results["harmless_test"]["added_random_direction"] = rate(
            refusal_flags(model, tok, test_less)
        )
    # Over-refusal: safe XSTest prompts that look unsafe. Adding the direction should raise it.
    results["xstest_safe"]["baseline"] = rate(refusal_flags(model, tok, xs_safe))
    with adding(model, layer, direction, coefficient):
        results["xstest_safe"]["added"] = rate(refusal_flags(model, tok, xs_safe))
    results["xstest_unsafe"]["baseline"] = rate(refusal_flags(model, tok, xs_unsafe))
    with ablating(model, direction):
        results["xstest_unsafe"]["ablated"] = rate(refusal_flags(model, tok, xs_unsafe))

    out = {
        "meta": {
            "model": MODEL_ID,
            "seed": SEEDS[0],
            "git_commit": git_commit(),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "n_layers": n_layers,
            "position": "last token of the chat-template prompt",
            "limit": lim,
            "dtype": str(next(model.parameters()).dtype),
        },
        "layer_ranking_val": ranking,
        "ablation_on_val_refusal_rate": val_effect,
        "chosen_layer": layer,
        "addition_coefficient": coefficient,
        "results": results,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "refusal_direction.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("chosen_layer", "results")}, indent=1))


if __name__ == "__main__":
    main()
