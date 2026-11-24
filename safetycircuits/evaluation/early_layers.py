"""Test-split check of the early layers that tied at 0% on validation (and the headline layers).

Run on a GPU after the main evaluation:  python -m safetycircuits.evaluation.early_layers
For each layer it removes the direction fitted at that layer and measures the refusal rate on
the 144 harmful test prompts, and how many of the non-refusals look like real compliance
(opens like a helpful answer, not broken text). A layer is only better if its replies are not
broken, so both numbers matter. Saves numbers only: results/early_layers.json
"""

import argparse
import json

import numpy as np
import torch

from safetycircuits.config import DATA_DIR, RESULTS_DIR, SEEDS
from safetycircuits.evaluation.checks import agreement
from safetycircuits.evaluation.run import gen, rate
from safetycircuits.model.load import load_model
from safetycircuits.refusal.direction import diff_in_means
from safetycircuits.refusal.run import git_commit, load_prompts, residuals, texts

LAYERS = [8, 10, 12, 14, 24, 25]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    lim = ap.parse_args().limit
    torch.manual_seed(SEEDS[0])

    data = load_prompts()
    splits = json.loads((DATA_DIR / "splits.json").read_text())
    h_train = texts(data["harmful"], splits["harmful"]["train"], lim)
    l_train = texts(data["harmless"], splits["harmless"]["train"], lim)
    h_test = texts(data["harmful"], splits["harmful"]["test"], lim)

    model, tok = load_model()
    unit, _ = diff_in_means(residuals(model, tok, h_train), residuals(model, tok, l_train))

    base = gen(model, tok, h_test)
    out = {"baseline_refusal": rate(base["r"]), "layers": {}}
    for layer in LAYERS:
        res = gen(model, tok, h_test, torch.tensor(unit[layer]))
        out["layers"][str(layer)] = {
            "refusal_after_ablation": rate(res["r"]),
            "agreement": agreement(res["r"], res["a"], res["d"]),
            "degenerate_share": float(np.mean(res["d"])),
        }
        print(layer, out["layers"][str(layer)]["refusal_after_ablation"]["rate"])
    out["meta"] = {"seed": SEEDS[0], "git_commit": git_commit(), "limit": lim}
    (RESULTS_DIR / "early_layers.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
