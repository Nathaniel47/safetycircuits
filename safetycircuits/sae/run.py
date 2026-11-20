"""SAE feature analysis on Gemma-2-2B-it with Gemma Scope SAEs (trained on the base model).

Run on a GPU after the refusal-direction run:  python -m safetycircuits.sae.run
Add --compare-base to also measure the same SAEs on the base model (needs the gemma-2-2b licence).
Saves numbers only: results/sae_features.json.
"""

import argparse
import json

import numpy as np
import torch

from safetycircuits.config import BASE_MODEL_ID, DATA_DIR, RESULTS_DIR, SAE_LAYERS, SEEDS
from safetycircuits.model.hooks import layer_residuals_all
from safetycircuits.model.load import encode, load_model
from safetycircuits.refusal.direction import diff_in_means
from safetycircuits.refusal.run import git_commit, load_prompts, residuals, texts
from safetycircuits.sae.features import (
    decoder_cosines,
    direction_overlap,
    neuronpedia_url,
    top_features,
)
from safetycircuits.sae.load_sae import fvu, load_sae, mean_l0

FVU_PROMPTS = 64  # prompts used to measure how well the SAE reconstructs activations
TOP_K = 20


def all_token_acts(
    model, tok, prompts: list[str], layers, drop_first: bool = False
) -> dict[int, np.ndarray]:
    """Residuals at every token of the chat-formatted prompts, per layer."""
    chunks: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    for i in range(0, len(prompts), 8):
        ids, mask = encode(tok, prompts[i : i + 8], model.device)
        for layer, acts in layer_residuals_all(model, ids, mask, layers, drop_first).items():
            chunks[layer].append(acts.cpu().numpy())
    return {layer: np.concatenate(c) for layer, c in chunks.items()}


def sae_fit(saes, acts_by_layer) -> dict:
    out = {}
    for layer, acts in acts_by_layer.items():
        feats = saes[layer].encode(acts)
        out[str(layer)] = {
            "fvu": fvu(acts, saes[layer].decode(feats)),
            "mean_l0": mean_l0(feats),
            "n_tokens": int(acts.shape[0]),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--compare-base", action="store_true")
    args = ap.parse_args()
    lim = args.limit
    torch.manual_seed(SEEDS[0])

    data = load_prompts()
    splits = json.loads((DATA_DIR / "splits.json").read_text())
    harm, less = data["harmful"], data["harmless"]
    model, tok = load_model()
    saes = {layer: load_sae(layer) for layer in SAE_LAYERS}

    acts = {
        (kind, split): residuals(model, tok, texts(pool, splits[kind][split], lim))
        for kind, pool in (("harmful", harm), ("harmless", less))
        for split in ("train", "val", "test")
    }
    saved_direction = np.load(RESULTS_DIR / "refusal_direction.npy")

    layers_out = {}
    for layer in SAE_LAYERS:
        feats = {k: saes[layer].encode(v[:, layer]) for k, v in acts.items()}
        rows = top_features(
            feats["harmful", "train"],
            feats["harmless", "train"],
            feats["harmful", "val"],
            feats["harmless", "val"],
            TOP_K,
        )
        unit, _ = diff_in_means(acts["harmful", "train"], acts["harmless", "train"])
        direction = unit[layer]
        cos = decoder_cosines(saes[layer].W_dec, [r["feature"] for r in rows], direction)
        for r, c in zip(rows, cos, strict=True):
            r["cos_with_direction"] = float(c)
            r["neuronpedia"] = neuronpedia_url(layer, r["feature"])
        layers_out[str(layer)] = {
            "top_features": rows,
            "direction_overlap": direction_overlap(saes[layer].W_dec, rows, direction),
            "cos_with_saved_direction": float(direction @ saved_direction) if layer == 24 else None,
            "sae_n_features": saes[layer].n_features,
        }

    fvu_prompts = texts(harm, splits["harmful"]["train"], FVU_PROMPTS) + texts(
        less, splits["harmless"]["train"], FVU_PROMPTS
    )

    def fit_both(m) -> dict:
        """SAE fit with and without the BOS token (huge norm, not in the SAEs' training data)."""
        return {
            "with_bos": sae_fit(saes, all_token_acts(m, tok, fvu_prompts, SAE_LAYERS)),
            "without_bos": sae_fit(saes, all_token_acts(m, tok, fvu_prompts, SAE_LAYERS, True)),
        }

    fit = {"instruct": fit_both(model)}
    out = {
        "meta": {
            "model": "google/gemma-2-2b-it",
            "sae_repo": "google/gemma-scope-2b-pt-res (trained on the BASE model)",
            "seed": SEEDS[0],
            "git_commit": git_commit(),
            "torch": torch.__version__,
            "limit": lim,
        },
        "layers": layers_out,
        "sae_fit": fit,
    }
    RESULTS_DIR.mkdir(exist_ok=True)

    def save() -> None:
        text = json.dumps(out, indent=1) + chr(10)
        (RESULTS_DIR / "sae_features.json").write_text(text)

    save()  # the main result is on disk before the optional base-model comparison starts
    if args.compare_base:
        del model
        torch.cuda.empty_cache()
        try:
            # The base model has no chat template, so its own tokenizer cannot format the prompts.
            # The instruct tokenizer has the same vocabulary, so it formats them for both models.
            base, _ = load_model(BASE_MODEL_ID)
            fit["base"] = fit_both(base)
        except Exception as err:  # keep the instruct results even if this part fails
            fit["base_error"] = f"{type(err).__name__}: {err}"[:300]
        save()
    print(json.dumps({"sae_fit": fit}, indent=1))
    for layer, v in layers_out.items():
        print(layer, "direction_overlap", v["direction_overlap"])


if __name__ == "__main__":
    main()
