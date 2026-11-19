"""Jailbreak study: 5 template and 3 suffix jailbreaks on the harmful test prompts.

Run on a GPU after the refusal-direction and SAE runs:  python -m safetycircuits.jailbreaks.run
Saves per-condition numbers and per-prompt flags and scores (never replies): results/jailbreaks.json
"""

import argparse
import json

import numpy as np
import torch

from safetycircuits.config import DATA_DIR, RESULTS_DIR, SAE_LAYERS, SEEDS
from safetycircuits.jailbreaks.study import (
    BASELINE,
    CONDITIONS,
    KIND,
    apply,
    pooled_success_vs_refused_features,
    rank_conditions,
    summarise,
)
from safetycircuits.model.load import generate_replies, load_model
from safetycircuits.refusal.classify import is_refusal, looks_degenerate
from safetycircuits.refusal.direction import calibrated_score
from safetycircuits.refusal.run import BATCH, git_commit, load_prompts, residuals, texts
from safetycircuits.sae.features import refusal_feature_score
from safetycircuits.sae.load_sae import load_sae


def judge(model, tok, prompts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    refused, degenerate = [], []
    for i in range(0, len(prompts), BATCH):
        for reply in generate_replies(model, tok, prompts[i : i + BATCH]):
            refused.append(is_refusal(reply))
            degenerate.append(looks_degenerate(reply))
    return np.array(refused), np.array(degenerate)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    lim = args.limit
    torch.manual_seed(SEEDS[0])

    ref = json.loads((RESULTS_DIR / "refusal_direction.json").read_text())
    layer = ref["chosen_layer"]
    unit = np.load(RESULTS_DIR / "refusal_direction.npy")
    sae_json_path = RESULTS_DIR / "sae_features.json"
    sae_rows = (
        json.loads(sae_json_path.read_text())["layers"].get(str(layer), {}).get("top_features")
        if sae_json_path.exists() and layer in SAE_LAYERS
        else None
    )

    data = load_prompts()
    splits = json.loads((DATA_DIR / "splits.json").read_text())
    harm_test = texts(data["harmful"], splits["harmful"]["test"], lim)
    model, tok = load_model()

    # Calibrate the score on train prompts: 0 = typical harmless, 1 = typical harmful.
    harmful_train = residuals(model, tok, texts(data["harmful"], splits["harmful"]["train"], 128))
    harmless_train = residuals(
        model, tok, texts(data["harmless"], splits["harmless"]["train"], 128)
    )
    h_mean = float((harmful_train[:, layer] @ unit).mean())
    l_mean = float((harmless_train[:, layer] @ unit).mean())

    sae, ids, scale = None, None, None
    if sae_rows:
        sae = load_sae(layer)
        ids = [r["feature"] for r in sae_rows]
        scale = sae.encode(harmful_train[:, layer])[:, ids].mean(axis=0)

    per_cond, pooled_feats, pooled_success = {}, [], []
    for cond in CONDITIONS:
        prompts = [apply(cond, p) for p in harm_test]
        acts = residuals(model, tok, prompts)[:, layer]
        score = calibrated_score(acts, unit, l_mean, h_mean)
        refused, degenerate = judge(model, tok, prompts)
        feats = sae.encode(acts) if sae is not None else None
        fscore = refusal_feature_score(feats, ids, scale) if feats is not None else None
        per_cond[cond] = {
            "kind": KIND[cond],
            "refused": refused.tolist(),
            "degenerate": degenerate.tolist(),
            "score": score.tolist(),
            "feature_score": fscore.tolist() if fscore is not None else None,
        }
        if cond != BASELINE and feats is not None:
            pooled_feats.append(feats)
            pooled_success.append(~refused & ~degenerate)
        print(cond, "refusal rate", refused.mean())

    base_score = float(np.mean(per_cond[BASELINE]["score"]))
    summary = {
        cond: summarise(
            np.array(d["refused"]),
            np.array(d["degenerate"]),
            np.array(d["score"]),
            np.array(d["feature_score"]) if d["feature_score"] is not None else None,
            base_score,
        )
        for cond, d in per_cond.items()
    }
    features = (
        pooled_success_vs_refused_features(
            np.concatenate(pooled_feats), np.concatenate(pooled_success)
        )
        if pooled_feats
        else []
    )
    out = {
        "meta": {
            "layer": layer,
            "seed": SEEDS[0],
            "git_commit": git_commit(),
            "torch": torch.__version__,
            "limit": lim,
            "score": "0 = typical harmless prompt, 1 = typical harmful prompt (train means)",
            "sae_features_used": ids,
        },
        "summary": summary,
        "ranking_by_success": rank_conditions(summary),
        "refused_vs_success_features": features,
        "per_prompt": per_cond,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "jailbreaks.json").write_text(json.dumps(out) + "\n")
    print(json.dumps({"ranking": out["ranking_by_success"], "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
