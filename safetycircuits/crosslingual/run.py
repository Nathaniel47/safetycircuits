"""Does the English refusal direction work for Twi, French and Swahili prompts?

Run on a GPU after translate:  python -m safetycircuits.crosslingual.run
Saves numbers only (no replies, no translated text): results/crosslingual.json
"""

import argparse
import json

import numpy as np
import torch

from safetycircuits.config import DATA_DIR, LANGUAGES, RESULTS_DIR, SEEDS
from safetycircuits.crosslingual.metrics import CHRF_GOOD, chrf, summarise_language
from safetycircuits.model.hooks import ablating
from safetycircuits.model.load import generate_replies, load_model
from safetycircuits.refusal.direction import calibrated_score
from safetycircuits.refusal.run import BATCH, git_commit, load_prompts, refusal_flags, residuals

FLORES_DIR = DATA_DIR / "raw" / "flores"
FLORES_CODES = {"english": "eng_Latn", **LANGUAGES}
TRANSLATE_PROMPT = (
    "Translate the following text into English. Reply with the translation only.\n\n{}"
)


def load_translations() -> dict[tuple[str, str], list[dict]]:
    """{(language, kind): rows}, rows keep their chrF so poor translations can be filtered."""
    out: dict[tuple[str, str], list[dict]] = {}
    with open(DATA_DIR / "interim" / "translations.jsonl", encoding="utf-8") as f:
        for row in map(json.loads, f):
            out.setdefault((row["lang"], row["kind"]), []).append(row)
    return out


def comprehension(model, tok, texts: list[str], english: list[str]) -> float:
    """Can Gemma translate these (harmless) prompts back to English? Mean chrF, 0 to 100."""
    scores = []
    for i in range(0, len(texts), BATCH):
        prompts = [TRANSLATE_PROMPT.format(t) for t in texts[i : i + BATCH]]
        replies = generate_replies(model, tok, prompts, max_new_tokens=64)
        scores += [chrf(r, e) for r, e in zip(replies, english[i : i + BATCH], strict=True)]
    return float(np.mean(scores))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    lim = ap.parse_args().limit
    torch.manual_seed(SEEDS[0])

    ref = json.loads((RESULTS_DIR / "refusal_direction.json").read_text())
    layer = ref["chosen_layer"]
    unit = np.load(RESULTS_DIR / "refusal_direction.npy")
    data = load_prompts()
    splits = json.loads((DATA_DIR / "splits.json").read_text())
    translations = load_translations()
    model, tok = load_model()

    def texts_of(pool, ids):
        return [pool[i] for i in ids][:lim]

    train_h = texts_of(data["harmful"], splits["harmful"]["train"][:128])
    train_l = texts_of(data["harmless"], splits["harmless"]["train"][:128])
    h_mean = float((residuals(model, tok, train_h)[:, layer] @ unit).mean())
    l_mean = float((residuals(model, tok, train_l)[:, layer] @ unit).mean())

    harm_ids = splits["translate_ids"]
    less_ids = sorted(splits["harmless"]["test"])[: len(translations[("french", "harmless")])]
    english = {
        "harmful": texts_of(data["harmful"], harm_ids),
        "harmless": texts_of(data["harmless"], less_ids),
    }
    idx = splits["flores_control_indices"][:lim]
    unit_t = torch.tensor(unit)

    def score(prompts):
        return calibrated_score(residuals(model, tok, prompts)[:, layer], unit, l_mean, h_mean)

    results = {}
    for lang in ("english", *LANGUAGES):
        if lang == "english":
            prompts = english
            keep = {k: np.ones(len(v), bool) for k, v in english.items()}
        else:
            rows = {k: translations[(lang, k)][:lim] for k in ("harmful", "harmless")}
            prompts = {k: [r["text"] for r in v] for k, v in rows.items()}
            keep = {k: np.array([r["chrf"] >= CHRF_GOOD for r in v]) for k, v in rows.items()}
        lines = (FLORES_DIR / f"{FLORES_CODES[lang]}.devtest").read_text(encoding="utf-8")
        flores = [lines.split("\n")[i] for i in idx]

        h_score, l_score, f_score = (
            score(prompts["harmful"]),
            score(prompts["harmless"]),
            score(flores),
        )
        h_ref = np.array(refusal_flags(model, tok, prompts["harmful"]))
        l_ref = np.array(refusal_flags(model, tok, prompts["harmless"]))
        with ablating(model, unit_t):
            h_abl = np.array(refusal_flags(model, tok, prompts["harmful"]))

        entry = {
            "all": summarise_language(h_score, l_score, f_score, h_ref, h_abl, l_ref),
            "n_good_translations": {k: int(v.sum()) for k, v in keep.items()},
        }
        if lang != "english" and keep["harmful"].sum() >= 10 and keep["harmless"].sum() >= 10:
            kh, kl = keep["harmful"], keep["harmless"]
            entry["good_translations_only"] = summarise_language(
                h_score[kh], l_score[kl], None, h_ref[kh], h_abl[kh], l_ref[kl]
            )
        if lang != "english":
            entry["gemma_comprehension_chrf_harmless"] = comprehension(
                model, tok, prompts["harmless"], english["harmless"]
            )
        results[lang] = entry
        print(lang, entry["all"]["direction_auroc"], entry["all"]["harmful_refusal_rate"])

    out = {
        "meta": {
            "layer": layer,
            "seed": SEEDS[0],
            "git_commit": git_commit(),
            "torch": torch.__version__,
            "limit": lim,
            "chrf_good_threshold": CHRF_GOOD,
            "note": "Refusal check also recognises French/Swahili phrases; Twi phrases unverified.",
        },
        "languages": results,
    }
    (RESULTS_DIR / "crosslingual.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
