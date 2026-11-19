"""Machine-translate the study prompts with NLLB-200 (CC-BY-NC-4.0, so research use only).

Run on a GPU:  python -m safetycircuits.crosslingual.translate
Writes data/interim/translations.jsonl (gitignored, holds translated prompt text) and
results/translation_quality.json (numbers only: back-translation chrF per language).

Quality is checked by translating back to English and comparing with the original. That catches
gross failures only; a Twi reader should still check a sample (see PLAN.md).
"""

import json

import numpy as np
import torch

from safetycircuits.config import DATA_DIR, LANGUAGES, RESULTS_DIR, TRANSLATION_MODEL
from safetycircuits.crosslingual.metrics import CHRF_GOOD, chrf

N_HARMLESS = 100
BATCH = 16


def load_nllb():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL, dtype=dtype)
    return model.to("cuda" if torch.cuda.is_available() else "cpu").eval(), tok


@torch.no_grad()
def translate(model, tok, texts: list[str], src: str, tgt: str) -> list[str]:
    tok.src_lang = src
    out = []
    for i in range(0, len(texts), BATCH):
        batch = tok(texts[i : i + BATCH], return_tensors="pt", padding=True, truncation=True)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        ids = model.generate(
            **batch, forced_bos_token_id=tok.convert_tokens_to_ids(tgt), max_new_tokens=128
        )
        out += tok.batch_decode(ids, skip_special_tokens=True)
    return out


def main() -> None:
    splits = json.loads((DATA_DIR / "splits.json").read_text())
    pools = {}
    for name in ("harmful", "harmless"):
        with open(DATA_DIR / "interim" / f"{name}.jsonl", encoding="utf-8") as f:
            pools[name] = {r["id"]: r["text"] for r in map(json.loads, f)}
    items = {
        "harmful": splits["translate_ids"],
        "harmless": sorted(splits["harmless"]["test"])[:N_HARMLESS],
    }
    model, tok = load_nllb()
    rows, quality = [], {}
    for lang, code in LANGUAGES.items():
        for kind, ids in items.items():
            english = [pools[kind][i] for i in ids]
            translated = translate(model, tok, english, "eng_Latn", code)
            back = translate(model, tok, translated, code, "eng_Latn")
            scores = np.array([chrf(b, e) for b, e in zip(back, english, strict=True)])
            quality[f"{lang}_{kind}"] = {
                "mean_backtranslation_chrf": float(scores.mean()),
                "share_good": float((scores >= CHRF_GOOD).mean()),
            }
            for i, t, s in zip(ids, translated, scores, strict=True):
                rows.append({"id": i, "kind": kind, "lang": lang, "text": t, "chrf": float(s)})
    with open(
        DATA_DIR / "interim" / "translations.jsonl", "w", encoding="utf-8", newline="\n"
    ) as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out = {"model": TRANSLATION_MODEL, "chrf_good_threshold": CHRF_GOOD, "quality": quality}
    (RESULTS_DIR / "translation_quality.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
