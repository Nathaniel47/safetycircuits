# Refusal direction: first full run (Gemma-2-2B-it)

> **Correction (2026-11-27).** Refusal rates below are a phrase-match judge's counts. A human check found the judge missed most refusals when the direction was removed, and that unflagged fiction-frame replies had not complied. Read [limitations.md](limitations.md), section 1, before using any rate or conclusion here.

Source: `results/refusal_direction.json` and `results/refusal_direction.npy`, written by `python -m safetycircuits.refusal.run` on Kaggle (T4, fp32, torch 2.11, Python 3.13, run from commit `c405369`). The numbers match the output first pasted from the notebook. Seed 0, one run, greedy decoding, refusal judged by the string-match classifier in `safetycircuits/refusal/classify.py`. Intervals are 95% Wilson.

Chosen layer: **24** (of 26), picked on the validation split. Ablating the top three layers on val gave harmful-prompt refusal of 4.9% (layer 24), 12.5% (layer 23) and 26.4% (layer 22). The addition strength is 465.7, the length of the mean difference at layer 24. The direction file holds the unit vector (2304 values).

| Prompts (test split) | n | Baseline | Intervention | Random-direction control |
|---|---|---|---|---|
| Harmful: refusal rate, direction **removed** | 144 | 97.9% (141) [94.1, 99.3] | **7.6%** (11) [4.3, 13.2] | 97.9% (141) [94.1, 99.3] |
| Harmless: refusal rate, direction **added** | 160 | 2.5% (4) [1.0, 6.3] | **42.5%** (68) [35.1, 50.2] | 1.9% (3) [0.6, 5.4] |
| XSTest safe: refusal rate, direction **added** | 250 | 7.6% (19) [4.9, 11.6] | **86.4%** (216) [81.6, 90.1] | not run |
| XSTest unsafe: refusal rate, direction **removed** | 200 | 77.5% (155) [71.2, 82.7] | **0.5%** (1) [0.1, 2.8] | not run |

## What this does and does not show

- Removing one direction at layer 24 takes harmful-prompt refusal from 98% to 8%, while removing a random direction changes nothing. Adding it makes Gemma refuse harmless prompts, including most XSTest safe prompts. That is consistent with a single direction mediating refusal in this model, as reported for other models.
- **Not yet checked:** whether the non-refusals after ablation are real compliance or broken or off-topic text. The classifier counts anything without a refusal phrase as "not a refusal". It has not been checked by hand on real replies (Step 6).
- The addition strength is the full gap between the harmful and harmless means, so the 86% on XSTest safe prompts may be a strong push rather than a natural effect. Other strengths were not tried.
- One seed and one run. The random-direction control used one random direction.
- XSTest unsafe baseline is 77.5%, so Gemma does not refuse every unsafe-contrast prompt at baseline.
- Harmfulness labels come from the benchmark authors and were not audited.
- **Layer search was narrow.** AUROC is about 0.9996 at many layers, so layers were ordered by Cohen's d, and only the top three (24, 23, 22, all late layers) were tried for ablation. Middle layers such as 17 and 19 separate the classes almost as well but were never tested, so layer 24 may not be the best layer.
