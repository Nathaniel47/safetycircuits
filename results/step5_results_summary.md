# Step 5: results (Kaggle run, 2026-11-20)

Source: `results/sae_features.json`, `jailbreaks.json`, `translation_quality.json`, `crosslingual.json` and `browser_cache.json`, written by the Step 5 modules on Kaggle (T4 x2, seed 0, run from commit `db3aa94`). The SAE and jailbreak numbers below were first read off the screen and match the JSON files exactly. One run, one seed, string-match refusal check, nothing hand-checked.

## SAE fit (Gemma Scope 16k SAEs trained on the base model)

Fraction of variance unexplained (FVU, 0 is perfect) and mean features firing per token (L0), on 2,767 tokens of chat-formatted train prompts.

| Model | Layer | FVU with BOS token | FVU without BOS | L0 without BOS |
|---|---|---|---|---|
| Instruct (Gemma-2-2B-it) | 17 | 8.98 | **0.257** | 105 |
| Instruct | 24 | 4.60 | **0.339** | 86 |
| Base (Gemma-2-2B) | 17 | 8.97 | **0.272** | 66 |
| Base | 24 | 4.14 | **0.314** | 57 |

- The BOS token (the first token) has a very large norm and wrecks the fit (FVU above 1) on both models. Without it the fit is moderate, and **the instruct model's FVU is within about 0.03 of the base model's**, so the SAEs transfer in this sense. The instruct model fires about 1.3 to 1.6 times more features per token than the base model, so features are somewhat less sparse there.
- An FVU of 0.26 to 0.34 is not a tight fit, and the prompts are chat-formatted, which the SAEs were not trained on.
- Top refusal features: at layer 24, validation AUROC about 0.99 or higher for the best ones (features 9228, 10878, 786, 15400, 4253...). Each feature's own direction has only a small cosine (0.05 to 0.29) with the refusal direction, and the top 20 together rebuild the direction with cosine 0.47 (layer 24) and 0.59 (layer 17). So the direction is spread over many features. The recomputed direction matches the saved one (cosine 1.0).

## Jailbreaks on the 144 harmful test prompts (layer 24)

Success = not a refusal and not obviously broken text (no broken text was found in any condition). Score: 0 = typical harmless prompt, 1 = typical harmful prompt.

| Condition | Refused | Success | Refusal score | Suppression | Score predicts refusal (AUROC) |
|---|---|---|---|---|---|
| baseline | 98% | 2% | 0.98 | 0.00 | 0.94 |
| fiction_frame | 57% | **43%** | 0.28 | 0.71 | 1.00 |
| refusal_suppression | 74% | 26% | 0.38 | 0.61 | 0.78 |
| unrestricted_persona | 85% | 15% | 0.45 | 0.54 | 0.97 |
| research_framing | 85% | 15% | **0.96** | **0.02** | 0.87 |
| prefix_injection | 94% | 6% | 0.75 | 0.23 | 1.00 |
| random_words_suffix (control) | 94% | 6% | 0.91 | 0.07 | 0.91 |
| polite_suffix (control) | 94% | 6% | 0.89 | 0.09 | 0.86 |
| affirmative_suffix | 98% | 2% | 0.88 | 0.10 | 1.00 |

- The strongest jailbreaks (fiction frame, refusal suppression, persona) also pull the refusal score down the most. This fits the single-direction story.
- **`research_framing` is the exception:** it lifts the non-refusal rate to 15% while the score stays at 0.96. Here the model stops refusing without the direction moving, so the direction does not explain every jailbreak.
- The two controls (random words, polite) already move success from 2% to 6%, so differences of a few points between conditions are within noise. No confidence intervals are shown here; the JSON has them.
- One seed, one run, string-match refusal check, not hand-checked.

## Translation quality (NLLB-200, back-translation chrF; 35 or more counts as good)

| Language | Harmful prompts: mean chrF, share good | Harmless prompts: mean chrF, share good |
|---|---|---|
| Twi | 35.4, 38% | 36.5, 48% |
| French | 79.3, 100% | 78.2, 100% |
| Swahili | 64.3, 97% | 60.7, 92% |

**Twi translations are poor:** more than half of them fail the chrF threshold, so any Twi result is about badly translated prompts as much as about Gemma. A Twi reader should check a sample, and results should be shown for the good-translation subset as well.

## Cross-lingual: does the English refusal direction work in other languages?

100 harmful and 100 harmless English prompts, machine-translated with NLLB-200 (translations are not committed). AUROC = how well the English direction separates harmful from harmless prompts (1 = perfect, 0.5 = chance).

| Language | Direction AUROC [95% CI] | Harmful refused | Refused, direction removed | Harmless refused | Gemma translates it back (chrF) |
|---|---|---|---|---|---|
| English | 0.999 [1.0, 1.0] | 97% | 6% | 3% | not measured |
| French | 0.995 [0.99, 1.0] | 87% | 0% | 4% | 70.7 |
| Swahili | 0.957 [0.93, 0.98] | 70% | 1% | 2% | 43.3 |
| Twi | **0.563 [0.48, 0.64]** | 6% | 0% | 18% | 18.1 |

- **French and Swahili:** the English direction still separates the classes almost perfectly (Swahili's mean harmful score is 0.29 against 0.97 in English, so it is weaker in size, not in ordering), and removing it removes the refusals.
- **Twi is at chance (AUROC 0.56), but this does not show a safety failure.** Gemma recovered the English meaning of Twi prompts with a chrF of only 18, so it barely understands them, and more than half of the Twi translations fail the quality threshold. On the good-translation subset (38 harmful, 48 harmless) AUROC is 0.58 [0.45, 0.70], too wide to say anything. The Twi refusal phrases in the classifier are unverified guesses, so the 6% refusal rate may be undercounted.
- Comprehension was measured on harmless prompts only. FLORES benign text scores near 0.1 in every language.
- **Not done:** a Twi reader has not checked any translation.
