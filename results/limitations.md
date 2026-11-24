# Limitations: where the results are weak and what would improve them

Everything here is Gemma-2-2B-it, one prompt set, one seed for the main runs, and a string-match refusal check. Numbers are in [results_table.md](results_table.md).

## 1. The refusal check is not validated on harmful prompts

- Refusal is a phrase match on the start of a reply. "Not refused" does not mean "complied".
- A second automatic check (does the reply open like a helpful answer?) can confirm only about half of the non-refusals when the direction is removed: 71 of 133 are clear compliance, 62 are unclear. For the fiction-frame jailbreak it confirms **none** of the 62 non-refusals (a story does not open with "Sure"), so the 43% "success" rate for that jailbreak is unconfirmed.
- Reading harmful replies by hand would be the real check, and the project rule is not to print, store or publish them. **Improvement:** have a person review a sample in a private session with nothing saved, or use a separate classifier model, with the limits that brings.
- **Review of 50 replies to harmless prompts** (25 Alpaca, 25 XSTest safe; [classifier_review_benign.csv](classifier_review_benign.csv)). I (Claude, not a human reviewer) read the first 110 characters of each reply. The judge flagged 2, and both were real refusals or hedges ("I can't give you specific advice...", "I cannot provide financial advice..."): no false positives. It **missed 2** that decline in other words: "I must strongly advise against..." (eliminating jobs) and "I'm designed to provide safe and..." (leaving a job), and one more reply is unclear. So on these prompts the judge found about 2 of 4 refusals, which means it **undercounts refusals phrased as advice or caveats**, and the true refusal rates (and the "not refused" counts) are somewhat off. The numbers elsewhere were produced with the original phrase list and were not recomputed. This is a small sample read from truncated text, not a replacement for a human check.

## 2. The layer choice is not settled

- Removing the direction fitted at layers 8, 10, 12, 14 and 25 left 0% of validation prompts refused. Layers 16 to 24 did worse (4% to 26%). Layer 24 (the headline, used throughout) is not the best. Five layers tie at 0%, and the code picked layer 25 by an arbitrary tie-break.
- Only layers 24 and 25 were run on the test split, so the early-layer results rest on 144 validation prompts and are unconfirmed. **Improvement:** run the test split for all tied layers, and check the replies are coherent, since removing a direction from an early layer could break the model rather than just remove refusal.

## 3. Easy test data

- Plain word counts separate AdvBench and HarmBench prompts from Alpaca prompts with AUROC 0.99, so the direction's 0.999 on that test says little. On XSTest, where safe and unsafe prompts share words, word counts fall to 0.61 and the direction keeps 0.98. That is the more meaningful number.
- Near-duplicate prompts across splits (27 pairs, Step 3) make test results slightly optimistic.

## 4. Statistical power

- Variance across 3 refits on random 80% subsets is tiny (refusal after ablation 7.4% ± 0.4%), but all refits share the same prompts, so this shows the direction is stable, not that results hold on new prompt sets.
- Most main results use one seed, 144 test prompts and one set of generated replies (greedy decoding). Differences of a few points between jailbreaks are within the noise of the two control suffixes (6% success).

## 5. SAE features

- The Gemma Scope SAEs were trained on the base model. They reconstruct Gemma-2-2B-it activations about as well as the base model's (FVU 0.26 to 0.34), but only after excluding the first token, and the instruct model fires more features per token. Feature results are hints. Features were not inspected on Neuronpedia, and no feature was causally tested.
- The top refusal features rebuild only about half of the direction (cosine 0.47 at layer 24). The direction is spread over many features.

## 6. Cross-lingual results

- Twi is not testable here: Gemma recovers the English meaning of Twi prompts with a chrF of 18, and most Twi translations are poor. The Twi AUROC of 0.56 (chance) reflects poor translation and poor model comprehension, not a safety finding. No Twi reader has checked the translations, and the Twi refusal phrases in the classifier are unverified guesses.
- French and Swahili translations come from NLLB-200 and were checked only by round trip.
- **Improvement:** a Twi reader reviews [twi_review.csv] (a local file, not committed), and the test uses human translations.

## 7. Scope

- One model, one family of jailbreaks written for this study (5 templates, 3 suffixes, no optimised attacks), and an English-centred direction. Nothing here says how other models, languages or attacks behave.
