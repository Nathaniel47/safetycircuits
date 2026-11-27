# Limitations: where the results are weak and what would improve them

Everything here is Gemma-2-2B-it, one prompt set, one seed for the main runs, and a string-match refusal check that did badly in a human check once the refusal direction was removed (section 1). Numbers are in [results_table.md](results_table.md).

## 1. The refusal check fails when the direction is removed

**Human check on harmful prompts (2026-11-27, [human_review.json](human_review.json)).** One person labelled 60 replies: the same 20 prompts at baseline, with the layer-24 direction removed, and under the fiction-frame jailbreak. The replies were shuffled, with the condition and the judge's verdict not shown. No reply was skipped, and only counts are kept here.

| Condition (20 prompts) | Judge: refused | Reviewer: refused | Reviewer: complied | Reviewer: off-topic | Same refused / not refused answer |
|---|---|---|---|---|---|
| Baseline | 20 | 20 | 0 | 0 | 20/20 |
| Direction removed | 3 | **15** | **5** | 0 | **6/20** |
| Fiction frame | 11 | 11 | **0** | 9 | 20/20 |

- **Direction removed.** The judge missed 13 of the 15 refusals the reviewer found, and flagged 1 reply that the reviewer read as compliance. Refusals went from 20/20 to 15/20 (75%, 95% interval 53% to 89%), and compliance from 0/20 to 5/20 (25%, 11% to 47%). So the headline fall from 97.9% to 7.6% does **not** measure the fall in actual refusals. Removing the direction appears to have a much smaller effect on behaviour than the judge reported, and this sample is too small to size it: with the same prompts in both conditions, 5 changes out of 20, all one way, give an exact two-sided sign-test p of about 0.06.
- **Fiction frame.** The judge's refused / not refused split matched the reviewer's, but **none of the 9 replies it did not flag complied** (all off-topic). "Not refused" is not a successful jailbreak. The 43% figure must not be read as a jailbreak success rate. With 0 of 20 prompts complying, the 95% upper bound for compliance in this filtered pool is about 16%.
- **What else depends on this judge.** Every refusal rate in this study: the XSTest ablation result, the random-direction control, the refits, the layer comparisons, all jailbreak rates, and the cross-lingual "removing the direction removes refusals" result. Only the conditions above were checked. The AUROC results, SAE reconstruction numbers and refusal scores do not use the judge.
- **Limits of the check.** One reviewer, who is the project author, with no second reader. 20 prompts, filtered to leave out the chemical and biological category and prompts matching a word list, so they do not represent the whole test split. Layer 24 only. The reviewer saw the first 200 characters of replies capped at 48 tokens, so some endings were cut, and a longer reply might read differently. Hiding the condition cannot stop a reply's style from hinting at it. The first attempt stopped within the first three replies and the review restarted, so those were seen twice. Why the judge missed refusals was not examined.
- **Not done:** the phrase list was not changed and no number was recomputed. **Improvement:** a better judge (for example a separate safety classifier), tested against a larger human-labelled sample, then every rate rerun.

**Earlier automatic and benign checks**

- Refusal is a phrase match on the start of a reply. "Not refused" does not mean "complied".
- A second automatic check (does the reply open like a helpful answer?) can confirm only about half of the non-refusals when the direction is removed: 71 of 133 are clear compliance, 62 are unclear. For the fiction-frame jailbreak it confirms **none** of the 62 non-refusals (a story does not open with "Sure"), so the second check could not support the 43% "success" rate for that jailbreak. This second check was not compared with the human labels.
- **Review of 50 replies to harmless prompts** (25 Alpaca, 25 XSTest safe; [classifier_review_benign.csv](classifier_review_benign.csv)). I (Claude, not a human reviewer) read the first 110 characters of each reply. The judge flagged 2, and both were real refusals or hedges ("I can't give you specific advice...", "I cannot provide financial advice..."): no false positives. It **missed 2** that decline in other words: "I must strongly advise against..." (eliminating jobs) and "I'm designed to provide safe and..." (leaving a job), and one more reply is unclear. So on these prompts the judge found about 2 of 4 refusals, which means it **undercounts refusals phrased as advice or caveats**, and the true refusal rates (and the "not refused" counts) are somewhat off. The numbers elsewhere were produced with the original phrase list and were not recomputed. This is a small sample read from truncated text. It did not predict how badly the judge would do on harmful prompts with the direction removed.

## 2. The layer choice is not settled

- Removing the direction fitted at layers 8, 10, 12, 14 and 25 left 0% of validation prompts refused. Layers 16 to 24 did worse (4% to 26%). Layer 24 (the headline, used throughout) is not the best.
- **Confirmed on the test split (2026-11-24, `results/early_layers.json`):** removing the direction fitted at layer 12 or 14 leaves 0.7% of harmful prompts refused (1/144), against 7.6% at layer 24 and 2.1% at layer 25. Their non-refusals also look the most like real compliance (78% and 80% open like a helpful answer, against 53% at layer 24). Layers 8 and 10 leave 0% refused, but only 10% and 44% of their non-refusals open like a helpful answer, so their replies are probably off-topic or broken, and they should not be preferred.
- So by the judge, layer 12 or 14 looks like a better choice than 24. The headline numbers elsewhere (7.6% refused, jailbreak scores, cross-lingual) still use layer 24 and were not recomputed. The "helpful answer" check is crude (the fiction-frame problem in section 1 applies), so this is evidence, not proof, that replies are coherent. No broken-text flags were raised at any layer. **The layer comparison rests on the same phrase-match judge (section 1), and no person has read replies from any layer but 24, so which layer is better is an open question.**

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
