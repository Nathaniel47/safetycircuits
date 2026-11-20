# LEARNING.md: SafetyCircuits

Plain-English notes on what was built and why it matters. One section per step.

## Step 2: Repo setup

**What was built.** An empty but working project: a Python package `safetycircuits/` with one folder per part of the study, a `config.py` that holds the model and SAE choices, dependency files, a Dockerfile, formatting and secret checks, and a GitHub Actions workflow that runs the tests.

**Why the dependencies are split in three.**
- `requirements.txt` is what the feature browser needs (Streamlit, Plotly, numpy, pandas). It has no PyTorch, so the app deploys on free hosting and CI stays fast.
- `requirements-gpu.txt` adds PyTorch, Transformers and SAELens. You only install it in Kaggle or Colab, where the GPU is.
- `requirements-dev.txt` adds pytest, ruff and pre-commit.

This mirrors the plan: the heavy model work happens on a free GPU and saves files, and the laptop and the web app only read those files.

**Secrets.** The only secret is the Hugging Face token (`HF_TOKEN`), needed because Gemma is gated. It lives in `.env` (ignored by git) or in Kaggle/Colab Secrets. The `.env.example` file lists the name with an empty value, and a test checks that stays true. The gitleaks pre-commit hook blocks commits that look like they contain a key.

**Responsible handling.** CLAUDE.md records the rule that the study stores refusal flags, scores and activations, never harmful model text.

**How to check it works.** Activate a virtual environment, run `pip install -r requirements-dev.txt`, then `pytest` and `ruff check .`. All should pass.

**Reading.** The pre-commit documentation (pre-commit.com) intro page: it explains hooks in five minutes.

## Step 3: Data pipeline

**What was built.** Three small scripts. `fetch` downloads the public datasets and records their size and checksum. `prepare` cleans them and splits them with a fixed seed. `report` writes `results/data_report.md`. The numbers: 720 harmful prompts (520 AdvBench + 200 HarmBench), 800 harmless prompts sampled from Alpaca-cleaned, 450 XSTest prompts, and 100 harmful test prompts picked for translation into Twi, French and Swahili.

**Why the splits matter.** The refusal direction is built from the *train* split, the best layer is chosen on *val*, and the final numbers come only from *test*. If the same prompts were used for all three, the results would look better than they are. The splits are stratified, so AdvBench and HarmBench appear in each split in the same proportions. The seed is fixed, so everyone gets the same split.

**Why the repo holds only ids.** `data/splits.json` lists prompt ids, not text. The harmful prompt text stays in gitignored folders and can be rebuilt with the scripts. That keeps harmful text out of the repo, and the report prints only counts and lengths for harmful prompts.

**Problems the report found (and why they matter).**
- AdvBench is short and repetitive: most prompts start with "write", "create" or "give". Alpaca prompts are more varied. A refusal direction could partly learn "sounds like AdvBench" instead of "is harmful". XSTest (safe prompts that look unsafe) is one check, and step 6 should add a style-matched control.
- 27 pairs of harmful prompts are near duplicates, so some test prompts have a close twin in train. Test scores are slightly optimistic.
- FLORES-200 has Twi, but only benign sentences. Harmful prompts in Twi must be machine-translated in step 5, and the Twi quality is unchecked until then.

**How to check it.** Run the three commands in the README, then `pytest`. Tests check that splits are repeatable, do not overlap, and contain ids only, and that the report never contains harmful text.

**Reading.** "Datasheets for Datasets" (Gebru et al., arXiv 1803.09010), first two pages: the questions to ask about any dataset before trusting it.

## Step 4: Core feature (the refusal direction)

**Status: code, tests and a first full run are done.** The numbers copied from the Kaggle output are in `results/refusal_direction_summary.md` (harmful refusal 98% -> 8% with the direction removed; random direction: no change). The raw `results/refusal_direction.json` and `.npy` are now in the repo. The replies have not been checked by hand, and only late layers were tried for ablation, so treat the numbers as preliminary.

**The idea in plain English.** While Gemma reads a prompt, each layer keeps a long list of numbers (the *residual stream*) that holds what the model currently "thinks" about it. Take that list at the last token of the prompt, for many harmful prompts and many harmless prompts, and average each group. The *difference of the two averages* is an arrow in that number space. The study asks whether that single arrow is what makes the model refuse.

**Two tests of the arrow.**
- **Ablation** (remove it): at every layer, subtract the part of each vector that points along the arrow. If the arrow really drives refusal, harmful prompts should stop being refused.
- **Addition** (add it): add the arrow to a harmless prompt. If it drives refusal, the model should start refusing harmless questions. Adding it to XSTest safe prompts measures over-refusal.

**Controls.** Each test also runs with a *random* arrow of the same length. If removing a random arrow changes refusal as much as removing the real one, the result means nothing. Rates come with 95% confidence intervals (Wilson).

**How the layer is chosen (no peeking at test data).** The arrow is built from the train prompts. Each layer is ranked on the val prompts by how well the arrow separates harmful from harmless (AUROC, with Cohen's d to break ties). The top three layers are then tried for ablation on val, and the one that removes the most refusal wins. The test split is touched only at the end.

**What the code does.**
- `refusal/direction.py`: the arrow, AUROC and the layer ranking (numpy only, tested in CI).
- `model/hooks.py`: reads the residual stream at the last token, and removes or adds the arrow. Removal hooks the embeddings and the attention and MLP outputs of every layer, which are the things that write into the stream.
- `refusal/classify.py`: decides "refusal or not" by matching phrases such as "I cannot" in the first 120 characters. No LLM judge. It is crude: it is only checked on a few unit-test examples so far, and Step 6 must check it by hand on 50 real replies.
- `refusal/run.py`: runs everything and saves only numbers, never model replies.

**How I checked it without the model.** The tests build a tiny random Gemma-2 and confirm that: captured values match what the model reports internally; after ablation the arrow's coordinate is about zero at every layer; adding shifts the coordinate by exactly the chosen amount; a random arrow does not remove the real one; and hooks are removed afterwards. They prove the plumbing, not the science.

**Things to watch in the real run.**
- The T4 and P100 have no native bf16, so the script uses fp32 (10.5 GB), which fits.
- If ablation barely lowers refusal, check the chosen layer and the classifier before concluding anything.
- Step 3 found that harmful and harmless prompts differ in style, so the arrow may partly encode style. The XSTest numbers are the first check.

**Reading.** Arditi et al. 2024, "Refusal in Language Models Is Mediated by a Single Direction" (arXiv 2406.11717): sections 2 and 3.

## Step 5.1: Sparse-autoencoder feature analysis

**Status: done and run on Kaggle; results in `results/sae_features.json` and `results/step5_results_summary.md`.** Result: with the BOS token removed, the base-model SAEs reconstruct the instruct model's activations about as well as the base model's (FVU 0.26 and 0.34 against 0.27 and 0.31), so they transfer. The BOS token must be excluded (it pushed FVU above 1 on both models).

**The idea in plain English.** The refusal direction (step 4) is one arrow in a 2,304-number space. It says *that* the model is about to refuse, but not *what is going on inside*. A sparse autoencoder (SAE) takes the same 2,304 numbers and re-writes them as activations of 16,384 "features", most of which are zero at any moment. Researchers often find that individual features look like single concepts ("requests for weapons", "polite apologies"). Gemma Scope is Google's free set of these SAEs for Gemma 2.

**What we measure.**
1. For each prompt, the SAE feature activations at the last prompt token.
2. The features that fire much more on harmful prompts than on harmless ones (AUROC, ranked on train, rechecked on val, so a feature that was lucky on train shows up as weak on val).
3. How each top feature's own vector points relative to the refusal direction (cosine), and how well the top 20 features together rebuild the direction (`direction_overlap`).
4. A link to each feature's Neuronpedia page, where its top-activating text can be read. The link format is unchecked; open one to confirm.

**The catch: these SAEs were trained on the base model, not on Gemma-2-2B-it.** There is no 2B-it release. So the first result to read is `sae_fit`: how much of the activation the SAE fails to explain (FVU, 0 is perfect) and how many features fire per token (L0), on the instruct model and, with `--compare-base`, on the base model. If the instruct numbers are close to the base numbers, the SAEs transfer. If not, feature results should be treated as weak. This is the week-3 decision in PLAN.md.

**Why not SAELens.** The plan named it, but its loading API changed between versions and installing it upgraded torch in the Kaggle image (a likely cause of the crashes). Gemma Scope files are plain arrays, so `sae/load_sae.py` reads them directly (`W_enc`, `W_dec`, `b_enc`, `b_dec`, `threshold`) and does the "JumpReLU" step itself: a feature fires only if its value passes its own threshold. A test checks that. The file keys are from Google's published format and are checked at load, so the first Kaggle run will confirm them.

**Which layers.** 24 (where the refusal direction was found) and 17 (a middle layer, because step 4 only tried late layers). The SAE used is the one whose average L0 is closest to 100.

**Reading.** Lieberum et al. 2024, "Gemma Scope" (arXiv 2408.05147), section 2 (what JumpReLU SAEs are).

## Step 5.2: Jailbreak study

**Status: done and run on Kaggle; see `results/jailbreaks.json` and `results/step5_results_summary.md`.** Result: fiction frame got 43% of harmful prompts past the refusal (baseline 2%), refusal suppression 26%, and the strongest ones also lowered the refusal score most. Research framing got 15% through with the score unchanged, so the direction does not explain every jailbreak. The two control suffixes alone gave 6%, so small gaps are noise.

**The question.** A jailbreak is a way of wording a request so that the model complies when it would normally refuse. If refusal really runs through one direction (step 4), a successful jailbreak should push the model's activations *away* from that direction. This study tests that rather than assuming it.

**The conditions.** The baseline (no jailbreak), five wrappers placed around the request (refusal suppression, prefix injection, fiction frame, an unrestricted-persona prompt, a research framing) and three suffixes: an "answer starting with Sure" suffix, a polite suffix and a suffix of random words. The polite and random-word suffixes are controls: if they lower refusal as much as the "clever" ones, the effect is just noise from a changed prompt. No optimised GCG suffixes are used (too expensive, and specific to one model). The wrappers hold no harmful content of their own, and a test checks that.

**What is measured for each, on the 144 harmful test prompts.**
- **Success rate:** the reply is not a refusal *and* not obviously broken. The broken-text check (`looks_degenerate`) is crude (very short, or one word repeated), so it will miss subtler nonsense. Replies are never saved.
- **Refusal score:** the prompt's position along the refusal direction, scaled so 0 is a typical harmless prompt and 1 is a typical harmful one. **Suppression** is how far a jailbreak pulls the score towards 0.
- **Refusal-feature score:** the same idea using the top SAE refusal features from step 5.1, when that result exists.
- **Does the score predict the outcome?** Within one condition, an AUROC of how well the score separates prompts that were refused from prompts that were not. If jailbreaks work by lowering the score, it should be high. If the model refuses anyway at a low score, or complies at a high one, the single-direction story is incomplete.
- **Features that mark failure:** pooling all jailbreak rows, which SAE features are most active when a jailbreak fails and the model still refuses.

**Limits to keep in mind.** One layer (24), one seed, greedy decoding, a string-match refusal check, and the score is a projection at the last prompt token only. A "success" here means "did not refuse in the first 48 tokens", not "gave a harmful answer".

**Reading.** Wei et al. 2023, "Jailbroken: How Does LLM Safety Training Fail?" (arXiv 2307.02483), the sections on competing objectives and mismatched generalization.

## Step 5.3: Cross-lingual test (Twi, French, Swahili)

**Status: done and run on Kaggle; see `results/crosslingual.json` and `results/step5_results_summary.md`.** Result: the English direction separates harmful from harmless almost perfectly in French (AUROC 0.995) and Swahili (0.957) but sits at chance in Twi (0.56). Gemma barely understands Twi (it translates it back with chrF 18) and the Twi translations are poor, so that is not evidence of a safety failure.

**The question.** The refusal direction was found from English prompts. Does the same arrow, unchanged, also separate harmful from harmless prompts written in Twi, French and Swahili? Safety training is known to transfer worse to languages with less data, so Twi is the interesting case.

**How the prompts are made.** FLORES-200 does have Twi, but it holds only harmless Wikipedia-style sentences. So 100 harmful and 100 harmless English prompts are machine-translated with NLLB-200 (Meta's translation model; its licence is non-commercial, fine for research). Translated text stays in gitignored folders and is never committed.

**Checking the translations.** Each translation is translated back to English and compared with the original using chrF (a character-overlap score from 0 to 100). That catches gross failures, such as garbage Twi, but not subtle ones. Results are reported for all prompts and for the "good" ones only (chrF at or above 35, a judgement call). A Twi reader should still check a sample by eye.

**What is measured, per language.**
- **Direction AUROC**: how well the English arrow separates harmful from harmless prompts in that language (1.0 perfect, 0.5 no better than chance), with a bootstrap interval.
- **Mean score** for harmful, harmless, and FLORES sentences (benign text, which should sit near 0).
- **Refusal rate** on harmful prompts, with and without the direction removed.
- **Comprehension control**: Gemma is asked to translate the harmless prompts back into English. If it can't read Twi, a low AUROC in Twi says nothing about safety. It only says the model did not understand. Harmless prompts only are used here, so no harmful text is generated.

**Reading the results honestly.** The refusal check looks for English refusal phrases plus French and Swahili ones. The Twi phrases are unverified guesses, so Twi refusal rates may be undercounted. Gemma may also answer in English. The direction AUROC does not depend on the classifier, so it is the more reliable number.

**Reading.** Yong et al. 2023, "Low-Resource Languages Jailbreak GPT-4" (arXiv 2310.02446), and the FLORES-200 / NLLB paper "No Language Left Behind" (arXiv 2207.04672), the evaluation section.

## Step 5.4: Interactive feature browser

**Status: done. The cache (237 prompts) was built on Kaggle and is in `results/browser_cache.json`, so the app works out of the box.**

**What it does.** Pick or type a prompt and see: a gauge with its refusal score (0 = like a typical harmless prompt, 1 = like a typical harmful one), whether Gemma refused it, a bar chart of the ten SAE features that fire most at the end of the prompt (refusal features in red, each linked to Neuronpedia), and a box plot showing where the prompt sits among all cached prompts. A second tab shows the study results (refusal direction table, jailbreak success rates, cross-lingual AUROC) from whichever result files exist.

**Why a cache and not a live model.** Free hosting has no GPU, and the model needs about 10 GB. So a GPU run scores about 250 prompts once and saves the numbers (`results/browser_cache.json`); the app only reads that file. The honest limit: **you can only look up prompts that are in the cache.** A typed prompt that isn't cached gets a clear message, not a made-up score. A live mode would need a GPU machine and is not built.

**What is in the cache.** Hand-written examples, harmless Alpaca prompts, XSTest safe and unsafe prompts, the XSTest unsafe prompts wrapped in each of the five jailbreak templates, harmless prompts translated into French, Swahili and Twi, and FLORES benign sentences. The "harmful-looking" examples are the short public XSTest ones. The long AdvBench and HarmBench text is kept out, and only refused/answered flags are stored, never replies.

**How it is tested.** The logic lives in `safetycircuits/browser_core.py` and is unit tested. The Streamlit app is exercised with Streamlit's own test runner on a small fake cache (lookup, example buttons, unknown prompt, no cache). I could not open the app in a browser during this session, so how the charts *look* is unchecked.

**Reading.** Streamlit docs, "App testing" (docs.streamlit.io), the page on `AppTest`.
