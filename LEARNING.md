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
