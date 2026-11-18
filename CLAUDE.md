# CLAUDE.md: SafetyCircuits coding rules

- Follow PLAN.md; if anything conflicts, PLAN.md wins. Work one BUILD_STEPS.md step at a time.
- Python 3.11. Plain PyTorch + Hugging Face hooks (TransformerLens only if it supports Gemma-2 cleanly). No nnsight.
- Secrets only from the `HF_TOKEN` environment variable. Never commit `.env`; `.env.example` has names with empty values.
- **Responsible handling:** public benchmark prompts only. Measure refusal behaviour only. Never save, print, commit or publish harmful completions: store refusal flags, scores and activations, not model text for harmful prompts.
- Refusal scoring is automatic (string match, hand-checked on a sample). No LLM judge.
- GPU work runs in Kaggle/Colab notebooks and saves outputs to files; analysis and the app run on the laptop from cached results. CI installs no torch.
- Fixed seeds. Record model id, SAE id, seed, library versions and git commit in `results/run_meta.json`.
- Format and lint with ruff (line length 100). Write tests in `tests/` for every feature; run `pytest` before each commit.
- Code comments only for non-obvious reasons. Plain-English explanations go in LEARNING.md, not chat.
- Commit with the step name in the message, then push (if a remote exists).
