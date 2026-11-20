# DEPLOY.md: put the feature browser online for free

**Status: deployed on 2026-11-20 at https://safetycircuits.streamlit.app** (repository `Nathaniel47/safetycircuits`, branch `master`, main file `app/browser.py`, Python 3.11, no secrets, "Make this app public" on). The steps below are how it was done and how to redo it.

The demo is the Streamlit app in `app/browser.py`. It reads precomputed files from `results/`, needs **no GPU, no model, no API key and no sign-up for visitors**. The recommended host is **Streamlit Community Cloud** (free, deploys straight from this GitHub repo). A Hugging Face Spaces alternative is at the end.

## Before you start (already done)

- The repo `Nathaniel47/safetycircuits` is **public** (checked 2026-11-20), which the free Streamlit tier needs.
- `requirements.txt` holds only `streamlit`, `plotly`, `numpy` and `pandas`. Streamlit Cloud installs it automatically; the app never imports torch or transformers.
- `results/browser_cache.json` (about 120 kB) and the three chart images are committed. Nothing is downloaded at start-up.
- No secrets are used. A test (`tests/test_deploy_readiness.py`) fails if a Hugging Face or AWS-style key appears in the source, and the gitleaks pre-commit hook blocks keys on every commit.
- `.streamlit/config.toml` turns off usage tracking.

## Steps (you need to do these: they use your own accounts)

1. **Check the repo is up to date.** On https://github.com/Nathaniel47/safetycircuits the latest commit should be the one you expect (the app and `DEPLOY.md` must be on `master`).
2. **Sign in to Streamlit Community Cloud.** Go to https://share.streamlit.io and click **Continue with GitHub**. Approve access when GitHub asks. (Streamlit may ask to read your public repos; that is all it needs.)
3. **Create the app.** Click **Create app** (or **New app**), then **Deploy a public app from GitHub**, and fill in:
   - **Repository:** `Nathaniel47/safetycircuits`
   - **Branch:** `master`
   - **Main file path:** `app/browser.py`
   - **App URL:** pick a short name, for example `safetycircuits`. Visitors will see `https://<name>.streamlit.app`.
4. **Advanced settings.** Choose **Python 3.11** (or 3.12). Leave **Secrets** empty: the app needs none.
5. Click **Deploy**. The first build installs four packages and takes a few minutes.
6. **Check it works** (see the checklist below), then copy the URL into the README (the "Demo link" line) and into `PORTFOLIO.md` when Step 8 creates it.

## Checklist after it is live

- [ ] The page opens without asking for a login (open it in a private window to be sure).
- [ ] Click an example button: a gauge, a feature chart and a box plot appear.
- [ ] The **Study results** tab shows a table and three charts.
- [ ] The **About** tab loads.
- [ ] Type a sentence that is not in the cache: you get a clear message and "Did you mean" buttons, not an error.

## Speed

- A running app answers an example click in well under a second.
- A **cold start can be slow.** Free Streamlit apps go to sleep after a few days without visitors, and the first visitor waits for the app to wake (often 30 seconds or more). Open the link yourself before sharing it. Visitors after that load it quickly. The load time was not measured on the real host, only locally.

## Updating the demo

Push to `master`; Streamlit Cloud redeploys automatically. To rebuild the data behind the demo, rerun notebooks 02 to 05 on Kaggle (see README.md), commit the new files in `results/`, and push.

## Secrets (only if you add a live model later)

The demo does not need them. If you ever add a feature that calls Hugging Face, put the token in **Settings, Secrets** in the Streamlit Cloud dashboard as `HF_TOKEN = "..."`, never in the repo. Do not add a live Gemma model to the free tier: it needs about 10 GB of memory and a GPU.

## Alternative: Hugging Face Spaces (Docker)

Use this if you prefer Hugging Face or Streamlit Cloud is not available.

1. Create a free account at https://huggingface.co and a new **Space** (SDK: **Docker**, hardware: **CPU basic, free**).
2. In the Space's `README.md` header set `app_port: 8501`.
3. Push this repository's files to the Space's git remote (the repo already has a `Dockerfile` that runs `streamlit run app/browser.py` on port 8501).
4. Add no secrets. Wait for the build, then open the Space URL.

Hugging Face Spaces also sleep when idle, so the same cold-start warning applies.

## If something goes wrong

- **"No module named ..."** on build: the module is missing from `requirements.txt`. Add it, push, and the app redeploys.
- **The app shows "No cache yet":** `results/browser_cache.json` is missing from the deployed branch. Check that it is committed on `master`.
- **The build fails on the Python version:** reopen Advanced settings and choose Python 3.11.
