"""Checks that the repo is ready for a free Streamlit host. No network, no model."""

import json
import re

from safetycircuits.config import ROOT

HEAVY = ("torch", "transformers", "accelerate", "huggingface", "sae-lens", "tensorflow")


def test_runtime_requirements_are_light():
    lines = (ROOT / "requirements.txt").read_text().lower().splitlines()
    names = [re.split(r"[<>=!~\[ ]", ln)[0] for ln in lines if ln.strip() and ln[0] != "#"]
    assert names, "requirements.txt is empty"
    assert not [n for n in names if any(h in n for h in HEAVY)]
    assert "streamlit" in names and "plotly" in names


def test_app_entry_point_and_config_exist():
    assert (ROOT / "app" / "browser.py").exists()
    config = (ROOT / ".streamlit" / "config.toml").read_text()
    assert "gatherUsageStats = false" in config


def test_cache_and_charts_are_committed_and_small():
    cache = ROOT / "results" / "browser_cache.json"
    assert cache.exists() and cache.stat().st_size < 2_000_000
    for name in ("chart_layer_sweep.png", "chart_jailbreaks.png", "chart_crosslingual.png"):
        assert (ROOT / "results" / name).stat().st_size < 500_000, name
    assert len(json.loads(cache.read_text(encoding="utf-8"))["prompts"]) > 100


def test_no_secrets_or_tokens_in_tracked_source():
    patterns = [re.compile(r"hf_[A-Za-z0-9]{20,}"), re.compile(r"AKIA[0-9A-Z]{16}")]
    for path in list((ROOT / "safetycircuits").rglob("*.py")) + [ROOT / "app" / "browser.py"]:
        text = path.read_text(encoding="utf-8")
        assert not any(p.search(text) for p in patterns), path


def test_gitignore_keeps_env_and_raw_data_out():
    ignore = (ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in ignore and "data/raw/" in ignore and "data/interim/" in ignore


def test_deploy_guide_exists_and_names_the_entry_point():
    text = (ROOT / "DEPLOY.md").read_text(encoding="utf-8")
    assert "app/browser.py" in text and "Streamlit Community Cloud" in text
