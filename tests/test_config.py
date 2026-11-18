from pathlib import Path

from safetycircuits import config


def test_model_and_sae_ids():
    assert config.MODEL_ID == "google/gemma-2-2b-it"
    assert config.SAE_REPO.startswith("google/gemma-scope-2b")


def test_languages_use_flores_codes():
    assert set(config.LANGUAGES.values()) == {"twi_Latn", "fra_Latn", "swh_Latn"}


def test_hf_token_comes_from_environment(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert config.hf_token() is None
    monkeypatch.setenv("HF_TOKEN", "x")
    assert config.hf_token() == "x"


def test_env_example_has_names_only():
    lines = (config.ROOT / ".env.example").read_text().splitlines()
    for line in lines:
        if line.strip() and not line.startswith("#"):
            name, _, value = line.partition("=")
            assert name and value == "", f"{name} must have an empty value"


def test_env_is_gitignored():
    assert ".env" in (Path(config.ROOT) / ".gitignore").read_text().splitlines()
