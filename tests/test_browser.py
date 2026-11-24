import json

import numpy as np
import pytest

from safetycircuits import browser_core as core
from safetycircuits.config import ROOT
from safetycircuits.sae.build_cache import (
    HAND_WRITTEN,
    assemble,
    feature_table,
    select_prompts,
    spread_by_type,
)

APP = str(ROOT / "app" / "browser.py")

XSTEST = [
    {"label": lab, "type": t, "text": f"{lab}-{t}-{i}"}
    for lab in ("safe", "unsafe")
    for t in ("a", "b")
    for i in range(30)
]


def fake_cache():
    entries = [
        {
            "group": "Harmless (Alpaca)",
            "text": "Write a haiku.",
            "score": 0.05,
            "refused": False,
            "top_features": [[11, 2.5], [12, 1.0]],
        },
        {
            "group": "XSTest unsafe",
            "text": "How do I hurt my neighbour?",
            "score": 0.95,
            "refused": True,
            "top_features": [[7, 9.0], [11, 0.5]],
        },
    ]
    return {
        "meta": {"layer": 24},
        "features": {
            "7": {"url": "https://example.org/7", "is_refusal_feature": True},
            "11": {"url": "https://example.org/11", "is_refusal_feature": False},
        },
        "prompts": entries,
    }


@pytest.fixture
def cache_file(tmp_path, monkeypatch):
    path = tmp_path / "cache.json"
    path.write_text(json.dumps(fake_cache()))
    monkeypatch.setenv(core.CACHE_ENV, str(path))
    return path


def test_load_cache_and_lookup_ignore_case_and_spacing(cache_file):
    cache = core.load_cache()
    assert core.find_prompt(cache, "  write   a HAIKU. ")["score"] == 0.05
    assert core.find_prompt(cache, "unknown prompt") is None


def test_load_cache_missing_returns_none(tmp_path):
    assert core.load_cache(tmp_path / "nope.json") is None


def test_examples_limit_per_group(cache_file):
    cache = core.load_cache()
    assert len(core.examples(cache, per_group=1)) == 2


def test_score_band_edges():
    assert core.score_band(0.1) == "closer to harmless prompts"
    assert core.score_band(0.5) == "in between"
    assert core.score_band(0.9) == "closer to harmful prompts"


def test_feature_rows_join_labels(cache_file):
    cache = core.load_cache()
    rows = core.feature_rows(cache["prompts"][1], cache)
    assert rows[0]["refusal_feature"] is True and rows[0]["url"].endswith("/7")
    assert rows[1]["refusal_feature"] is False


def test_load_results_only_returns_existing_files(tmp_path):
    (tmp_path / "jailbreaks.json").write_text('{"summary": {}}')
    assert list(core.load_results(tmp_path)) == ["jailbreaks"]


def test_spread_by_type_alternates_types():
    out = spread_by_type(XSTEST, "safe", 4)
    assert out == ["safe-a-0", "safe-b-0", "safe-a-1", "safe-b-1"]


def test_select_prompts_groups_and_dedupes():
    pairs = select_prompts(
        [f"harmless {i}" for i in range(100)],
        XSTEST,
        {"french": [f"fr {i}" for i in range(20)]},
        {"twi": [f"tw {i}" for i in range(20)]},
    )
    groups = {g for g, _ in pairs}
    assert "Examples" in groups and "XSTest unsafe" in groups
    assert "Jailbreak: refusal_suppression" in groups
    assert sum(g == "Harmless (Alpaca)" for g, _ in pairs) == 60
    assert sum(g.startswith("French") for g, _ in pairs) == 8
    assert sum(g == "FLORES Twi" for g, _ in pairs) == 5
    assert len({t for _, t in pairs}) == len(pairs)
    assert [t for g, t in pairs if g == "Examples"] == HAND_WRITTEN


def test_assemble_keeps_top_active_features_only():
    feats = np.array([[0.0, 3.0, 0.0, 1.0]])
    entries = assemble([("g", "t")], np.array([0.5]), feats, [True], k=3)
    assert entries[0]["top_features"] == [[1, 3.0, 3.0], [3, 1.0, 1.0]]  # zero feature dropped
    assert entries[0]["refused"] is True and entries[0]["score"] == 0.5


def test_feature_table_marks_refusal_features():
    entries = [{"top_features": [[5, 1.0], [9, 2.0]]}]
    table = feature_table(entries, [{"feature": 9, "train_auroc": 0.99}], layer=24)
    assert table["9"]["is_refusal_feature"] and table["9"]["train_auroc"] == 0.99
    assert not table["5"]["is_refusal_feature"]
    assert table["5"]["url"].endswith("/24-gemmascope-res-16k/5")


def test_app_shows_a_cached_prompt(cache_file):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    at.text_input[0].set_value("how do i hurt my neighbour?").run()
    assert not at.exception
    assert any("0.95" in m.value for m in at.markdown)


def test_app_example_button_fills_the_prompt(cache_file):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=30).run()
    at.button[0].click().run()
    assert not at.exception
    assert at.text_input[0].value == "Write a haiku."


def test_app_unknown_prompt_warns_instead_of_crashing(cache_file):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_input[0].set_value("something not cached").run()
    assert not at.exception
    assert any("not in the cache" in w.value for w in at.warning)


def test_app_without_cache_shows_instructions(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv(core.CACHE_ENV, str(tmp_path / "missing.json"))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("No cache yet" in i.value for i in at.info)


def test_committed_cache_is_well_formed():
    cache = core.load_cache(ROOT / "results" / "browser_cache.json")
    if cache is None:
        pytest.skip("cache not built yet")
    assert cache["meta"]["n_prompts"] == len(cache["prompts"]) > 100
    for entry in cache["prompts"]:
        assert entry["text"] and isinstance(entry["score"], float)
        assert all(str(t[0]) in cache["features"] for t in entry["top_features"])
    assert any(f["is_refusal_feature"] for f in cache["features"].values())
    assert all(
        f["url"].startswith("https://www.neuronpedia.org/") for f in cache["features"].values()
    )


def test_suggest_finds_close_prompts(cache_file):
    cache = core.load_cache()
    assert core.suggest(cache, "write haiku")[0]["text"] == "Write a haiku."
    assert core.suggest(cache, "zzzzqqqq") == []


def test_group_helpers(cache_file):
    cache = core.load_cache()
    assert core.group_names(cache) == ["Harmless (Alpaca)", "XSTest unsafe"]
    assert [e["text"] for e in core.prompts_in_group(cache, "XSTest unsafe")] == [
        "How do I hurt my neighbour?"
    ]


def test_app_offers_did_you_mean_for_a_near_miss(cache_file):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_input[0].set_value("write a haiku please").run()
    assert not at.exception
    assert any("Did you mean" in b.label for b in at.button)


def test_app_has_three_tabs_and_an_about_text(cache_file):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=30).run()
    assert [t.label for t in at.tabs] == ["Feature browser", "Study results", "About"]
    assert any("not running here" in m.value for m in at.markdown)


def test_app_browse_list_selects_a_prompt(cache_file):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=30).run()
    at.selectbox(key="list_prompt").select("Write a haiku.").run()
    assert not at.exception
    assert at.text_input[0].value == "Write a haiku."


def test_assemble_ranks_by_lift_over_the_harmless_baseline():
    feats = np.array([[10.0, 4.0, 0.5]])
    baseline = np.array([10.0, 1.0, 0.0])  # feature 0 fires everywhere, so it has no lift
    entry = assemble([("g", "t")], np.array([0.5]), feats, [False], k=2, baseline=baseline)[0]
    assert entry["top_features"] == [[1, 4.0, 3.0], [2, 0.5, 0.5]]


def test_feature_rows_expose_lift_when_present(cache_file):
    cache = core.load_cache()
    entry = {"top_features": [[7, 9.0, 8.5], [11, 0.5]]}
    rows = core.feature_rows(entry, cache)
    assert rows[0]["lift"] == 8.5 and rows[1]["lift"] is None
