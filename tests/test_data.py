import json

import pytest

from safetycircuits.config import DATA_DIR, RESULTS_DIR
from safetycircuits.data.fetch import SOURCES
from safetycircuits.data.prepare import dedupe, key, normalise
from safetycircuits.data.splits import split_ids

FRACTIONS = {"train": 0.6, "val": 0.2, "test": 0.2}


def make_groups():
    return {"a": [f"a{i}" for i in range(50)], "b": [f"b{i}" for i in range(20)]}


def test_split_is_deterministic_and_seed_dependent():
    one = split_ids(make_groups(), FRACTIONS, seed=0)
    assert one == split_ids(make_groups(), FRACTIONS, seed=0)
    assert one != split_ids(make_groups(), FRACTIONS, seed=1)


def test_split_is_disjoint_complete_and_stratified():
    groups = make_groups()
    out = split_ids(groups, FRACTIONS, seed=0)
    everything = [i for ids in out.values() for i in ids]
    assert sorted(everything) == sorted(i for ids in groups.values() for i in ids)
    for name in out:
        assert any(i.startswith("a") for i in out[name])
        assert any(i.startswith("b") for i in out[name])


def test_normalise_and_dedupe():
    assert normalise("  Write   a\n script ") == "Write a script"
    assert key("Write a script!") == key("write  a SCRIPT")
    items = [{"text": "Write a script!"}, {"text": "write a script"}, {"text": "Other"}]
    kept, dropped = dedupe(items, set())
    assert len(kept) == 2 and dropped == 1


def test_every_source_has_a_licence_and_https_url():
    for name, meta in SOURCES.items():
        assert meta["licence"], name
        assert meta["url"].startswith("https://"), name


def test_committed_splits_are_consistent():
    splits = json.loads((DATA_DIR / "splits.json").read_text())
    for group in ("harmful", "harmless"):
        parts = [set(ids) for ids in splits[group].values()]
        assert sum(len(p) for p in parts) == len(set().union(*parts)), "splits overlap"
    assert set(splits["translate_ids"]) <= set(splits["harmful"]["test"])
    assert len(splits["translate_ids"]) == 100
    assert len(splits["xstest_ids"]) == 450


def test_splits_contain_ids_not_text():
    raw = (DATA_DIR / "splits.json").read_text()
    for token in json.loads(raw)["harmful"]["train"][:20]:
        assert token.startswith(("advbench-", "harmbench-"))


@pytest.mark.skipif(not (DATA_DIR / "interim" / "harmful.jsonl").exists(), reason="needs data")
def test_report_does_not_leak_harmful_text():
    report = (RESULTS_DIR / "data_report.md").read_text(encoding="utf-8")
    with open(DATA_DIR / "interim" / "harmful.jsonl", encoding="utf-8") as f:
        for line in list(f)[:100]:
            assert json.loads(line)["text"] not in report
