"""Download the public datasets into data/raw/ (gitignored) and record checksums.

Run: python -m safetycircuits.data.fetch
"""

import hashlib
import json
import tarfile
import urllib.request
from datetime import date

from safetycircuits.config import DATA_DIR, LANGUAGES

RAW = DATA_DIR / "raw"
FLORES_CODES = ["eng_Latn", *LANGUAGES.values()]

SOURCES = {
    "advbench.csv": {
        "url": "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
        "licence": "MIT",
    },
    "harmbench.csv": {
        "url": "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv",
        "licence": "MIT",
    },
    "xstest.csv": {
        "url": "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv",
        "licence": "CC-BY-4.0",
    },
    "alpaca.json": {
        "url": "https://huggingface.co/datasets/yahma/alpaca-cleaned/resolve/main/alpaca_data_cleaned.json",
        "licence": "CC-BY-4.0 on the dataset card (disputed: text came from OpenAI models)",
    },
    "flores.tgz": {
        "url": "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz",
        "licence": "CC-BY-SA 4.0",
    },
}


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(name: str, force: bool = False) -> None:
    dest = RAW / name
    if dest.exists() and not force:
        return
    req = urllib.request.Request(SOURCES[name]["url"], headers={"User-Agent": "safetycircuits"})
    with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as out:
        while block := resp.read(1 << 20):
            out.write(block)


def extract_flores() -> None:
    """Keep only the dev/devtest files we need, flat, under data/raw/flores/."""
    out = RAW / "flores"
    out.mkdir(exist_ok=True)
    wanted = {f"{c}.{split}" for c in FLORES_CODES for split in ("dev", "devtest")}
    with tarfile.open(RAW / "flores.tgz") as tar:
        for member in tar:
            name = member.name.rsplit("/", 1)[-1]
            if member.isfile() and name in wanted:
                (out / name).write_bytes(tar.extractfile(member).read())


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, meta in SOURCES.items():
        download(name)
        path = RAW / name
        manifest[name] = {
            **meta,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "fetched": date.today().isoformat(),
        }
    extract_flores()
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"fetched {len(manifest)} sources; manifest written")


if __name__ == "__main__":
    main()
