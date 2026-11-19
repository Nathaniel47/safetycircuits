"""Gemma Scope JumpReLU SAEs, loaded straight from the Hugging Face files as numpy arrays.

SAELens is not used: its loading API changed between versions, and installing it upgraded
torch in the Kaggle image. The SAE maths is a few lines and is tested here.
"""

import re
from dataclasses import dataclass

import numpy as np

from safetycircuits.config import SAE_REPO, SAE_TARGET_L0, SAE_WIDTH, hf_token

REQUIRED_KEYS = ("W_enc", "W_dec", "b_enc", "b_dec", "threshold")


@dataclass
class JumpReLUSAE:
    W_enc: np.ndarray  # (d_model, n_features)
    W_dec: np.ndarray  # (n_features, d_model)
    b_enc: np.ndarray
    b_dec: np.ndarray
    threshold: np.ndarray

    @property
    def n_features(self) -> int:
        return self.W_enc.shape[1]

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Feature activations (n, n_features). A feature fires only above its own threshold."""
        pre = x @ self.W_enc + self.b_enc
        return pre * (pre > self.threshold)

    def decode(self, feats: np.ndarray) -> np.ndarray:
        return feats @ self.W_dec + self.b_dec


def pick_l0_file(files: list[str], layer: int, width: str, target: int) -> str:
    """The params file for this layer and width whose average L0 is closest to the target."""
    pattern = re.compile(rf"layer_{layer}/width_{width}/average_l0_(\d+)/params\.npz$")
    found = [(int(m.group(1)), f) for f in files if (m := pattern.match(f))]
    if not found:
        raise FileNotFoundError(f"no SAE for layer {layer}, width {width}")
    return min(found, key=lambda t: abs(t[0] - target))[1]


def from_arrays(arrays) -> JumpReLUSAE:
    missing = [k for k in REQUIRED_KEYS if k not in arrays]
    if missing:
        raise KeyError(f"SAE file is missing {missing}; found {list(arrays)}")
    return JumpReLUSAE(**{k: np.asarray(arrays[k], dtype=np.float32) for k in REQUIRED_KEYS})


def load_sae(layer: int, width: str = SAE_WIDTH, target_l0: int = SAE_TARGET_L0) -> JumpReLUSAE:
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(SAE_REPO, token=hf_token())
    name = pick_l0_file(files, layer, width, target_l0)
    path = hf_hub_download(SAE_REPO, name, token=hf_token())
    return from_arrays(np.load(path))


def fvu(x: np.ndarray, recon: np.ndarray) -> float:
    """Fraction of variance unexplained: 0 is perfect, 1 is no better than the mean."""
    return float(((x - recon) ** 2).sum() / ((x - x.mean(axis=0)) ** 2).sum())


def mean_l0(feats: np.ndarray) -> float:
    return float((feats > 0).sum(axis=1).mean())
