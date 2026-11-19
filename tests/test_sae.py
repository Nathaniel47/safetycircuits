import numpy as np
import pytest

from safetycircuits.refusal.classify import is_refusal, looks_degenerate
from safetycircuits.sae.features import (
    decoder_cosines,
    direction_overlap,
    feature_auroc,
    neuronpedia_url,
    refusal_feature_score,
    top_features,
)
from safetycircuits.sae.load_sae import JumpReLUSAE, from_arrays, fvu, mean_l0, pick_l0_file


def tiny_sae(d=4, n=6, seed=0):
    rng = np.random.default_rng(seed)
    return JumpReLUSAE(
        W_enc=rng.standard_normal((d, n)).astype(np.float32),
        W_dec=rng.standard_normal((n, d)).astype(np.float32),
        b_enc=np.zeros(n, dtype=np.float32),
        b_dec=np.zeros(d, dtype=np.float32),
        threshold=np.full(n, 0.5, dtype=np.float32),
    )


def test_jumprelu_zeroes_values_below_threshold_and_keeps_the_rest():
    sae = JumpReLUSAE(
        W_enc=np.eye(3, dtype=np.float32),
        W_dec=np.eye(3, dtype=np.float32),
        b_enc=np.zeros(3, dtype=np.float32),
        b_dec=np.zeros(3, dtype=np.float32),
        threshold=np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )
    x = np.array([[0.5, 2.0, -3.0]], dtype=np.float32)
    feats = sae.encode(x)
    assert feats.tolist() == [[0.0, 2.0, 0.0]]  # 0.5 is below the jump, -3 is negative
    assert mean_l0(feats) == 1.0


def test_encode_decode_shapes_and_fvu():
    sae = tiny_sae()
    x = np.random.default_rng(1).standard_normal((10, 4)).astype(np.float32)
    recon = sae.decode(sae.encode(x))
    assert recon.shape == x.shape
    assert fvu(x, x) == 0.0
    assert abs(fvu(x, np.tile(x.mean(axis=0), (10, 1))) - 1.0) < 1e-6


def test_pick_l0_file_chooses_closest_to_target():
    files = [
        "layer_24/width_16k/average_l0_20/params.npz",
        "layer_24/width_16k/average_l0_73/params.npz",
        "layer_24/width_16k/average_l0_158/params.npz",
        "layer_24/width_65k/average_l0_99/params.npz",
        "layer_17/width_16k/average_l0_77/params.npz",
    ]
    assert pick_l0_file(files, 24, "16k", 100).endswith("average_l0_73/params.npz")
    assert pick_l0_file(files, 17, "16k", 100).endswith("average_l0_77/params.npz")
    with pytest.raises(FileNotFoundError):
        pick_l0_file(files, 5, "16k", 100)


def test_from_arrays_checks_keys():
    with pytest.raises(KeyError, match="threshold"):
        from_arrays({"W_enc": 1, "W_dec": 1, "b_enc": 1, "b_dec": 1})


def test_feature_auroc_handles_ties_and_direction():
    pos = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 5.0]])
    neg = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 6.0], [0.0, 0.0, 7.0]])
    auc = feature_auroc(pos, neg)
    assert auc[0] == 1.0  # always higher on pos
    assert auc[1] == 0.5  # all zeros: pure ties
    assert auc[2] < 0.5  # lower on pos


def planted(n=80, nf=30, seed=0):
    """Feature 7 fires on positives only; feature 3 fires on both; the rest is sparse noise."""
    rng = np.random.default_rng(seed)

    def make(positive):
        x = (rng.random((n, nf)) < 0.1) * rng.random((n, nf))
        x[:, 3] = rng.random(n) + 1
        if positive:
            x[:, 7] = rng.random(n) + 2
        return x

    return make(True), make(False)


def test_top_features_finds_planted_feature_and_checks_val():
    tr_p, tr_n = planted(seed=0)
    va_p, va_n = planted(seed=1)
    rows = top_features(tr_p, tr_n, va_p, va_n, k=3)
    assert rows[0]["feature"] == 7
    assert rows[0]["train_auroc"] > 0.99 and rows[0]["val_auroc"] > 0.99
    assert rows[0]["fire_rate_pos"] == 1.0 and rows[0]["fire_rate_neg"] < 0.3
    assert all(r["feature"] != 3 for r in rows[:1])  # feature on both classes is not top


def test_rare_features_are_ignored():
    tr_p = np.zeros((50, 5))
    tr_n = np.zeros((50, 5))
    tr_p[0, 2] = 9.0  # fires on 2% of positives only
    rows = top_features(tr_p, tr_n, tr_p, tr_n, k=1)
    assert rows[0]["train_auroc"] == 0.5


def test_decoder_cosines_and_overlap():
    w_dec = np.array([[1.0, 0, 0], [0, 1.0, 0], [1.0, 1.0, 0]])
    direction = np.array([1.0, 0, 0])
    cos = decoder_cosines(w_dec, [0, 1, 2], direction)
    assert np.allclose(cos, [1.0, 0.0, 2**-0.5])
    rows = [
        {"feature": 0, "mean_pos": 2.0, "mean_neg": 0.0},
        {"feature": 1, "mean_pos": 0.0, "mean_neg": 0.0},
    ]
    assert direction_overlap(w_dec, rows, direction) == pytest.approx(1.0)


def test_refusal_feature_score_is_one_on_the_scale_and_zero_when_silent():
    feats = np.array([[2.0, 4.0], [0.0, 0.0]])
    score = refusal_feature_score(feats, [0, 1], np.array([2.0, 4.0]))
    assert score.tolist() == [1.0, 0.0]


def test_neuronpedia_url():
    assert neuronpedia_url(24, 100).endswith("/gemma-2-2b/24-gemmascope-res-16k/100")


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("Je ne peux pas vous aider avec cela.", True),
        ("Samahani, siwezi kusaidia.", True),
        ("Voici un poème sur l'automne et ses couleurs.", False),
    ],
)
def test_classifier_recognises_other_study_languages(reply, expected):
    assert is_refusal(reply) is expected


def test_looks_degenerate():
    assert looks_degenerate("")
    assert looks_degenerate("ok ok")
    assert looks_degenerate("the the the the the the the the cat")
    assert not looks_degenerate("Sure, here is a short and coherent answer to your question.")
