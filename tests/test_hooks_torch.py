"""Hook tests on a tiny random Gemma-2 (no download). Skipped without torch/transformers."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from safetycircuits.model.hooks import ablating, adding, last_token_residuals  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = transformers.Gemma2Config(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=16,
        attn_implementation="eager",
    )
    return transformers.Gemma2ForCausalLM(cfg).eval()


@pytest.fixture
def ids():
    return torch.randint(1, 100, (4, 10))


def direction(d=32):
    torch.manual_seed(1)
    return torch.randn(d)


def test_capture_shape_and_matches_hidden_states(tiny, ids):
    res = last_token_residuals(tiny, ids)
    assert res.shape == (4, 3, 32)
    with torch.no_grad():
        hs = tiny(input_ids=ids, output_hidden_states=True).hidden_states
    assert torch.allclose(res[:, 0], hs[1][:, -1], atol=1e-4)


def test_ablation_removes_the_direction_everywhere(tiny, ids):
    r = direction()
    unit = r / r.norm()
    before = last_token_residuals(tiny, ids)
    with ablating(tiny, r):
        after = last_token_residuals(tiny, ids)
    assert (before @ unit).abs().mean() > 0.05  # the direction was present
    assert (after @ unit).abs().max() < 1e-4  # and is gone at every layer


def test_ablation_changes_logits_and_hooks_are_removed(tiny, ids):
    r = direction()
    with torch.no_grad():
        base = tiny(input_ids=ids).logits
        with ablating(tiny, r):
            changed = tiny(input_ids=ids).logits
        again = tiny(input_ids=ids).logits
    assert not torch.allclose(base, changed, atol=1e-4)
    assert torch.allclose(base, again, atol=1e-6)


def test_adding_shifts_residual_by_coefficient(tiny, ids):
    r = direction()
    unit = r / r.norm()
    before = last_token_residuals(tiny, ids)
    with adding(tiny, 0, r, 5.0):
        after = last_token_residuals(tiny, ids)
    assert torch.allclose(after[:, 0] @ unit - before[:, 0] @ unit, torch.tensor(5.0), atol=1e-3)
    assert torch.allclose(last_token_residuals(tiny, ids), before, atol=1e-6)


def test_random_control_direction_leaves_the_target_alone(tiny, ids):
    r = direction()
    unit = r / r.norm()
    other = torch.randn(32, generator=torch.Generator().manual_seed(5))
    with ablating(tiny, other):
        after = last_token_residuals(tiny, ids)
    assert (after @ unit).abs().mean() > 0.01  # ablating another direction does not remove r


def test_all_token_capture_drops_padding_and_matches_hidden_states(tiny):
    from safetycircuits.model.hooks import layer_residuals_all

    ids = torch.randint(1, 100, (2, 6))
    mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    out = layer_residuals_all(tiny, ids, mask, [0, 2])
    assert out[0].shape == (10, 32)  # 4 + 6 real tokens
    with torch.no_grad():
        hs = tiny(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
    assert torch.allclose(out[0][4:], hs[1][1], atol=1e-4)  # second prompt, layer 0


def test_all_token_capture_can_drop_the_first_real_token(tiny):
    from safetycircuits.model.hooks import layer_residuals_all

    ids = torch.randint(1, 100, (2, 6))
    mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    out = layer_residuals_all(tiny, ids, mask, [1], drop_first=True)
    assert out[1].shape == (8, 32)  # 3 + 5 tokens: each prompt loses its first real token
