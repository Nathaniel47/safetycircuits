"""Plain PyTorch hooks on a Hugging Face Gemma-2 model. Needs torch (GPU notebooks and tests)."""

from contextlib import contextmanager

import torch


def decoder_layers(model):
    return model.model.layers


def _first(output):
    return output[0] if isinstance(output, tuple) else output


def _replace_first(output, new):
    return (new, *output[1:]) if isinstance(output, tuple) else new


@torch.no_grad()
def last_token_residuals(model, input_ids, attention_mask=None) -> torch.Tensor:
    """Residual stream after every decoder layer at the last position: (batch, n_layers, d).

    Prompts must be left-padded so position -1 is the last real token (the generation prompt).
    Hooks on the layers are used, not output_hidden_states, because Hugging Face applies the
    final norm to the last hidden state and we want the raw residual stream.
    """
    captured: list[torch.Tensor] = []
    handles = [
        layer.register_forward_hook(lambda _m, _i, out: captured.append(_first(out)[:, -1].float()))
        for layer in decoder_layers(model)
    ]
    try:
        model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()
    return torch.stack(captured, dim=1)


@contextmanager
def ablating(model, direction: torch.Tensor):
    """Remove the direction from everything written to the residual stream.

    Hooks the token embedding and the attention and MLP outputs of every layer (the tensors
    that get added to the stream), so no layer can write along the direction.
    """
    unit = direction / direction.norm()
    handles = []

    def project_out(_module, _inputs, output):
        x = _first(output)
        r = unit.to(device=x.device, dtype=x.dtype)
        return _replace_first(output, x - (x @ r).unsqueeze(-1) * r)

    handles.append(model.model.embed_tokens.register_forward_hook(project_out))
    for layer in decoder_layers(model):
        handles.append(layer.post_attention_layernorm.register_forward_hook(project_out))
        handles.append(layer.post_feedforward_layernorm.register_forward_hook(project_out))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def adding(model, layer_index: int, direction: torch.Tensor, coefficient: float):
    """Add coefficient * direction to the residual stream after one layer, at every position."""
    unit = direction / direction.norm()

    def add(_module, _inputs, output):
        x = _first(output)
        r = unit.to(device=x.device, dtype=x.dtype)
        return _replace_first(output, x + coefficient * r)

    handle = decoder_layers(model)[layer_index].register_forward_hook(add)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def layer_residuals_all(
    model, input_ids, attention_mask, layers, drop_first: bool = False
) -> dict[int, torch.Tensor]:
    """Residual stream after the given layers at every real token: {layer: (n_tokens, d)}.

    Padding positions are dropped, so the result can be fed straight to an SAE. With drop_first,
    each prompt's first real token (the BOS token) is dropped too: it has a huge norm that the
    Gemma Scope SAEs were not trained on.
    """
    captured: dict[int, torch.Tensor] = {}
    keep = attention_mask.bool()
    if drop_first:
        keep = keep & (attention_mask.cumsum(dim=1) > 1)
    handles = [
        decoder_layers(model)[i].register_forward_hook(
            lambda _m, _i, out, i=i: captured.__setitem__(i, _first(out)[keep].float())
        )
        for i in layers
    ]
    try:
        model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()
    return captured
