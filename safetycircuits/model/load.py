"""Load Gemma-2-2B-it, format prompts with its chat template and generate. Needs torch."""

import torch

from safetycircuits.config import MODEL_ID, hf_token


def load_model(model_id: str = MODEL_ID):
    """Returns (model, tokenizer). Needs HF_TOKEN and an accepted Gemma licence.

    bf16 where the GPU supports it (P100 and T4 do not), otherwise fp32 (10.5 GB, fits 16 GB).
    Eager attention is the setting Google recommends for Gemma-2 (attention soft-capping).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cuda = torch.cuda.is_available()
    native_bf16 = cuda and torch.cuda.is_bf16_supported(including_emulation=False)
    dtype = torch.bfloat16 if native_bf16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token(), padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token(),
        dtype=dtype,
        attn_implementation="eager",
        device_map="cuda" if cuda else None,
    )
    model.eval()
    return model, tokenizer


def chat_prompt(tokenizer, prompt: str) -> str:
    """The prompt as the model sees it, ending where its answer begins."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def encode(tokenizer, prompts: list[str], device):
    texts = [chat_prompt(tokenizer, p) for p in prompts]
    batch = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    return batch["input_ids"].to(device), batch["attention_mask"].to(device)


@torch.no_grad()
def generate_replies(model, tokenizer, prompts: list[str], max_new_tokens: int = 48) -> list[str]:
    """Greedy replies. Callers must reduce them to a verdict and not store harmful replies."""
    input_ids, mask = encode(tokenizer, prompts, model.device)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return tokenizer.batch_decode(out[:, input_ids.shape[1] :], skip_special_tokens=True)
