import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.qwen2_5_0_5b import Qwen2Config
from treekv.model.qwen2 import Qwen2ForCausalLM


def test_kv_cache_matches_full_forward_offline():
    # Fast 0.1-second synthetic test (runs with zero internet/downloads)
    torch.manual_seed(0)
    tiny_cfg = Qwen2Config(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    )
    model = Qwen2ForCausalLM(tiny_cfg).eval()
    ids = torch.tensor([[1, 5, 9, 12, 3]])
    pos = torch.arange(5).unsqueeze(0)

    with torch.no_grad():
        full_logits, _ = model(ids, pos)
        _, kv = model(ids[:, :3], pos[:, :3])
        _, kv = model(ids[:, 3:4], pos[:, 3:4], kv_caches=kv)
        last_logits, _ = model(ids[:, 4:5], pos[:, 4:5], kv_caches=kv)

    max_diff = (full_logits[:, 4, :] - last_logits[:, 0, :]).abs().max().item()
    assert max_diff < 1e-5, f"KV cache drift: {max_diff}"


def test_exact_match_vs_huggingface_qwen2_5():
    model_id = "Qwen/Qwen2.5-0.5B"
    tok = AutoTokenizer.from_pretrained(model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    my_model = Qwen2ForCausalLM.from_huggingface(hf_model, Qwen2Config()).eval()

    prompts = [
        "The capital of France is",
        "Solve step by step: If 3x + 7 = 22, then x =",
        "def fibonacci(n):",
    ]

    for text in prompts:
        ids = tok(text, return_tensors="pt")["input_ids"]
        pos = torch.arange(ids.shape[1]).unsqueeze(0)

        with torch.no_grad():
            hf_logits = hf_model(ids).logits
            my_logits, kv = my_model(ids, pos)

        max_diff = (hf_logits - my_logits).abs().max().item()
        print(f"\n[{text[:25]}...] max logit diff = {max_diff:.2e}")
        assert max_diff < 1e-4, f"Logit mismatch on '{text}': {max_diff}"

        generated = ids.clone()
        next_tok = my_logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_tok], dim=1)
        for step in range(14):
            cur_pos = torch.tensor([[ids.shape[1] + step]])
            with torch.no_grad():
                step_logits, kv = my_model(next_tok, cur_pos, kv_caches=kv)
            next_tok = step_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)

        hf_out = hf_model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            pad_token_id=tok.eos_token_id,
            max_new_tokens=15,
            do_sample=False,
        )
        assert torch.equal(generated, hf_out), f"Token mismatch on '{text}'!"