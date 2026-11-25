import torch

from configs.qwen2_5_0_5b import Qwen2Config
from treekv.engine.speculative import adaptive_gamma, prompt_lookup_draft, rollback_kv
from treekv.memory.block_manager import PagedKVCache
from treekv.model.sampler import speculative_verify


def test_lossless_distribution_equivalence():
    torch.manual_seed(0)
    p = torch.tensor([[0.70, 0.20, 0.10], [0.33, 0.33, 0.34]])
    q = torch.tensor([[0.20, 0.60, 0.20]])
    counts = [0, 0, 0]
    trials = 4000
    for _ in range(trials):
        draft_tok = int(torch.multinomial(q[0], 1).item())
        out_toks, _ = speculative_verify([draft_tok], q, p, temperature=1.0)
        counts[out_toks[0]] += 1
    freqs = [c / trials for c in counts]
    assert abs(freqs[0] - 0.70) < 0.025, f"Mismatch: {freqs}"
    assert abs(freqs[1] - 0.20) < 0.025, f"Mismatch: {freqs}"
    assert abs(freqs[2] - 0.10) < 0.025, f"Mismatch: {freqs}"


def test_cross_boundary_kv_rollback_zero_leaks():
    cfg = Qwen2Config(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    )
    paged = PagedKVCache(cfg, num_blocks=10, block_size=16)
    table: list[int] = []
    paged.ensure_capacity(table, 18)
    assert len(table) == 2 and paged.allocator.num_free() == 8

    rollback_kv(paged, table, new_valid_len=16)
    assert len(table) == 1 and paged.allocator.num_free() == 9

    paged.allocator.free_table(table)
    assert paged.allocator.num_free() == 10


def test_prompt_lookup_and_adaptive_gamma():
    hist = [10, 20, 30, 40, 50, 60, 10, 20, 30]
    assert prompt_lookup_draft(hist, gamma=3, ngram=3) == [40, 50, 60]
    assert adaptive_gamma(1) == 4
    assert adaptive_gamma(6) == 2
    assert adaptive_gamma(16) == 0
