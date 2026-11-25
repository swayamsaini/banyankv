import random

import torch

from configs.qwen2_5_0_5b import Qwen2Config
from treekv.memory.block_manager import PagedKVCache
from treekv.model.qwen2 import Qwen2ForCausalLM


def test_paged_kv_shuffled_matches_phase0_offline():
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    model = Qwen2ForCausalLM(cfg).eval()

    paged = PagedKVCache(cfg, num_blocks=16, block_size=16)
    shuffled = list(paged.allocator.free_blocks)
    random.Random(42).shuffle(shuffled)
    paged.allocator.free_blocks.clear()
    paged.allocator.free_blocks.extend(shuffled)

    # 20-token prompt (> 16 block boundary!) + 20 decode steps (crosses 2nd boundary at 32!)
    ids = torch.randint(0, 128, (1, 20))
    P = ids.shape[1]
    pos0 = torch.arange(P).unsqueeze(0)

    # Phase 0 baseline
    with torch.no_grad():
        logits0, kv0 = model(ids, pos0)
    out0 = [logits0[:, -1, :].argmax(dim=-1).item()]
    for step in range(20):
        nxt = torch.tensor([[out0[-1]]])
        with torch.no_grad():
            l_step, kv0 = model(nxt, torch.tensor([[P + step]]), kv_caches=kv0)
        out0.append(l_step[:, -1, :].argmax(dim=-1).item())

    # Phase 1 Paged KV on shuffled blocks
    block_table = []
    paged.ensure_capacity(block_table, P)
    with torch.no_grad():
        logits1, _ = model(ids, pos0, paged_cache=paged, block_table=block_table, seq_len=P)
    out1 = [logits1[:, -1, :].argmax(dim=-1).item()]
    for step in range(20):
        cur_len = P + step + 1
        paged.ensure_capacity(block_table, cur_len)
        nxt = torch.tensor([[out1[-1]]])
        with torch.no_grad():
            l_step, _ = model(
                nxt,
                torch.tensor([[P + step]]),
                paged_cache=paged,
                block_table=block_table,
                seq_len=cur_len,
            )
        out1.append(l_step[:, -1, :].argmax(dim=-1).item())

    assert out0 == out1, f"Paged KV mismatch! {out0} vs {out1}"
    paged.allocator.free_table(block_table)
    assert paged.allocator.num_free() == 16, "Block leak detected!"


def test_copy_on_write_isolation():
    cfg = Qwen2Config(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    )
    paged = PagedKVCache(cfg, num_blocks=8, block_size=16)
    table_a = []
    paged.ensure_capacity(table_a, 4)
    ones = torch.ones(1, cfg.num_key_value_heads, 1, cfg.head_dim)
    paged.write_kv(0, table_a, torch.tensor([0]), ones, ones)

    table_b = paged.fork_table(table_a)
    assert table_a[0] == table_b[0] and paged.allocator.ref_count[table_a[0]] == 2

    nines = torch.full((1, cfg.num_key_value_heads, 1, cfg.head_dim), 9.0)
    paged.write_kv(0, table_b, torch.tensor([1]), nines, nines)

    assert table_a[0] != table_b[0], "CoW failed to allocate private block!"
    assert paged.allocator.ref_count[table_a[0]] == 1
    assert paged.allocator.ref_count[table_b[0]] == 1
    assert torch.all(paged.k_cache[0, table_b[0], :, 0, :] == 1.0)
    assert torch.all(paged.k_cache[0, table_b[0], :, 1, :] == 9.0)