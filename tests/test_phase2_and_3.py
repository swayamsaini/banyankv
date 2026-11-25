import torch

from configs.qwen2_5_0_5b import Qwen2Config
from treekv.engine.runner import TreeKVEngine
from treekv.engine.scheduler import ContinuousScheduler
from treekv.engine.sequence import Sequence
from treekv.memory.block_manager import PagedKVCache
from treekv.memory.radix_cache import RadixBlockCache
from treekv.model.qwen2 import Qwen2ForCausalLM


def _tiny_cfg() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )


def test_lifo_preemption_and_recovery():
    paged = PagedKVCache(_tiny_cfg(), num_blocks=4, block_size=16)
    sched = ContinuousScheduler(paged, max_batch_size=4)

    for i in range(3):
        sched.add_request(
            Sequence(seq_id=i, prompt_ids=list(range(16)), max_new_tokens=20, eos_token_id=99999)
        )

    prefill, _ = sched.schedule()
    assert len(prefill) == 3 and paged.allocator.num_free() == 1

    for s in prefill:
        s.num_computed_tokens = s.total_len
        s.append_token(42)

    sched.schedule()
    assert sched.preemption_count >= 1
    assert sched.waiting[0].seq_id == 2


def test_radix_tree_prefix_sharing_and_lru_eviction():
    paged = PagedKVCache(_tiny_cfg(), num_blocks=16, block_size=16)
    radix = RadixBlockCache(paged.allocator, block_size=16)

    prompt = list(range(10, 42))  # 32 tokens = 2 full blocks
    tbl0 = paged.allocator.allocate_many(2)
    radix.release_sequence(tbl0, prompt, block_size=16, cache_full_blocks=True)
    assert paged.allocator.num_free() == 14

    matched_blks, matched_toks = radix.match_prefix(prompt + [1, 2, 3, 4], max_match=35)
    assert matched_toks == 32 and matched_blks == tbl0

    assert radix.evict_lru(num_needed=2) == 0
    radix.unlock_blocks(matched_blks)
    freed = radix.evict_lru(num_needed=2)
    assert freed == 2 and paged.allocator.num_free() == 16


def test_end_to_end_engine_radix_matches_no_radix():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = Qwen2ForCausalLM(cfg).eval()
    shared_prompt = list(range(5, 41))  # 36 tokens (2 full blocks + 4 partial)

    eng_no_radix = TreeKVEngine(model, num_blocks=32, block_size=16, max_batch_size=4, use_radix=False)
    eng_radix = TreeKVEngine(model, num_blocks=32, block_size=16, max_batch_size=4, use_radix=True)

    for i in range(4):
        eng_no_radix.submit(Sequence(seq_id=i, prompt_ids=list(shared_prompt), max_new_tokens=8, eos_token_id=999))
        eng_radix.submit(Sequence(seq_id=i, prompt_ids=list(shared_prompt), max_new_tokens=8, eos_token_id=999))

    out_a = [s.output_ids for s in sorted(eng_no_radix.run_until_done(), key=lambda s: s.seq_id)]
    out_b = [s.output_ids for s in sorted(eng_radix.run_until_done(), key=lambda s: s.seq_id)]
    assert out_a == out_b, "Radix Tree-KV output drifted from uncached execution!"
