from treekv.memory.block_manager import PagedKVCache


def rollback_kv(paged_cache: PagedKVCache, block_table: list[int], new_valid_len: int) -> None:
    needed = (new_valid_len + paged_cache.block_size - 1) // paged_cache.block_size
    while len(block_table) > needed:
        extra_bid = block_table.pop()
        paged_cache.allocator.free(extra_bid)


def prompt_lookup_draft(history_ids: list[int], gamma: int = 4, ngram: int = 3) -> list[int]:
    if len(history_ids) < ngram + 1 or gamma <= 0:
        return []
    suffix = history_ids[-ngram:]
    for i in range(len(history_ids) - ngram - 1, -1, -1):
        if history_ids[i : i + ngram] == suffix:
            start = i + ngram
            return history_ids[start : min(start + gamma, len(history_ids))]
    return []


def adaptive_gamma(active_batch_size: int, max_gamma: int = 4) -> int:
    # ADR-005: Shrink speculation as batch approaches GPU compute-bound knee
    if active_batch_size <= 4:
        return max_gamma
    if active_batch_size <= 8:
        return max(1, max_gamma // 2)
    return 0
