from collections import deque

from treekv.engine.sequence import SeqStatus, Sequence
from treekv.interfaces import PrefixIndexPort
from treekv.memory.block_manager import PagedKVCache


class ContinuousScheduler:
    def __init__(
        self,
        paged_cache: PagedKVCache,
        max_batch_size: int = 16,
        radix_cache: PrefixIndexPort | None = None,
    ):
        self.cache = paged_cache
        self.radix = radix_cache
        self.max_batch_size = max_batch_size
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.finished: list[Sequence] = []
        self.preemption_count: int = 0

    def add_request(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_pending(self) -> bool:
        return bool(self.waiting or self.running)

    def _ensure_free_blocks(self, needed: int) -> bool:
        if self.cache.allocator.num_free() < needed and self.radix is not None:
            shortfall = needed - self.cache.allocator.num_free()
            self.radix.evict_lru(shortfall)
        return self.cache.allocator.num_free() >= needed

    def _preempt_newest(self) -> None:
        victim = self.running.pop()
        if self.radix is not None:
            self.radix.release_sequence(
                victim.block_table,
                victim.all_token_ids,
                self.cache.block_size,
                cache_full_blocks=False,
            )
        else:
            self.cache.allocator.free_table(victim.block_table)
        victim.block_table = []
        victim.num_computed_tokens = 0
        victim.status = SeqStatus.WAITING
        self.waiting.appendleft(victim)
        self.preemption_count += 1

    def schedule(self) -> tuple[list[Sequence], list[Sequence]]:
        still_running: list[Sequence] = []
        for seq in self.running:
            if seq.status == SeqStatus.FINISHED:
                if self.radix is not None:
                    self.radix.release_sequence(
                        seq.block_table,
                        seq.all_token_ids,
                        self.cache.block_size,
                        cache_full_blocks=True,
                    )
                else:
                    self.cache.allocator.free_table(seq.block_table)
                seq.block_table = []
                self.finished.append(seq)
            else:
                still_running.append(seq)
        self.running = still_running

        prefill_batch: list[Sequence] = []
        while self.waiting and len(self.running) < self.max_batch_size:
            seq = self.waiting[0]
            matched_blocks: list[int] = []
            matched_toks = 0
            if self.radix is not None and seq.total_len > 1:
                matched_blocks, matched_toks = self.radix.match_prefix(
                    seq.all_token_ids, max_match=seq.total_len - 1
                )

            total_needed = (seq.total_len + self.cache.block_size - 1) // self.cache.block_size
            fresh_needed = max(0, total_needed - len(matched_blocks))

            if not self._ensure_free_blocks(fresh_needed):
                if self.radix is not None and matched_blocks:
                    self.radix.unlock_blocks(matched_blocks)
                break

            self.waiting.popleft()
            seq.block_table = list(matched_blocks)
            seq.num_computed_tokens = matched_toks
            self.cache.ensure_capacity(seq.block_table, seq.total_len)
            seq.status = SeqStatus.RUNNING
            self.running.append(seq)
            prefill_batch.append(seq)

        decode_batch: list[Sequence] = []
        i = 0
        while i < len(self.running):
            seq = self.running[i]
            if seq in prefill_batch:
                i += 1
                continue
            needed_total = (seq.total_len + 1 + self.cache.block_size - 1) // self.cache.block_size
            extra = max(0, needed_total - len(seq.block_table))
            while extra > 0 and not self._ensure_free_blocks(extra):
                if len(self.running) <= 1:
                    raise RuntimeError("Single sequence exceeds total GPU KV pool!")
                self._preempt_newest()
                if i >= len(self.running):
                    break
            if i < len(self.running) and self.running[i] is seq:
                self.cache.ensure_capacity(seq.block_table, seq.total_len + 1)
                decode_batch.append(seq)
                i += 1

        return prefill_batch, decode_batch
