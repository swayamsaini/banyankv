from collections import deque

import torch

from configs.qwen2_5_0_5b import Qwen2Config


class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int = 16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks: deque[int] = deque(range(num_blocks))
        self.ref_count: dict[int, int] = {i: 0 for i in range(num_blocks)}

    def num_free(self) -> int:
        return len(self.free_blocks)

    def allocate(self) -> int:
        if not self.free_blocks:
            raise RuntimeError("Out of KV blocks! Scheduler must preempt.")
        bid = self.free_blocks.popleft()
        assert self.ref_count[bid] == 0, f"Allocated block {bid} still referenced!"
        self.ref_count[bid] = 1
        return bid

    def allocate_many(self, count: int) -> list[int]:
        if count > len(self.free_blocks):
            raise RuntimeError(f"Need {count} blocks, only {len(self.free_blocks)} free.")
        return [self.allocate() for _ in range(count)]

    def share(self, bid: int) -> int:
        assert self.ref_count[bid] > 0, f"Cannot share unallocated block {bid}"
        self.ref_count[bid] += 1
        return bid

    def free(self, bid: int) -> None:
        assert self.ref_count[bid] > 0, f"Double-free detected on block {bid}!"
        self.ref_count[bid] -= 1
        if self.ref_count[bid] == 0:
            self.free_blocks.append(bid)

    def free_table(self, block_table: list[int]) -> None:
        for bid in block_table:
            self.free(bid)


class PagedKVCache:
    def __init__(
        self,
        cfg: Qwen2Config,
        num_blocks: int,
        block_size: int = 16,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.cfg = cfg
        self.block_size = block_size
        self.device = device
        self.dtype = dtype
        self.allocator = BlockAllocator(num_blocks, block_size)

        shape = (
            cfg.num_hidden_layers,
            num_blocks,
            cfg.num_key_value_heads,
            block_size,
            cfg.head_dim,
        )
        self.k_cache = torch.empty(shape, device=device, dtype=dtype)
        self.v_cache = torch.empty(shape, device=device, dtype=dtype)

    def ensure_capacity(self, block_table: list[int], total_tokens: int) -> None:
        needed = (total_tokens + self.block_size - 1) // self.block_size
        while len(block_table) < needed:
            block_table.append(self.allocator.allocate())

    def copy_on_write(self, block_table: list[int], logical_idx: int) -> int:
        old_bid = block_table[logical_idx]
        if self.allocator.ref_count[old_bid] > 1:
            new_bid = self.allocator.allocate()
            self.k_cache[:, new_bid].copy_(self.k_cache[:, old_bid])
            self.v_cache[:, new_bid].copy_(self.v_cache[:, old_bid])
            self.allocator.free(old_bid)
            block_table[logical_idx] = new_bid
            return new_bid
        return old_bid

    def fork_table(self, src_table: list[int]) -> list[int]:
        return [self.allocator.share(bid) for bid in src_table]

    def write_kv(
        self,
        layer_idx: int,
        block_table: list[int],
        positions: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        logical_ids = (positions // self.block_size).tolist()
        if layer_idx == 0:
            for l_idx in set(logical_ids):
                self.copy_on_write(block_table, l_idx)

        phys_ids = torch.tensor(
            [block_table[l_idx] for l_idx in logical_ids],
            device=self.device,
            dtype=torch.long,
        )
        slots = (positions % self.block_size).to(self.device, dtype=torch.long)

        self.k_cache[layer_idx, phys_ids, :, slots, :] = k[0].transpose(0, 1)
        self.v_cache[layer_idx, phys_ids, :, slots, :] = v[0].transpose(0, 1)

    def gather_kv(self, layer_idx: int, block_table: list[int], seq_len: int):
        num_used = (seq_len + self.block_size - 1) // self.block_size
        phys_ids = torch.tensor(block_table[:num_used], device=self.device, dtype=torch.long)

        k = (
            self.k_cache[layer_idx, phys_ids]
            .transpose(0, 1)
            .reshape(self.cfg.num_key_value_heads, -1, self.cfg.head_dim)[:, :seq_len, :]
            .unsqueeze(0)
        )
        v = (
            self.v_cache[layer_idx, phys_ids]
            .transpose(0, 1)
            .reshape(self.cfg.num_key_value_heads, -1, self.cfg.head_dim)[:, :seq_len, :]
            .unsqueeze(0)
        )
        return k, v