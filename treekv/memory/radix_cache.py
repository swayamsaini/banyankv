import heapq
import time

from treekv.memory.block_manager import BlockAllocator


class NullPrefixCache:
    """No-op adapter for clean Phase 5 ablations (Radix OFF vs. ON)."""

    def __init__(self, allocator: BlockAllocator):
        self.allocator = allocator

    def match_prefix(self, token_ids: list[int], max_match: int) -> tuple[list[int], int]:
        return [], 0

    def unlock_blocks(self, block_ids: list[int]) -> None:
        self.allocator.free_table(block_ids)

    def release_sequence(
        self,
        block_table: list[int],
        token_ids: list[int],
        block_size: int,
        cache_full_blocks: bool = True,
    ) -> None:
        self.allocator.free_table(block_table)

    def evict_lru(self, num_needed: int) -> int:
        return 0


class RadixNode:
    def __init__(
        self,
        key: tuple[int, ...] | None = None,
        block_id: int = -1,
        parent: "RadixNode | None" = None,
    ):
        self.key = key
        self.block_id = block_id
        self.parent = parent
        self.children: dict[tuple[int, ...], RadixNode] = {}
        self.lock_count: int = 0
        self.last_access: float = time.monotonic()


class RadixBlockCache:
    def __init__(self, allocator: BlockAllocator, block_size: int = 16):
        self.allocator = allocator
        self.block_size = block_size
        self.root = RadixNode()
        self.block_to_node: dict[int, RadixNode] = {}
        self.total_matched_tokens: int = 0

    def match_prefix(self, token_ids: list[int], max_match: int) -> tuple[list[int], int]:
        matched_blocks: list[int] = []
        node = self.root
        pos = 0
        now = time.monotonic()

        while pos + self.block_size <= max_match:
            chunk = tuple(token_ids[pos : pos + self.block_size])
            child = node.children.get(chunk)
            if child is None:
                break
            child.lock_count += 1
            child.last_access = now
            self.allocator.share(child.block_id)
            matched_blocks.append(child.block_id)
            node = child
            pos += self.block_size

        self.total_matched_tokens += pos
        return matched_blocks, pos

    def unlock_blocks(self, block_ids: list[int]) -> None:
        for bid in block_ids:
            node = self.block_to_node.get(bid)
            if node is not None and node.lock_count > 0:
                node.lock_count -= 1
            self.allocator.free(bid)

    def insert_full_blocks(self, token_ids: list[int], block_table: list[int]) -> None:
        num_full = min(len(token_ids) // self.block_size, len(block_table))
        node = self.root
        now = time.monotonic()

        for i in range(num_full):
            chunk = tuple(token_ids[i * self.block_size : (i + 1) * self.block_size])
            bid = block_table[i]
            if chunk in node.children:
                child = node.children[chunk]
                child.last_access = now
            else:
                self.allocator.share(bid)
                child = RadixNode(key=chunk, block_id=bid, parent=node)
                child.last_access = now
                node.children[chunk] = child
                self.block_to_node[bid] = child
            node = child

    def release_sequence(
        self,
        block_table: list[int],
        token_ids: list[int],
        block_size: int,
        cache_full_blocks: bool = True,
    ) -> None:
        if cache_full_blocks:
            self.insert_full_blocks(token_ids, block_table)
        for bid in block_table:
            node = self.block_to_node.get(bid)
            if node is not None and node.lock_count > 0:
                node.lock_count -= 1
            self.allocator.free(bid)

    def evict_lru(self, num_needed: int) -> int:
        evicted = 0
        heap: list[tuple[float, int, RadixNode]] = []
        stack = [self.root]
        while stack:
            cur = stack.pop()
            for child in cur.children.values():
                if not child.children and child.lock_count == 0:
                    heapq.heappush(heap, (child.last_access, child.block_id, child))
                else:
                    stack.append(child)

        while evicted < num_needed and heap:
            _, bid, leaf = heapq.heappop(heap)
            if leaf.lock_count > 0 or leaf.children:
                continue
            parent = leaf.parent
            assert parent is not None and leaf.key is not None
            del parent.children[leaf.key]
            del self.block_to_node[bid]
            self.allocator.free(bid)
            evicted += 1
            if parent is not self.root and not parent.children and parent.lock_count == 0:
                heapq.heappush(heap, (parent.last_access, parent.block_id, parent))

        return evicted
