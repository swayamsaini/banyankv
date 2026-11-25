from typing import Protocol


class PrefixIndexPort(Protocol):
    def match_prefix(self, token_ids: list[int], max_match: int) -> tuple[list[int], int]: ...
    def unlock_blocks(self, block_ids: list[int]) -> None: ...
    def release_sequence(
        self,
        block_table: list[int],
        token_ids: list[int],
        block_size: int,
        cache_full_blocks: bool = True,
    ) -> None: ...
    def evict_lru(self, num_needed: int) -> int: ...
