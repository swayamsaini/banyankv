from dataclasses import dataclass, field
from enum import Enum


class SeqStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Sequence:
    seq_id: int
    prompt_ids: list[int]
    max_new_tokens: int
    eos_token_id: int
    status: SeqStatus = SeqStatus.WAITING
    output_ids: list[int] = field(default_factory=list)
    block_table: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    arrival_time: float = 0.0
    first_token_time: float | None = None
    finish_time: float | None = None

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_ids + self.output_ids

    @property
    def total_len(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    @property
    def is_prefill(self) -> bool:
        return self.num_computed_tokens < len(self.prompt_ids)

    def append_token(self, token_id: int, now: float | None = None) -> None:
        if self.first_token_time is None and now is not None:
            self.first_token_time = now
        self.output_ids.append(token_id)
        if token_id == self.eos_token_id or len(self.output_ids) >= self.max_new_tokens:
            self.status = SeqStatus.FINISHED
            if now is not None:
                self.finish_time = now
