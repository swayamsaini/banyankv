import torch

from treekv.bench.metrics import sync_time
from treekv.engine.scheduler import ContinuousScheduler
from treekv.engine.sequence import Sequence
from treekv.memory.block_manager import PagedKVCache
from treekv.memory.radix_cache import NullPrefixCache, RadixBlockCache
from treekv.model.qwen2 import Qwen2ForCausalLM
from treekv.model.sampler import sample_token


class TreeKVEngine:
    def __init__(
        self,
        model: Qwen2ForCausalLM,
        num_blocks: int = 256,
        block_size: int = 16,
        max_batch_size: int = 16,
        use_radix: bool = True,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.model = model.to(device=device, dtype=dtype).eval()
        self.device = device
        self.paged_cache = PagedKVCache(
            model.cfg, num_blocks=num_blocks, block_size=block_size, device=device, dtype=dtype
        )
        self.radix = (
            RadixBlockCache(self.paged_cache.allocator, block_size=block_size)
            if use_radix
            else NullPrefixCache(self.paged_cache.allocator)
        )
        self.scheduler = ContinuousScheduler(
            self.paged_cache, max_batch_size=max_batch_size, radix_cache=self.radix
        )

    def submit(self, seq: Sequence) -> None:
        seq.arrival_time = sync_time()
        self.scheduler.add_request(seq)

    def step(self, temperature: float = 0.0) -> None:
        prefill_batch, decode_batch = self.scheduler.schedule()

        with torch.no_grad():
            for seq in prefill_batch:
                uncomputed = seq.all_token_ids[seq.num_computed_tokens :]
                ids = torch.tensor([uncomputed], device=self.device, dtype=torch.long)
                pos = torch.arange(
                    seq.num_computed_tokens, seq.total_len, device=self.device, dtype=torch.long
                ).unsqueeze(0)
                logits, _ = self.model(
                    ids,
                    pos,
                    paged_cache=self.paged_cache,
                    block_table=seq.block_table,
                    seq_len=seq.total_len,
                )
                seq.num_computed_tokens = seq.total_len
                next_tok = sample_token(logits[0, -1, :], temperature=temperature)
                seq.append_token(next_tok, now=sync_time())

            for seq in decode_batch:
                last_tok = seq.all_token_ids[-1]
                ids = torch.tensor([[last_tok]], device=self.device, dtype=torch.long)
                cur_pos = seq.total_len - 1
                pos = torch.tensor([[cur_pos]], device=self.device, dtype=torch.long)
                logits, _ = self.model(
                    ids,
                    pos,
                    paged_cache=self.paged_cache,
                    block_table=seq.block_table,
                    seq_len=seq.total_len,
                )
                seq.num_computed_tokens = seq.total_len
                next_tok = sample_token(logits[0, -1, :], temperature=temperature)
                seq.append_token(next_tok, now=sync_time())

    def run_until_done(self, temperature: float = 0.0) -> list[Sequence]:
        while self.scheduler.has_pending():
            self.step(temperature=temperature)
        return self.scheduler.finished
