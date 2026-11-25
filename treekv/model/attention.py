import math

import torch
import torch.nn.functional as F
from torch import nn

from configs.qwen2_5_0_5b import Qwen2Config
from treekv.model.layers import apply_rotary_pos_emb


class Qwen2Attention(nn.Module):
    def __init__(self, cfg: Qwen2Config):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.num_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache=None,
        paged_cache=None,
        layer_idx: int = 0,
        block_table=None,
        positions=None,
        seq_len: int = 0,
    ):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if paged_cache is not None:
            assert B == 1, "Single-sequence paged path"
            paged_cache.write_kv(layer_idx, block_table, positions[0], k, v)
            k, v = paged_cache.gather_kv(layer_idx, block_table, seq_len)
            new_kv_cache = None
        else:
            if kv_cache is not None:
                k = torch.cat([kv_cache[0], k], dim=2)
                v = torch.cat([kv_cache[1], v], dim=2)
            new_kv_cache = (k, v)

        k_exp = k.repeat_interleave(self.num_groups, dim=1)
        v_exp = v.repeat_interleave(self.num_groups, dim=1)

        scores = torch.matmul(q, k_exp.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if S > 1:
            total_len = k_exp.shape[2]
            causal = torch.triu(
                torch.full((S, total_len), float("-inf"), device=x.device, dtype=scores.dtype),
                diagonal=total_len - S + 1,
            )
            scores = scores + causal

        weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(weights, v_exp).transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out), new_kv_cache