import torch
from torch import nn

from configs.qwen2_5_0_5b import Qwen2Config
from treekv.model.attention import Qwen2Attention
from treekv.model.layers import Qwen2MLP, RMSNorm, RotaryEmbedding


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen2Config):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Qwen2Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = Qwen2MLP(cfg.hidden_size, cfg.intermediate_size)

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
        attn_out, new_kv = self.self_attn(
            self.input_layernorm(x),
            cos,
            sin,
            kv_cache=kv_cache,
            paged_cache=paged_cache,
            layer_idx=layer_idx,
            block_table=block_table,
            positions=positions,
            seq_len=seq_len,
        )
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_kv


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen2Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.rotary_emb = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)
        self.layers = nn.ModuleList([Qwen2DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches=None,
        paged_cache=None,
        block_table=None,
        seq_len: int = 0,
    ):
        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(positions)
        new_caches = []
        for idx, layer in enumerate(self.layers):
            layer_cache = kv_caches[idx] if kv_caches is not None else None
            x, new_kv = layer(
                x,
                cos,
                sin,
                kv_cache=layer_cache,
                paged_cache=paged_cache,
                layer_idx=idx,
                block_table=block_table,
                positions=positions,
                seq_len=seq_len,
            )
            new_caches.append(new_kv)
        return self.lm_head(self.norm(x)), new_caches

    @classmethod
    def from_huggingface(cls, hf_model, cfg: Qwen2Config):
        engine = cls(cfg)
        mapped = {}
        for k, v in hf_model.state_dict().items():
            new_k = k.removeprefix("model.")
            mapped[new_k] = v
        if cfg.tie_word_embeddings and "lm_head.weight" not in mapped:
            mapped["lm_head.weight"] = mapped["embed_tokens.weight"]
        missing, unexpected = engine.load_state_dict(mapped, strict=True)
        assert not missing and not unexpected, f"Weight mismatch: {missing} / {unexpected}"
        return engine