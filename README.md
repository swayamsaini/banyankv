# TreeKV-Engine

A from-scratch, reasoning-first LLM inference engine in pure PyTorch targeting **Qwen2.5-0.5B / 1.5B** on free compute (Kaggle T4 x2 + local CPU unit tests).

## Implemented Subsystems
- **Phase 0 (`treekv/model/`)**: From-scratch Qwen2.5 decoder (RMSNorm, RoPE, 7:1 GQA, SwiGLU, tied embeddings) verified token-for-token against Hugging Face.
- **Phase 1 (`treekv/memory/block_manager.py`)**: Paged KV Cache (16-token blocks, 192 KB/block in FP16), reference counting, and Copy-on-Write isolation.
- **Phase 2 (`treekv/engine/scheduler.py`)**: Iteration-level Continuous Batching with LIFO recompute preemption.
- **Phase 3 (`treekv/memory/radix_cache.py`)**: Radix Tree-KV prefix sharing for parallel Best-of-N reasoning with safe LRU leaf eviction.
- **Phase 4 (`treekv/model/sampler.py`, `treekv/engine/speculative.py`)**: Lossless Speculative Decoding (`min(1, p/q)` + residual correction), cross-boundary KV rollback, Prompt-Lookup drafting, and batch-aware adaptive `gamma` gating.
- **Phase 5 (`treekv/bench/`)**: Reproducible CUDA-synced benchmarking (`TTFT`, `TPOT`, `p99`, 95% CIs, Poisson traces) and one-command Kaggle T4 study runner.

## Quickstart (Local CPU Unit Tests)
```bash
pytest -v tests/ -k "not huggingface_qwen2_5"
```
