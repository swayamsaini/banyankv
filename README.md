# TreeKV-Engine

A from-scratch, pure-PyTorch **Mini LLM Inference Engine** targeting [`Qwen/Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B), designed for parallel reasoning workloads (Best-of-N). Built with clean modular interfaces, strict correctness gates verified token-for-token against Hugging Face, and reproducible CUDA-synced benchmarking on NVIDIA Tesla T4.

## Highlights

- **Exact Hugging Face Parity (`12/12` Tests Passing on Kaggle Tesla T4):** From-scratch `Qwen2.5-0.5B` decoder (`RMSNorm`, `RoPE` with `theta=1e6`, 7:1 `GQA`, `SwiGLU`, tied embeddings, and Qwen2 `q/k/v` projection biases) verified token-for-token against `transformers`.
- **Paged KV Cache + Copy-on-Write (`Phase 1`):** 16-token physical blocks (`192 KB/block` in FP16 across 24 layers), `BlockAllocator`, reference counting, and Copy-on-Write isolation inspired by [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180).
- **Iteration-Level Continuous Batching (`Phase 2`):** `Sequence` state machine, Prefill-First `ContinuousScheduler`, and LIFO recompute preemption inspired by [Orca](https://www.usenix.org/conference/osdi22/presentation/yu).
- **Radix Tree-KV Prefix Cache (`Phase 3`):** Chained block-hash prefix tree (`RadixBlockCache` + `NullPrefixCache` ablation adapter) with safe LRU leaf eviction (`lock_count == 0`), inspired by [SGLang RadixAttention](https://arxiv.org/abs/2312.07104).
- **Lossless Speculative Decoding (`Phase 4`):** Exact rejection sampling (`min(1, p/q)` + residual `norm(max(0, p - q))`), cross-boundary KV rollback with zero block leaks, Prompt-Lookup n-gram drafter, and batch-aware `adaptive_gamma` gating inspired by [Leviathan et al.](https://arxiv.org/abs/2211.17192).
- **CUDA-Synced Benchmarking (`Phase 5`):** `torch.cuda.synchronize()` timing, deterministic Poisson arrival traces, `p50`/`p99` latency, and 95% Student-t confidence intervals.

---

## Verified Tesla T4 Benchmark Results (`v1.0.0`)

Measured on **Kaggle NVIDIA Tesla T4** (`15,360 MiB VRAM`, `PyTorch 2.11.0+cu130`, `transformers 5.20.2`, `Qwen/Qwen2.5-0.5B` in `FP16`, Best-of-8 parallel reasoning branches, `max_new_tokens=32`, 5 CUDA-synced repeats):

![TreeKV-Engine Tesla T4 Best-of-8 Benchmark](docs/t4_benchmark_best_of_8.png)

| Metric (Best-of-8, 5 repeats) | No-Radix (Paged KV) | With Radix (Tree-KV) | Improvement |
|---|---:|---:|---:|
| **TTFT p50 (ms)** ↓ | `222.80 ms` | **`177.02 ms`** | **`-20.55%`** |
| **Throughput (tok/s, Mean ± 95% CI)** ↑ | `24.75 ± 1.18` | **`26.52 ± 0.73`** | **`+7.15%`** |
| **End-to-End p99 (ms)** ↓ | `11,019.32 ms` | **`10,020.75 ms`** | **`-9.06%`** |
| **Correctness Gate (`pytest -v tests/`)** | — | **`12 passed in 40.60s`** | Exact HF match |

> **Engineering Note:** Even on a short ~56-token math prompt where autoregressive decode dominates total wall time, Radix Tree-KV cuts median **TTFT by 20.6%**, trims **p99 E2E latency by 9.1%**, and raises throughput by **+7.2%** with tighter variance (`±0.73` vs `±1.18 tok/s`). Longer shared system prompts (512–2,048 tokens) further amplify prefill savings. Raw output is archived in [`results/study_summary.json`](results/study_summary.json).

---

## Architecture

```text
configs/
  qwen2_5_0_5b.py          # Exact Qwen2.5-0.5B hyperparameters & KV math
treekv/
  interfaces.py            # Clean Protocol contracts (PrefixCache, BlockAllocator, Drafter)
  model/
    layers.py              # RMSNorm, RoPE, SwiGLU
    attention.py           # 7:1 GQA with contiguous & paged KV paths
    qwen2.py               # Scratch Qwen2ForCausalLM + Hugging Face weight loader
    sampler.py             # Greedy & temperature/top-p sampling
  memory/
    block_manager.py       # 16-token PagedKVCache, BlockAllocator, Copy-on-Write
    radix_cache.py         # RadixBlockCache (LRU eviction) & NullPrefixCache
  engine/
    sequence.py            # Sequence FSM & timing metadata
    scheduler.py           # Iteration-level ContinuousScheduler + LIFO preemption
    speculative.py         # Lossless rejection sampling, KV rollback, Prompt-Lookup drafter
    runner.py              # End-to-end TreeKVEngine orchestrator
  bench/
    workloads.py           # Deterministic Poisson arrival traces
    metrics.py             # CUDA-synced timing, p50/p99, 95% Student-t CI
    run_study.py           # Reproducible Best-of-N ablation study runner
tests/                     # 12 unit & integration gate tests (Phases 0–5)
notebooks/
  kaggle_t4_runner.ipynb   # One-click free Kaggle T4 x2 verification & benchmark notebook
```

---

## Quickstart & Reproducibility

### 1. Fast Offline Verification (CPU, ~11 seconds)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
ruff check configs/ treekv/ tests/
pytest -v tests/ -k "not huggingface_qwen2_5"
```

### 2. Full Hugging Face Parity + GPU Benchmark (Free Kaggle `GPU T4 x2`)
Upload `notebooks/kaggle_t4_runner.ipynb` to Kaggle with **Accelerator: `GPU T4 x2`** and **Internet: `ON`**, or run:
```bash
pytest -v tests/
python -m treekv.bench.run_study --model Qwen/Qwen2.5-0.5B --n-branches 8 --max-new-tokens 32 --repeats 5 --out results/
```

## License

MIT
