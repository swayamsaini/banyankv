import argparse
import json
import os
import platform

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.qwen2_5_0_5b import Qwen2Config
from treekv.bench.metrics import mean_and_ci95, percentile, sync_time
from treekv.engine.runner import TreeKVEngine
from treekv.engine.sequence import Sequence
from treekv.model.qwen2 import Qwen2ForCausalLM


def collect_env_metadata() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }


def benchmark_best_of_n(
    model: Qwen2ForCausalLM,
    prompt_ids: list[int],
    eos_id: int,
    n_branches: int,
    max_new_tokens: int,
    use_radix: bool,
    device: str,
    dtype: torch.dtype,
    repeats: int = 3,
) -> dict:
    throughputs, ttfts, e2es = [], [], []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        engine = TreeKVEngine(
            model, num_blocks=2048, block_size=16, max_batch_size=n_branches,
            use_radix=use_radix, device=device, dtype=dtype,
        )
        t0 = sync_time()
        for i in range(n_branches):
            engine.submit(
                Sequence(
                    seq_id=i,
                    prompt_ids=list(prompt_ids),
                    max_new_tokens=max_new_tokens,
                    eos_token_id=eos_id,
                )
            )
        finished = engine.run_until_done(temperature=0.0)
        wall_s = sync_time() - t0
        total_out = sum(len(s.output_ids) for s in finished)
        throughputs.append(total_out / max(1e-9, wall_s))
        ttfts.extend([(s.first_token_time - s.arrival_time) * 1000.0 for s in finished if s.first_token_time])
        e2es.extend([(s.finish_time - s.arrival_time) * 1000.0 for s in finished if s.finish_time])

    return {
        "use_radix": use_radix,
        "n_branches": n_branches,
        "throughput_tok_s_mean_ci95": mean_and_ci95(throughputs),
        "ttft_p50_ms": percentile(ttfts, 50) if ttfts else 0.0,
        "e2e_p99_ms": percentile(e2es, 99) if e2es else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-branches", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(args.model)
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).eval()
    model = Qwen2ForCausalLM.from_huggingface(hf, Qwen2Config()).to(device=device, dtype=dtype).eval()
    del hf

    prompt = (
        "You are an expert competition mathematician. Solve the following problem carefully "
        "step by step and verify every algebraic step before giving the final answer: "
        "A store sells notebooks for $4 each and pens for $2 each. If Maya buys 7 notebooks "
        "and 9 pens and pays with a $50 bill, how much change does she receive?"
    )
    prompt_ids = tok(prompt)["input_ids"]

    res_no_radix = benchmark_best_of_n(
        model, prompt_ids, tok.eos_token_id, args.n_branches,
        args.max_new_tokens, False, device, dtype, args.repeats,
    )
    res_radix = benchmark_best_of_n(
        model, prompt_ids, tok.eos_token_id, args.n_branches,
        args.max_new_tokens, True, device, dtype, args.repeats,
    )

    report = {
        "env": collect_env_metadata(),
        "no_radix": res_no_radix,
        "with_radix": res_radix,
    }
    out_path = os.path.join(args.out, "study_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
