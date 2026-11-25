import json
import math
import statistics
import time
from dataclasses import dataclass

import torch

T_CRIT_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 10: 2.262}


def sync_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def percentile(values: list[float], p: float) -> float:
    assert values, "Cannot compute percentile of empty list"
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, math.ceil((p / 100.0) * len(xs)) - 1))
    return xs[idx]


def mean_and_ci95(samples: list[float]) -> tuple[float, float]:
    n = len(samples)
    m = statistics.mean(samples)
    if n < 2:
        return m, 0.0
    sd = statistics.stdev(samples)
    t_val = T_CRIT_95.get(n, 1.96)
    return m, t_val * (sd / math.sqrt(n))


@dataclass
class RequestMetric:
    req_id: int
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float
    tpot_ms: float | None
    e2e_ms: float


def save_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
