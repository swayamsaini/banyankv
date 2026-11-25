import math
import random
from dataclasses import dataclass


@dataclass
class TraceRequest:
    req_id: int
    arrival_s: float
    prompt_ids: list[int]
    max_new_tokens: int
    group_id: int = -1


def build_poisson_trace(
    prompts_ids: list[list[int]],
    max_new_tokens_list: list[int],
    arrival_rate_rps: float,
    seed: int = 42,
) -> list[TraceRequest]:
    rng = random.Random(seed)
    t = 0.0
    trace: list[TraceRequest] = []
    for idx, (p_ids, max_tok) in enumerate(zip(prompts_ids, max_new_tokens_list, strict=True)):
        u = max(1e-12, rng.random())
        t += -math.log(u) / arrival_rate_rps
        trace.append(
            TraceRequest(req_id=idx, arrival_s=t, prompt_ids=p_ids, max_new_tokens=max_tok)
        )
    return trace


def build_best_of_n_workload(
    base_prompts_ids: list[list[int]],
    n_branches: int,
    max_new_tokens: int = 128,
) -> list[TraceRequest]:
    trace: list[TraceRequest] = []
    rid = 0
    for gid, p_ids in enumerate(base_prompts_ids):
        for _ in range(n_branches):
            trace.append(
                TraceRequest(
                    req_id=rid,
                    arrival_s=0.0,
                    prompt_ids=list(p_ids),
                    max_new_tokens=max_new_tokens,
                    group_id=gid,
                )
            )
            rid += 1
    return trace
