from treekv.bench.metrics import mean_and_ci95, percentile
from treekv.bench.workloads import build_best_of_n_workload, build_poisson_trace


def test_poisson_trace_determinism():
    t1 = build_poisson_trace([[1, 2], [3, 4], [5, 6]], [10, 20, 30], arrival_rate_rps=2.0, seed=7)
    t2 = build_poisson_trace([[1, 2], [3, 4], [5, 6]], [10, 20, 30], arrival_rate_rps=2.0, seed=7)
    assert [r.arrival_s for r in t1] == [r.arrival_s for r in t2]
    assert t1[0].arrival_s < t1[1].arrival_s < t1[2].arrival_s


def test_ci95_and_percentile_exact():
    mean, half = mean_and_ci95([41.0, 43.0, 39.0, 44.0, 43.0])
    assert abs(mean - 42.0) < 1e-6
    assert abs(half - 2.4829) < 0.01
    assert percentile([10.0, 20.0, 30.0, 40.0, 50.0], 50) == 30.0
    assert len(build_best_of_n_workload([[1, 2, 3]], n_branches=8)) == 8
