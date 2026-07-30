"""Multi-seed sweep (branin v2 + camel v4.3 + griewank 6-D v3), FIXED-DESIGN convention (user decision
2026-07-28): the SLHD locations are seed 1's design for ALL seeds; each seed only re-draws the
noise replicates at those locations (design_seed=1). Error bars therefore measure
noise-realization + refit variability CONDITIONAL on the design. Within a seed all four models
share byte-identical data. Outputs -> results_v2/. Cache-resumable; MATLAB cells first."""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MLVGP_LAUNCH_GAP_S", "3")   # 15 workers x 6 launches/cell: 8s gap would bind
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from concurrent.futures import ProcessPoolExecutor, as_completed

SEEDS = list(range(1, 31))
CONFIGS = [("branin_hetero", 25), ("branin_hetero", 40),
           ("sixhump_camel", 25), ("sixhump_camel", 40),
           ("griewank_6d", 48), ("griewank_6d", 64),
           ("griewank_6d", 80), ("griewank_6d", 120), ("griewank_6d", 160)]
MODELS_MATLAB = ["heter_LVGP", "standard_LVGP"]
MODELS_PY = ["separate_gp", "categorical_kernel"]
WORKERS = 15  # user-confirmed capacity; paired with MLVGP_LAUNCH_GAP_S=3 (5-6 launches/cell).
# 2026-07-29 incident note: a stall at 0 throughput was misdiagnosed as worker contention; the
# real cause was MathWorks-service 5001 burst throttling (verified: a single standalone launch
# failed identically). If the bridge prints repeated license-retry lines, STOP and give the
# service ~15 min of quiet rather than lowering WORKERS.


def _cell(task):
    prob, ni, m, seed = task
    from utils import study
    t = time.time()
    study.fit_predict(prob, m, seed, 10, n_init=ni, design_seed=1, verbose=False)
    return f"{m}/n{ni}/seed{seed:02d}", time.time() - t


def main():
    from utils import doe_cache
    t0 = time.time()
    for seed in SEEDS:                                   # serial DOE pre-generation
        for prob, ni in CONFIGS:
            doe_cache.ensure(prob, seed, 10, n_init=ni, design_seed=1)
    print(f"DOEs ready ({time.time()-t0:.0f}s)", flush=True)
    tasks = [(prob, ni, m, sd) for m in MODELS_MATLAB
             for sd in SEEDS for prob, ni in CONFIGS]
    tasks += [(prob, ni, m, sd) for m in MODELS_PY
              for sd in SEEDS for prob, ni in CONFIGS]
    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_cell, t): t for t in tasks}
        for f in as_completed(futs):
            done += 1
            try:
                name, dt = f.result()
                print(f"done {name}  ({dt:.0f}s)  [{done}/{len(tasks)}, "
                      f"total {time.time()-t0:.0f}s]", flush=True)
            except Exception as e:
                print(f"FAIL {futs[f]}: {e}", flush=True)
    print("ALL DONE 30-seed campaign", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
