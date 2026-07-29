#!/usr/bin/env python
"""Parallel precompute of all cached predictions for the study notebook.

Grid: 5 (problem, budget) configs x 4 models x n_rep {3, 10}, seed 1 (seeds are a later axis).
Worker-pool pattern from hetero_lvgp/benchmark/run.py: one BLAS thread per worker (set BEFORE
numpy import), shared initial designs pre-generated SERIALLY (avoids a first-wave stampede on the
DOE cache), and MATLAB launches spaced/retried inside utils/models/matlab_lvgp.py (flock'd
launch-gap throttle + license-5001 backoff -- see that file), so a burst of workers cannot wedge
the license server that the parent repo's sweep is already leaning on.

Cache-resumable: finished (problem, model, seed, n_rep, n_init) cells are skipped instantly.
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from concurrent.futures import ProcessPoolExecutor, as_completed

# REDUCED SCOPE (2026-07-28): branin 25/40 + camel 20/32, n_rep=10 only.
# Dropped: n_rep=3 (too few), branin n=10 / camel n=8 (data-starved), rastrigin_6d (excluded).
CONFIGS = [("branin_hetero", 25), ("branin_hetero", 40),
           ("sixhump_camel", 25), ("sixhump_camel", 40),
           ("griewank_6d", 48), ("griewank_6d", 64)]
MODELS = ["separate_gp", "categorical_kernel", "standard_LVGP", "heter_LVGP"]
N_REPS = [10]
SEED = 1
WORKERS = 6


def _cell(task):
    prob, n_init, model, n_rep = task
    from utils import study
    t = time.time()
    study.fit_predict(prob, model, SEED, n_rep, n_init=n_init, verbose=False)
    return f"{prob}(n={n_init or 'default'})/nrep{n_rep}/{model}", time.time() - t


def main():
    from utils import study, doe_cache, problems as P
    t0 = time.time()
    for prob, n_init in CONFIGS:                 # serial DOE pre-generation (parent-repo pattern)
        for n_rep in N_REPS:
            doe_cache.ensure(prob, SEED, n_rep, n_init=n_init)
    print(f"DOEs ready ({time.time()-t0:.0f}s)", flush=True)

    # MATLAB cells first (they dominate wall time; python cells backfill the idle workers)
    tasks = [(p, ni, m, nr) for m in ["heter_LVGP", "standard_LVGP"]
             for p, ni in CONFIGS for nr in N_REPS]
    tasks += [(p, ni, m, nr) for m in ["separate_gp", "categorical_kernel"]
              for p, ni in CONFIGS for nr in N_REPS]
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
    print("ALL DONE", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
