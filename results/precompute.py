"""Precompute all cached predictions for the study notebook (safe to re-run; cache-resumable).
Run with the ml_gp_env python from the repo root."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import study

PROBLEMS = ["branin_hetero", "rastrigin_6d"]
MODELS = ["separate_gp", "categorical_kernel", "standard_LVGP", "heter_LVGP"]
SEEDS = [1, 2, 3]
N_REP = 10

t0 = time.time()
for prob in PROBLEMS:
    for seed in SEEDS:
        for m in MODELS:
            t = time.time()
            study.fit_predict(prob, m, seed, N_REP)
            print(f"done {prob}/{m}/seed{seed}  ({time.time()-t:.0f}s, total {time.time()-t0:.0f}s)",
                  flush=True)
print("ALL DONE", time.time() - t0)
