"""Precompute the branin_hetero n_init=40 (8 pts/level) budget variant. Cache-resumable."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import study

t0 = time.time()
for seed in [1, 2, 3]:
    for m in ["separate_gp", "categorical_kernel", "standard_LVGP", "heter_LVGP"]:
        t = time.time()
        study.fit_predict("branin_hetero", m, seed, 10, n_init=40)
        print(f"done branin_hetero_n40/{m}/seed{seed}  ({time.time()-t:.0f}s, "
              f"total {time.time()-t0:.0f}s)", flush=True)
print("ALL DONE n40", time.time() - t0)
