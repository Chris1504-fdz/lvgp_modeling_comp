"""Precompute sixhump_camel at both budgets (n_init=8 canonical, n_init=32 = 8/level)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import study

t0 = time.time()
for n_init in [None, 32]:
    for seed in [1, 2, 3]:
        for m in ["separate_gp", "categorical_kernel", "standard_LVGP", "heter_LVGP"]:
            t = time.time()
            study.fit_predict("sixhump_camel", m, seed, 10, n_init=n_init)
            print(f"done sixhump_camel(n={n_init or 8})/{m}/seed{seed}  ({time.time()-t:.0f}s, "
                  f"total {time.time()-t0:.0f}s)", flush=True)
print("ALL DONE camel", time.time() - t0)
