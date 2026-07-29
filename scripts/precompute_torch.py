"""Precompute the AUXILIARY pytorch LVGP (lvgp-bayes) predictions for the CatGP comparison.
NOT models of record (that stays the MATLAB engines) -- experimental sweep, seed 1, n_rep=10."""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from utils import study

CONFIGS = [("branin_hetero", 25), ("branin_hetero", 40),
           ("sixhump_camel", 20), ("sixhump_camel", 32),
           ("griewank_6d", 20), ("griewank_6d", 32)]
t0 = time.time()
for prob, ni in CONFIGS:
    for m in ["lvgp_torch", "heter_lvgp_torch"]:
        t = time.time()
        torch.manual_seed(1)
        study.fit_predict(prob, m, 1, 10, n_init=ni, verbose=False)
        print(f"done {prob}(n={ni or 'default'})/{m}  ({time.time()-t:.0f}s, "
              f"total {time.time()-t0:.0f}s)", flush=True)
print("ALL DONE torch", time.time() - t0)
