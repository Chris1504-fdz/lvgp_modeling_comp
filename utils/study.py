"""
study.py -- driver + prediction cache for the two-problem modeling study.

For a given (problem, seed, n_rep, model) this fits the model on the shared cached DOE and
evaluates (mu, epistemic s2, aleatoric r) on the study's FIXED test set, caching the arrays to
results/pred_cache/<problem>/<model>_seed<s>_nrep<r>.npz. The MATLAB LVGP engines cost ~30-60 s
per fit + seconds per prediction batch, so caching makes the notebook cheap to re-run; the
python GPs are cached too for uniformity. Delete a cache file to force a refit.

Test sets (deterministic, shared by every model):
  d == 1 : dense grid of N_GRID_1D points per level over [lb, ub].
  d >  1 : the SAME scrambled-Sobol sample of N_SOBOL points (seed TEST_SEED) for every level.
"""
import os
import numpy as np
from scipy.stats import qmc

from . import problems as P, doe_cache
from .models import MODELS

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # repo root
CACHE_ROOT = os.path.join(HERE, "results", "pred_cache")

N_GRID_1D = 200
N_SOBOL = 1024
TEST_SEED = 7


def test_points(spec):
    """{level: (n, d) test inputs} -- dense 1-D grid, or one shared Sobol cloud for d > 1."""
    if spec.d == 1:
        X = np.linspace(spec.lb, spec.ub, N_GRID_1D).reshape(-1, 1)
    else:
        u = qmc.Sobol(spec.d, scramble=True, seed=TEST_SEED).random(N_SOBOL)
        b = spec.bounds
        X = b[:, 0] + u * (b[:, 1] - b[:, 0])
    return {lv: X for lv in spec.levels}


def train_data(problem, seed, n_rep, n_init=None, design_seed=None):
    """Shared cached SLHD design -> the per-level dict every model's fit() consumes."""
    spec = P.get(problem)
    ds = doe_cache.load(problem, seed, n_rep, n_init=n_init, design_seed=design_seed)
    X, Y, V = ds["X_sample"], ds["Y_sample"], ds["Var_sample"]
    lv_col = X[:, spec.d].astype(int)
    return {int(lv): dict(X=X[lv_col == lv, :spec.d], y_mean=Y[lv_col == lv],
                          y_var=V[lv_col == lv]) for lv in spec.levels}, ds


# Multi-seed (fixed-design) campaigns live under results_v2/results_files, organized
# <problem>/<model>/, so they never mix with the single-seed exploration caches in results/
# (user request 2026-07-28). ANY design_seed (including == seed) routes there, so a campaign
# is self-contained for sharing; seed-1 cells are duplicated from the plain cache.
CACHE_ROOT_V2 = os.path.join(HERE, "results_v2", "results_files")


def _cache_path(problem, model, seed, n_rep, n_init=None, design_seed=None):
    tag = "" if n_init is None or int(n_init) == P.get(problem).n_init else f"_n{int(n_init):03d}"
    if design_seed is None:
        return os.path.join(CACHE_ROOT, problem,
                            f"{model}_seed{seed:02d}_nrep{n_rep:02d}{tag}.npz")
    fd = f"_fd{int(design_seed):02d}"
    n_val = int(n_init) if n_init is not None else P.get(problem).n_init
    return os.path.join(CACHE_ROOT_V2, problem, model, f"n{n_val:03d}",
                        f"{model}_seed{seed:02d}_nrep{n_rep:02d}{tag}{fd}.npz")


def fit_predict(problem, model, seed, n_rep, n_init=None, design_seed=None, verbose=True):
    """Cached (mu, s2, r) on the fixed test set. Returns {level: dict(mu, s2, r)}."""
    p = _cache_path(problem, model, seed, n_rep, n_init, design_seed)
    spec = P.get(problem)
    if not os.path.exists(p):
        if verbose:
            print(f"[fit] {problem} / {model} / seed {seed} (nrep {n_rep}, n_init {n_init}) ...",
                  flush=True)
        data, _ = train_data(problem, seed, n_rep, n_init, design_seed)
        mdl = MODELS[model].cls.fit(data, needs_r=True, bounds=spec.bounds)
        out = {}
        for lv, Xt in test_points(spec).items():
            mu, s2 = mdl.predict(lv, Xt)
            try:
                r = np.asarray(mdl.r(lv, Xt), float).ravel()
            except Exception:
                r = np.full(len(np.atleast_2d(Xt)), np.nan)
            out[f"mu_{lv}"] = np.ravel(mu); out[f"s2_{lv}"] = np.ravel(s2); out[f"r_{lv}"] = r
        os.makedirs(os.path.dirname(p), exist_ok=True)
        np.savez(p, **out)
    z = np.load(p)
    return {lv: dict(mu=z[f"mu_{lv}"], s2=z[f"s2_{lv}"], r=z[f"r_{lv}"]) for lv in spec.levels}


class CachedPredictions:
    """Adapter giving cached arrays the model interface metrics_table() expects. Test inputs must
    be exactly the study's fixed test set (asserted by shape)."""

    def __init__(self, problem, model, seed, n_rep, n_init=None, design_seed=None):
        self.spec = P.get(problem)
        self.pred = fit_predict(problem, model, seed, n_rep, n_init, design_seed)

    def mean_std(self, level, X):
        d = self.pred[int(level)]
        assert len(np.atleast_2d(X)) == len(d["mu"]), "test set mismatch with cache"
        return d["mu"], np.sqrt(np.clip(d["s2"], 1e-24, None))

    def r(self, level, X):
        return self.pred[int(level)]["r"]
