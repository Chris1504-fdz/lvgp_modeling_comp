"""
doe_cache.py -- ONE shared initial design per (function, seed, n_rep), generated in Python and loaded
by BOTH engines (common random numbers). This removes the DOE as a confound: for a given seed every
model (separate_gp, categorical_kernel, standard_LVGP, heter_LVGP) starts from the *byte-identical*
initial dataset, so downstream differences are purely model/acquisition.

Design locations = SLHD over the full domain (see notebooks/doe.ipynb); values = n_rep heteroscedastic
noisy replicates. Deterministic in (function, seed, n_rep). The .mat is what MATLAB reads (v7); the
Python models read the same file, so the two engines cannot drift.
"""
import os
import tempfile
import numpy as np
import scipy.io

from . import problems as P

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # benchmark/
CACHE_ROOT = os.path.join(HERE, "doe_cache")


def _n_init(function, n_init):
    return int(P.get(function).n_init if n_init is None else n_init)


def path(function, seed, n_rep, mode=None, n_init=None, design_seed=None):
    """Cache file for one (function, seed, n_rep) under a given DOE mode. The mode is part of the
    FILENAME, so changing problems.DOE_MODE can never silently reuse a design built under the old one.
    A non-default n_init (modeling-study budget sweeps) is part of the filename too.
    design_seed: FIXED-DESIGN multi-seed mode -- the SLHD locations come from `design_seed`'s
    stream while the noise replicates come from `seed`'s stream (tag _fdNN). design_seed None or
    == seed -> the plain single-stream dataset."""
    mode = P.DOE_MODE if mode is None else mode
    n = _n_init(function, n_init)
    tag = "" if n == int(P.get(function).n_init) else f"_n{n:03d}"
    fd = "" if design_seed in (None, int(seed)) else f"_fd{int(design_seed):02d}"
    return os.path.join(CACHE_ROOT, function,
                        f"seed{int(seed):02d}_nrep{int(n_rep):02d}_{mode}{tag}{fd}.mat")


def _build_mat(function, seed, n_rep, mode, n_init=None, design_seed=None):
    """Deterministic full initial dataset -> dict of MATLAB-friendly arrays.
    With design_seed set (and != seed): design locations from design_seed's rng, replicates from
    seed's OWN fresh rng -- the fixed-design multi-seed convention (design held constant, noise
    re-drawn per seed)."""
    n = _n_init(function, n_init)
    if design_seed not in (None, int(seed)):
        import numpy as _np
        spec = P.get(function)
        rng_d = _np.random.default_rng(int(design_seed))
        doe = P._doe.make_doe_nd(mode, rng_d, spec.bounds, spec.n_levels, n_init=n)
        rng_n = _np.random.default_rng(int(seed))
        dd = spec.d
        X_sample = _np.zeros((n, dd + 1)); Y_sample = _np.zeros(n)
        Var_sample = _np.zeros(n); Y_rep = _np.zeros((n, n_rep))
        row = 0
        for i in range(1, spec.n_levels + 1):
            for xrow in doe[i]:
                y_rep = P.noisy_eval(spec, xrow, i, n_rep, rng_n)
                X_sample[row, :dd] = xrow; X_sample[row, dd] = i
                Y_sample[row] = y_rep.mean(); Var_sample[row] = y_rep.var(ddof=1)
                Y_rep[row] = y_rep; row += 1
        d = dict(X_sample=X_sample, Y_sample=Y_sample, Var_sample=Var_sample, Y_rep=Y_rep)
    else:
        d = P.initial_doe(P.get(function), n_rep, seed=seed, mode=mode, n_init=n)  # SLHD + replicates
    return {
        "X_sample":   np.asarray(d["X_sample"], float),              # (n_tr,2) [x1, level]
        "Y_sample":   np.asarray(d["Y_sample"], float).reshape(-1, 1),
        "Var_sample": np.asarray(d["Var_sample"], float).reshape(-1, 1),
        "Y_rep":      np.asarray(d["Y_rep"], float),                 # (n_tr, n_rep)
        "meta_function": function, "meta_seed": int(seed), "meta_n_rep": int(n_rep),
        "meta_doe_mode": mode, "meta_n_init": n,
    }


def _stale(p, mode, n_init):
    """Missing, unreadable, or written under a different DOE mode / initial-design size. Belt-and-braces
    on top of the mode-keyed filename -- also catches an n_init change (the spreadsheet budget moving)."""
    if not os.path.exists(p):
        return True
    try:
        m = scipy.io.loadmat(p)
        return (str(np.ravel(m["meta_doe_mode"])[0]).strip() != mode
                or int(np.ravel(m["meta_n_init"])[0]) != int(n_init))
    except Exception:
        return True


def ensure(function, seed, n_rep, mode=None, n_init=None, design_seed=None):
    """Write the shared initial-design .mat if missing/stale (atomic) and return its path. Safe under
    the concurrent sweep: several workers building the same key write identical content, rename wins."""
    mode = P.DOE_MODE if mode is None else mode
    p = path(function, seed, n_rep, mode, n_init, design_seed)
    if not _stale(p, mode, _n_init(function, n_init)):
        return p
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".mat", dir=os.path.dirname(p)); os.close(fd)
    scipy.io.savemat(tmp, _build_mat(function, seed, n_rep, mode, n_init, design_seed),
                     do_compression=False)
    os.replace(tmp, p)                                               # atomic
    return p


def load(function, seed, n_rep, mode=None, n_init=None, design_seed=None):
    """Shared initial dataset as arrays (ensures the .mat exists first, then reads THAT file so the
    Python models see exactly what MATLAB sees)."""
    m = scipy.io.loadmat(ensure(function, seed, n_rep, mode, n_init, design_seed))
    return dict(X_sample=np.asarray(m["X_sample"], float),
                Y_sample=np.ravel(m["Y_sample"]).astype(float),
                Var_sample=np.ravel(m["Var_sample"]).astype(float),
                Y_rep=np.asarray(m["Y_rep"], float))
