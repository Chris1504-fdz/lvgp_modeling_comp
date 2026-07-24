"""
matlab_lvgp.py -- the MATLAB LVGP engines (the models of record for this study), exposed through
the same fit/predict/r interface as the python models.

fit()  -> one `matlab -batch mfit(...)` call: LVGP_fit (standard) or LVGP_fit_noise + the
          aleatoric polynomial (heter); the fitted model struct is saved to a temp .mat.
predict()/mean_std()/r() -> one `matlab -batch mpredict(...)` call per (level, X_new) batch,
          cached, returning the posterior mean, epistemic variance and aleatoric variance r
          (r = NaN for the standard model).

Costs: ~10-40 s per MATLAB launch+fit and ~5-10 s per prediction batch -- fine for a modeling
study (few fits, batched grids); do NOT use these classes inside a tight loop.
"""
import os
import subprocess
import tempfile
import shutil
import atexit
import numpy as np
import scipy.io

from .base import BaseModel, _as_2d

MATLAB = "/data/zhq7531/MATLAB/bin/matlab"
XVFB_LIBDIR = "/data/zhq7531/envs/xvfblib/lib"          # libtirpc for headless R2026a (see parent repo)
MATLAB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          "matlab")
FIT_TIMEOUT_S = 3600
PRED_TIMEOUT_S = 900


def _run_matlab(call, timeout):
    env = dict(os.environ)
    env["DISPLAY"] = os.environ.get("BENCH_DISPLAY", "")
    env["LD_LIBRARY_PATH"] = XVFB_LIBDIR + ":" + env.get("LD_LIBRARY_PATH", "")
    pref = tempfile.mkdtemp(prefix="mlpref_"); tmp = tempfile.mkdtemp(prefix="mltmp_")
    env["MATLAB_PREFDIR"] = pref; env["TMPDIR"] = tmp
    try:
        r = subprocess.run([MATLAB, "-nodisplay", "-singleCompThread", "-batch", call],
                           cwd=MATLAB_DIR, env=env, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(f"MATLAB failed (rc={r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-500:]}")
    finally:
        shutil.rmtree(pref, ignore_errors=True); shutil.rmtree(tmp, ignore_errors=True)


class _MatlabLVGPBase(BaseModel):
    ENGINE = "matlab"
    MODEL_TYPE = "heter"                                  # subclass overrides

    def __init__(self, model_mat, workdir, levels):
        self._model_mat = model_mat
        self._workdir = workdir                           # cleaned at exit
        self.levels = list(levels)
        self._cache = {}                                  # (level, X bytes) -> (mu, s2, r)
        atexit.register(shutil.rmtree, workdir, ignore_errors=True)

    @classmethod
    def fit(cls, data_by_level, needs_r=True, bounds=None, **_kw):
        levels = sorted(int(lv) for lv in data_by_level)
        Xs, ys, vs = [], [], []
        for lv in levels:
            dd = data_by_level[lv]
            X = _as_2d(dd["X"])
            Xs.append(np.hstack([X, np.full((X.shape[0], 1), float(lv))]))   # level = LAST column
            ys.append(np.asarray(dd["y_mean"], float))
            vs.append(np.asarray(dd["y_var"], float))
        X = np.vstack(Xs); y = np.concatenate(ys); v = np.concatenate(vs)
        wd = tempfile.mkdtemp(prefix="mlvgp_")
        train = os.path.join(wd, "train.mat"); model_mat = os.path.join(wd, "model.mat")
        scipy.io.savemat(train, dict(X=X, Y=y.reshape(-1, 1), noise_var=v.reshape(-1, 1)))
        _run_matlab(f"mfit('{train}','{model_mat}','{cls.MODEL_TYPE}')", FIT_TIMEOUT_S)
        if not os.path.exists(model_mat):
            raise RuntimeError("mfit produced no model file")
        return cls(model_mat, wd, levels)

    def _eval(self, level, x_new):
        X = _as_2d(x_new)
        key = (int(level), X.tobytes())
        if key not in self._cache:
            Xp = np.hstack([X, np.full((X.shape[0], 1), float(int(level)))])
            pred = os.path.join(self._workdir, "pred.mat"); out = os.path.join(self._workdir, "out.mat")
            scipy.io.savemat(pred, dict(Xpred=Xp))
            _run_matlab(f"mpredict('{self._model_mat}','{pred}','{out}')", PRED_TIMEOUT_S)
            d = scipy.io.loadmat(out)
            self._cache[key] = (np.ravel(d["mu"]).astype(float),
                                np.ravel(d["s2"]).astype(float),
                                np.ravel(d["r"]).astype(float))
        return self._cache[key]

    def predict(self, level, x_new, observation_noise=False):
        mu, s2, _ = self._eval(level, x_new)
        return mu, s2

    def mean_std(self, level, x_new):
        mu, s2, _ = self._eval(level, x_new)
        return mu, np.maximum(np.sqrt(np.clip(s2, 1e-24, None)), 1e-12)

    def r(self, level, x_new):
        _, _, r = self._eval(level, x_new)
        return r


class MatlabHeterLVGP(_MatlabLVGPBase):
    """Heteroscedastic LVGP: MATLAB LVGP_fit_noise + aleatoric polynomial (the lab's model)."""
    SUPPORTS = ("ei", "lcb", "pi", "haei", "anpei", "rahbo")
    MODEL_TYPE = "heter"


class MatlabStandardLVGP(_MatlabLVGPBase):
    """Homoscedastic LVGP: MATLAB LVGP_fit (noise-blind baseline). r() returns NaN."""
    SUPPORTS = ("ei", "lcb", "pi")
    MODEL_TYPE = "standard"
