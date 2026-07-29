"""
metrics.py -- ground-truth modeling metrics for the surrogate comparison, computed PER LEVEL
(the rows of the results tables) plus a pooled/mean summary row. All metrics are evaluated on a
dense test set with the ANALYTIC f_true (never on training data), so they measure true modeling
quality with no estimation noise in the reference.

Per level l, with test points {x_i}, truth f_i = f_true(x_i, l), prediction mu_i and TOTAL
predictive std s_i (epistemic; callers may add aleatoric variance for observation-level scoring):

  MAE      = mean |mu_i - f_i|                                    (surface accuracy; lower better)
  RRMSE    = sqrt(mean (mu_i - f_i)^2) / std(f_i)                 (relative accuracy; lower better)
  Coverage = frac( |mu_i - f_i| <= z * s_i ),  z = 1.96 (95% CI)   (calibration; target = 0.95)
  IS       = Gneiting-Raftery 95% interval score, mean over i      (accuracy+calibration; lower)
             [L_i, U_i] = mu_i -/+ z s_i, alpha = 0.05:
             IS_i = (U_i - L_i) + (2/alpha)(L_i - f_i)_+ + (2/alpha)(f_i - U_i)_+

RRMSE normalizes by the per-level std of the TRUE surface over the test set, so it is comparable
across levels and problems (RRMSE ~ 1 means "no better than predicting the level mean").
"""
import numpy as np
import pandas as pd

Z95 = 1.959963984540054           # Phi^{-1}(0.975): two-sided 95% interval
ALPHA = 0.05
Z90 = Z95                         # legacy alias (metric switched from 90% to 95% on request)
COV_TARGET = 1.0 - ALPHA


def interval_score(f, lo, hi, alpha=ALPHA):
    """Mean Gneiting-Raftery interval score of the (1-alpha) interval [lo, hi] for truth f."""
    f, lo, hi = (np.asarray(a, float).ravel() for a in (f, lo, hi))
    return float(np.mean((hi - lo)
                         + (2.0 / alpha) * np.maximum(lo - f, 0.0)
                         + (2.0 / alpha) * np.maximum(f - hi, 0.0)))


def level_metrics(f_true, mu, s):
    """The 4 table metrics for ONE level. s = predictive std used for the 90% interval."""
    f, mu, s = (np.asarray(a, float).ravel() for a in (f_true, mu, s))
    err = mu - f
    sd = f.std()
    lo, hi = mu - Z95 * s, mu + Z95 * s
    return dict(
        MAE=float(np.mean(np.abs(err))),
        RRMSE=float(np.sqrt(np.mean(err ** 2)) / (sd if sd > 0 else 1.0)),
        IS=interval_score(f, lo, hi),
        Coverage=float(np.mean(np.abs(err) <= Z95 * s)),
    )


def metrics_table(spec, model, test_X_by_level, level_label=None, add_noise_var=False):
    """Per-level metrics DataFrame (one row per level + 'Mean' row) for one fitted model.

    test_X_by_level : {level: (n, d) array} of test inputs (dense grid or Sobol sample).
    level_label     : optional fn(level)->str for the row index (default 'level l' or cat value).
    add_noise_var   : if True, score the OBSERVATION interval (s^2 + r) instead of the epistemic
                      one -- only meaningful for models whose r() is finite (heteroscedastic).
    """
    rows, idx = [], []
    for lv in spec.levels:
        Xt = np.atleast_2d(np.asarray(test_X_by_level[lv], float))
        f = np.ravel(spec.f_true_level(Xt, lv))
        mu, s = model.mean_std(lv, Xt)
        if add_noise_var:
            r = np.asarray(model.r(lv, Xt), float).ravel()
            s = np.sqrt(s ** 2 + np.maximum(r, 0.0))
        rows.append(level_metrics(f, mu, s))
        if level_label is not None:
            idx.append(level_label(lv))
        elif spec.meta.get("cat_values"):
            idx.append(f"x{spec.d + 1} = {spec.meta['cat_values'][lv - 1]:g}")
        else:
            idx.append(f"level {lv}")
    df = pd.DataFrame(rows, index=idx)
    df.loc["Mean"] = df.mean()
    return df


def combined_table(tables_by_model):
    """Side-by-side comparison: {model_label: per-level DataFrame} -> one MultiIndex-column table
    (level rows x (model, metric) columns), like the study's summary figure."""
    return pd.concat(tables_by_model, axis=1)


def rank_summary(tables_by_model):
    """Mean-row summary across models: one row per model with the 4 pooled metrics, plus the
    winner per metric (min for MAE/RRMSE/IS; closest to COV_TARGET for Coverage)."""
    out = pd.DataFrame({m: t.loc["Mean"] for m, t in tables_by_model.items()}).T
    best = {c: (out[c].sub(COV_TARGET).abs().idxmin() if c == "Coverage" else out[c].idxmin())
            for c in out.columns}
    out.loc["<best>"] = [best[c] for c in out.columns]
    return out


# ---------------------------------------------------------------------------
# Additional UQ metrics (section 9-10 of the study notebook).
# Miscalibration area + NLL follow Ozbayram et al. (CMAME 2024) / Tran et al. (2020);
# noise-surface recovery (nMAE) is possible here because the problems are synthetic
# (analytic sigma^2), mirroring the paper's 1-D known-noise test case quantitatively.
# ---------------------------------------------------------------------------
from scipy.stats import norm as _norm


def miscalibration_area(f_true, mu, s, n_grid=101):
    """Area between the ECDF of normalized residuals z=(f-mu)/s and the standard-Gaussian CDF
    (Tran et al. 2020). 0 = perfectly calibrated at every level; ~0.5 = maximally miscalibrated."""
    f, mu, s = (np.asarray(a, float).ravel() for a in (f_true, mu, s))
    z = (f - mu) / np.maximum(s, 1e-12)
    ps = np.linspace(0.01, 0.99, n_grid)
    obs = np.array([(z <= q).mean() for q in _norm.ppf(ps)])
    return float(np.trapezoid(np.abs(obs - ps), ps))


def nll(f_true, mu, s):
    """Mean Gaussian negative log predictive density of the truth (proper score; lower better).
    Punishes overconfidence quadratically in the standardized error."""
    f, mu, s = (np.asarray(a, float).ravel() for a in (f_true, mu, s))
    s2 = np.maximum(s, 1e-12) ** 2
    return float(np.mean(0.5 * np.log(2.0 * np.pi * s2) + 0.5 * (f - mu) ** 2 / s2))


def noise_nmae(r, sigma2_true):
    """Normalized MAE of the predicted aleatoric VARIANCE surface r(x) vs the true sigma^2:
    mean|r - sigma2| / mean sigma2. NaN-safe (the standard LVGP has r = NaN)."""
    r, s2 = (np.asarray(a, float).ravel() for a in (r, sigma2_true))
    if np.all(np.isnan(r)):
        return float("nan")
    return float(np.nanmean(np.abs(r - s2)) / np.mean(s2))
