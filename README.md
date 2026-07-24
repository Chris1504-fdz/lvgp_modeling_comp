# LVGP modeling study — heteroscedastic mixed-variable surrogates

**Purpose.** This repo studies the *modeling* quality of surrogates for heteroscedastic
mixed-variable (continuous × categorical) data: surface accuracy, uncertainty calibration, noise
capture, and latent-space recovery — as a function of training-set size, replicates, and problem
structure. It is deliberately **not** an optimization study: no BO loop, no acquisitions, no
regret. The sibling repo `hetero_lvgp` (at `/data/zhq7531/IDEAL/hetero_lvgp`, benchmark in
`benchmark/`) covers the Bayesian-optimization side; this repo isolates the question that study
kept pointing back to — *which model actually models this class of data best, and why*.

**Scope:** two test problems; four surrogate models — two python GPs and the two **MATLAB LVGP engines (models of record)** — scored on ground-truth-based metrics.

---

## Context for a new session (read this first)

### What an LVGP is

The Latent-Variable GP (Zhang, Tao, Chen & Apley, *Technometrics* 2019) handles categorical
inputs by learning a **2-D latent embedding** per categorical level: level ℓ ↦ z_ℓ ∈ R², jointly
with the kernel hyperparameters, by maximum likelihood. Inputs become [x_continuous, z_level] and
a standard Gaussian kernel `R(u,u') = exp(-Σ_i 10^{φ_i} (u_i-u'_i)²)` operates on the joint
vector (latent dims use φ=0). Identifiability gauge: level 1 is pinned at the origin and the 2nd
latent coordinate of level 2 at 0. The learned z-geometry *is* the model's estimate of how
similar the categorical levels are — recovering the true level-similarity structure is one of the
things this study measures.

### The heteroscedastic LVGP (the lab's model — "ours")

Observations are replicate means with replicate variances (n_rep noisy evaluations per design).
The heteroscedastic variant feeds those **empirical replicate variances into the correlation
matrix** — `R ← R + diag(σ̂²_i / scale²)` inside the likelihood — with the process variance σ²
profiled out analytically, responses **min–max normalized** to [0,1] (not z-scored), and an
**eps-ladder eigenvalue nugget** (floors 10^{-1}…10^{-8} with continuation, likelihood-selected)
guaranteeing conditioning. A **pooled aleatoric model** r(x, ℓ) — one ridge polynomial on
log-σ over features [x, z_ℓ] (degree 2 + pairwise cross terms) — predicts noise at unsampled
designs.

`utils/models/hetero_lvgp_native.py` is a **faithful numpy port of the lab's MATLAB
implementation** (`LVGP_fit_noise.m` / `LVGP_predict_noise.m`), validated against it in the
parent repo: identical latent optima (|z| error ~0.01 on shared data, same profile-NLL),
statistically equivalent BO outcomes, and equal-or-better loop-free modeling metrics. Two
hard-won implementation facts — do not regress them:

1. **The hyperparameter fit uses SLSQP with scrambled-Sobol multistart + the eps continuation.**
   Projected L-BFGS-B slides the latent onto the ±3 box bound and lands ~10 NLL above the true
   optimum (inflating posterior variance ~2–3×); SLSQP reaches the same optimum as MATLAB's
   interior-point fmincon at ~5 s/fit. (scipy's trust-constr also matches but is ~14× slower.)
2. **The aleatoric r(x) is the pooled [x, z] polynomial, not a per-level polynomial** — the
   per-level version inflates r ~4× on under-sampled levels.

### The four models compared here (same `fit/predict` interface)

**The LVGP models of record are the MATLAB implementations**, called through a thin bridge
(`matlab/mfit.m` + `matlab/mpredict.m`, wrapped by `utils/models/matlab_lvgp.py`): `fit()` runs
the actual `LVGP_fit` / `LVGP_fit_noise` (+ the aleatoric polynomial) in a `matlab -batch`
subprocess and predictions come from `LVGP_predict(_noise)`. Costs ~30-40 s per fit and ~5-25 s
per prediction batch (batch your grids; don't call in tight loops).

| registry key | engine | idea | noise treatment |
|---|---|---|---|
| `separate_gp` | python (botorch) | independent GP per level | replicate var as fixed noise, per level |
| `categorical_kernel` | python (botorch) | single GP, categorical kernel over levels | replicate var as fixed noise |
| `standard_LVGP` | **MATLAB** (`LVGP_fit`) | LVGP, homoscedastic | none — the noise-blind baseline (`r()` = NaN) |
| `heter_LVGP` | **MATLAB** (`LVGP_fit_noise`) | heteroscedastic LVGP | replicate var in the correlation matrix + aleatoric poly |

Reference-only entries (kept for fast iteration/debugging, NOT models of record):
`lvgp_native` / `heter_lvgp_native` — the validated numpy ports of the MATLAB code (details
below; useful when MATLAB round-trips are too slow for an experiment sweep, but any headline
result should be produced or confirmed with the MATLAB engines); `lvgp_torch` /
`heter_lvgp_torch` — the external `lvgp-bayes` wrapper, whose modeling choices diverge from the
MATLAB reference.

**Model interface** (all four):
```python
data = {level: {"X": (n_l, d) array, "y_mean": (n_l,), "y_var": (n_l,)} for each level}
model = MODELS[name].cls.fit(data, needs_r=True, bounds=spec.bounds)
mu, var = model.predict(level, X_new)     # posterior mean & epistemic variance, RAW y units
mu, s   = model.mean_std(level, X_new)    # mean & epistemic std
r       = model.r(level, X_new)           # predicted aleatoric VARIANCE (needs needs_r=True)
```

### The two test problems

Both come from `utils/problems.py` (analytic true `f_true_level(x, level)` and noise std
`sigma_level(x, level)` — so every metric can be computed against exact ground truth).

- **`branin_hetero`** — the low-dimensional categorical problem: d=1 continuous, **5 levels**,
  n_init=10. Its signature feature: within level 2, twin minima — a quiet one (x≈3.18, σ²≈1.4,
  the global optimum) and a lower-f-looking but noisy region (x≈9.4, σ²≈46). Levels {2,4} are
  similar to each other and dissimilar from {1,3,5} (the structure a good latent should recover).
- **`rastrigin_6d`** — the five-dimensional problem: d=5 continuous, **4 levels**, n_init=32,
  rugged multimodal surface. In the parent BO benchmark this was the problem where the simple
  categorical-kernel GP beat both LVGPs on accuracy (median EI regret 3.3 vs 26–32) — *why* is a
  central question for this study (latent misfit? lengthscale misfit? ruggedness vs the Gaussian
  kernel?).

### Data generation

`utils/doe_cache.py` builds shared SLHD (sliced Latin hypercube) designs with replicates:
`doe_cache.ensure(problem, seed, n_rep)` then `doe_cache.load(...)` → dict with `X_sample`
(cols = [x_1..x_d, level]), `Y_sample` (replicate means), `Var_sample` (replicate variances,
ddof=1), `Y_rep` (raw replicates). Deterministic per (problem, seed, n_rep) — same seed ⇒ same
design for every model (CRN). Cache dir `doe_cache/` is auto-created and gitignored.

### Metrics (evaluate on a dense ground-truth grid, NOT on training data)

| metric | definition | reads as | target |
|---|---|---|---|
| MAE | mean \|μ − f_true\| | surface accuracy | low |
| RRMSE | RMSE / std(f_true on grid) | relative accuracy, cross-problem comparable | low |
| cov90 | frac(\|μ − f_true\| ≤ 1.645·s) | calibration of the 90% CI | ≈ 0.90 |
| IS90 | Gneiting 90% interval score (width + 20×miss), normalized | accuracy+calibration in one | low |
| nMAE | mean \|r − σ²_true\| / mean σ²_true | noise capture (heter models) | low |
| latent ρ | Spearman(latent level-distances, true level-distances) | latent recovery (LVGPs) | → 1 |

Headline finding from the parent repo that this study deepens: the noise-blind LVGP is severely
**overconfident** (cov90 ≈ 0.02–0.30) while the heteroscedastic treatment restores ≈ 0.9–1.0 —
heteroscedasticity buys calibration, not raw accuracy (the categorical GP led accuracy overall).

### Suggested experiment axes

1. **Sample-size sweep**: n_train ∈ {10, 20, 40, 80, …} × metrics (learning curves per model).
2. **Replicate sweep**: n_rep ∈ {1, 3, 10, 30} at fixed budget n_train × n_rep (means-vs-replicates
   trade-off — where does knowing σ̂² start paying?).
3. **Calibration curves**: empirical coverage vs nominal over τ ∈ (0.5, 0.99).
4. **Latent recovery**: fitted z-geometry vs true level-distance (RMS gap between level curves),
   as a function of n — when does the LVGP have enough data to find the structure?
5. **Noise-surface recovery**: r(x, ℓ) vs true σ²(x, ℓ) over the domain.
6. Seeds: designs are cheap here — use ≥ 20 seeds per configuration (fits are seconds each).

### Environment

- MATLAB: `/data/zhq7531/MATLAB/bin/matlab` (R2026a, headless `-nodisplay -batch`; the bridge
  sets isolated `MATLAB_PREFDIR`/`TMPDIR` per call and prepends `/data/zhq7531/envs/xvfblib/lib`
  to `LD_LIBRARY_PATH` — required on this node).
- Python: `/data/zhq7531/envs/ml_gp_env/bin/python` (torch 2.7.1, gpytorch 1.14, botorch 0.14.0,
  numpy 2.2.6, scipy 1.16). Register as a Jupyter kernel or run notebooks on the `ml_gp_env`
  kernel.
- `lvgp_torch` additionally needs the `lvgp-bayes` package (installed editable in ml_gp_env from
  `/data/zhq7531/IDEAL/hetero_lvgp/testing_lvgp_bayes`; import is guarded — absence is fine).

### Layout

```
lvgp_modeling_study/
├── README.md                  <- this file
├── .gitignore
├── utils/
│   ├── problems.py            <- 10 test problems w/ analytic f_true & sigma (2 used here)
│   ├── doe.py                 <- SLHD design generators
│   ├── doe_cache.py           <- deterministic cached designs w/ replicates
│   └── models/                <- the surrogate models (see table above)
├── matlab/                    <- the lab's LVGP code (standard_lvgp/, heter_lvgp/) + the
│                                 mfit.m / mpredict.m bridge + standalone aleatoric poly fns
├── notebooks/modeling.ipynb   <- starter: fit all 4 models, metrics vs ground truth
├── notes/                     <- personal working notes (gitignored)
├── results/                   <- experiment outputs (gitignored; regenerable)
└── plots/                     <- figures (gitignored; regenerable)
```

Code snapshot: copied from `hetero_lvgp/benchmark/utils` on 2026-07-22 (post-validation state).
If the parent repo's models evolve, re-sync deliberately — do not assume they stay in lockstep.
