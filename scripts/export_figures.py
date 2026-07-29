"""Render the truth-function figures for the LaTeX document (latex/figs/*.pdf).
One figure per problem: true f per level (+/- 2 sigma noise band) and the true noise std.
Griewank (5-D) is shown as the main-effect slice along x1 (other dims = 0)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import problems as P

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "latex", "figs")
os.makedirs(OUT, exist_ok=True)


def onedim(problem, fname):
    spec = P.get(problem)
    xs = np.linspace(spec.lb, spec.ub, 400)
    cm = plt.cm.viridis(np.linspace(0, .9, spec.n_levels))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2))
    for lv in spec.levels:
        f = spec.f_true_level(xs, lv); sg = spec.sigma_level(xs, lv)
        lab = f"$x_2$={spec.meta['cat_values'][lv-1]:g}"
        axes[0].plot(xs, f, color=cm[lv-1], lw=1.6, label=lab)
        axes[0].fill_between(xs, f-2*sg, f+2*sg, color=cm[lv-1], alpha=.15, lw=0)
        axes[1].plot(xs, sg, color=cm[lv-1], lw=1.6, label=lab)
    axes[0].set(title="true $f$ per level ($\\pm2\\sigma$ band)", xlabel="$x_1$", ylabel="$f$")
    axes[1].set(title="true noise std $\\sigma(x_1,\\ell)$", xlabel="$x_1$", ylabel="$\\sigma$")
    for ax in axes:
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname)


def griewank(fname):
    spec = P.get("griewank_6d")
    ts = np.linspace(spec.lb, spec.ub, 400)
    XS = np.zeros((len(ts), 5)); XS[:, 0] = ts
    PAIR = {1: ("C0", "-"), 2: ("C0", "--"), 3: ("C3", "-"), 4: ("C3", "--")}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2))
    for lv in spec.levels:
        c, lsty = PAIR[lv]
        f = spec.f_true_level(XS, lv); sg = np.ravel(spec.sigma_level(XS, lv))
        lab = f"lv {lv} ($v$={spec.meta['cat_values'][lv-1]:g})"
        axes[0].plot(ts, f, color=c, ls=lsty, lw=1.7, label=lab)
        axes[0].fill_between(ts, f-2*sg, f+2*sg, color=c, alpha=.12, lw=0)
        axes[1].plot(ts, sg, color=c, ls=lsty, lw=1.7, label=lab)
    axes[0].set(title="main-effect slice: true $f$ per level ($\\pm2\\sigma$ band)",
                xlabel="$x_1$ (other dims $=0$)", ylabel="$f$")
    axes[1].set(title="noise std on the slice", xlabel="$x_1$ (other dims $=0$)",
                ylabel="$\\sigma$")
    for ax in axes:
        ax.legend(fontsize=7); ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname)


if __name__ == "__main__":
    onedim("branin_hetero", "branin_truth.pdf")
    onedim("sixhump_camel", "camel_truth.pdf")
    griewank("griewank_truth.pdf")
