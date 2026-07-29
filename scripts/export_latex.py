"""Generate the study's LaTeX tables into latex/ (one file per table + tables_main.tex wrapper).

Two versions of every table (same data, winners recomputed per version):
  * full   -- Separate GPs / Categorical GP / Standard LVGP / H-LVGP
  * _nocat -- without the Categorical GP (3-model comparison)

Layout (booktabs + multirow, winner per row/metric in bold on green):
  * summary[_nocat].tex            -- mean-over-levels: Problem (multirow) x Budget (multirow)
                                      x Model rows; column blocks n_rep=10 / n_rep=3.
  * <problem>_nrep<NN>[_nocat].tex -- per-level: Budget (multirow) x Level rows; column blocks
                                      the models, each MAE/RRMSE/IS/Cov (screenshot layout).
Winner rule: min for MAE/RRMSE/IS; closest to 0.95 for Coverage.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import problems as P, study, metrics as MET

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "latex")
os.makedirs(OUT, exist_ok=True)

LABELS = {"separate_gp": "Separate GPs", "categorical_kernel": "Categorical GP",
          "standard_LVGP": "Standard LVGP", "heter_LVGP": "H-LVGP"}
SETS = [("", ["separate_gp", "categorical_kernel", "standard_LVGP", "heter_LVGP"]),
        ("_nocat", ["separate_gp", "standard_LVGP", "heter_LVGP"])]
METRICS = ["MAE", "RRMSE", "IS", "Coverage"]
ARROWS = {"MAE": r"MAE~$\downarrow$", "RRMSE": r"RRMSE~$\downarrow$",
          "IS": r"IS~$\downarrow$", "Coverage": r"Cov.~$\to.95$"}
PROBLEMS = [("branin_hetero", "Branin", [(25, 25), (40, 40)]),
            ("sixhump_camel", "Six-hump camel", [(25, 25), (40, 40)]),
            ("griewank_6d", "Griewank 6-D", [(40, 40), (80, 80), (160, 160)])]
SEED, N_REPS = 1, [10]

_cache = {}


def table_of(prob, n_init, n_rep, key):
    ck = (prob, n_init, n_rep, key)
    if ck not in _cache:
        spec = P.get(prob)
        _cache[ck] = MET.metrics_table(
            spec, study.CachedPredictions(prob, key, SEED, n_rep, n_init), study.test_points(spec))
    return _cache[ck]


def tabs(prob, n_init, n_rep, keys):
    return {LABELS[k]: table_of(prob, n_init, n_rep, k) for k in keys}


def fmt(v):
    if v >= 100: return f"{v:.1f}"
    if v >= 10:  return f"{v:.2f}"
    if v >= 1:   return f"{v:.3f}"
    return f"{v:.4f}"


def winner(vals, metric):
    if metric == "Coverage":
        return min(vals, key=lambda m: abs(vals[m] - 0.95))
    return min(vals, key=vals.get)


def cell(v, win, highlight=True):
    """Winner formatting: bold always; the green box only when `highlight` (Mean rows)."""
    s = f"${fmt(v)}$"
    if not win:
        return s
    return (r"\colorbox{green!20}{\boldmath" + s + "}") if highlight else (r"{\boldmath" + s + "}")


def per_level_table(prob, disp, budgets, n_rep, keys, sfx):
    models = [LABELS[k] for k in keys]
    note = "" if not sfx else " (without the Categorical GP)"
    L = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
         r"\setlength{\tabcolsep}{2.5pt}", r"\setlength{\fboxsep}{1pt}",
         r"\caption{%s --- per-level metrics%s, $n_{\mathrm{rep}}=%d$ (seed %d; MAE/RRMSE/IS"
         r" lower is better, coverage target $0.95$). Bold = best model in the row; the green"
         r" highlight marks the winners of the \emph{Mean} row.}" %
         (disp, note, n_rep, SEED),
         r"\label{tab:%s_nrep%02d%s}" % (prob, n_rep, sfx),
         r"\begin{tabular}{ll%s}" % ("rrrr" * len(models)), r"\toprule",
         r" & & " + " & ".join(r"\multicolumn{4}{c}{\textbf{%s}}" % m for m in models) + r" \\",
         " ".join(r"\cmidrule(lr){%d-%d}" % (3 + 4 * i, 6 + 4 * i) for i in range(len(models))),
         r"$n$ & Level & " + " & ".join(" & ".join(ARROWS[met] for met in METRICS)
                                        for _ in models) + r" \\",
         r"\midrule"]
    for bi, (n_init, n_shown) in enumerate(budgets):
        T = tabs(prob, n_init, n_rep, keys)
        levels = list(T[models[0]].index)
        for li, lvl in enumerate(levels):
            lead = (r"\multirow{%d}{*}{%d}" % (len(levels), n_shown)) if li == 0 else ""
            name = r"\textit{Mean}" if lvl == "Mean" else lvl.replace("x2", "$x_2$")
            row = [lead, name]
            for m in models:
                for met in METRICS:
                    vals = {mm: float(T[mm].loc[lvl, met]) for mm in models}
                    row.append(cell(vals[m], m == winner(vals, met), highlight=(lvl == "Mean")))
            L.append(" & ".join(row) + r" \\")
        L.append(r"\midrule" if bi < len(budgets) - 1 else r"\bottomrule")
    L += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def summary_table(keys, sfx):
    models = [LABELS[k] for k in keys]
    note = "" if not sfx else " (without the Categorical GP)"
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\setlength{\tabcolsep}{3.5pt}", r"\setlength{\fboxsep}{1.5pt}",
         r"\caption{Mean-over-levels metrics for every problem $\times$ budget $\times$ model%s"
         r" (seed %d). Bold on green = best model in each (problem, budget, $n_{\mathrm{rep}}$)"
         r" block per metric (MAE/RRMSE/IS: lower; coverage: closest to $0.95$).}" % (note, SEED),
         r"\label{tab:summary%s}" % sfx,
         r"\begin{tabular}{lll%s}" % ("rrrr" * len(N_REPS)), r"\toprule",
         r" & & & " + " & ".join(r"\multicolumn{4}{c}{$n_{\mathrm{rep}}=%d$}" % nr
                                 for nr in N_REPS) + r" \\",
         " ".join(r"\cmidrule(lr){%d-%d}" % (4 + 4 * i, 7 + 4 * i) for i in range(len(N_REPS))),
         r"Problem & $n$ & Model & " + " & ".join(" & ".join(ARROWS[m] for m in METRICS)
                                                  for _ in N_REPS) + r" \\",
         r"\midrule"]
    for pi, (prob, disp, budgets) in enumerate(PROBLEMS):
        nrow_p = len(budgets) * len(models)
        first_p = True
        for bi, (n_init, n_shown) in enumerate(budgets):
            per_rep = {nr: tabs(prob, n_init, nr, keys) for nr in N_REPS}
            for mi, m in enumerate(models):
                lead_p = (r"\multirow{%d}{*}{%s}" % (nrow_p, disp)) if first_p else ""
                lead_b = (r"\multirow{%d}{*}{%d}" % (len(models), n_shown)) if mi == 0 else ""
                first_p = False
                row = [lead_p, lead_b, m]
                for nr in N_REPS:
                    T = per_rep[nr]
                    for met in METRICS:
                        vals = {mm: float(T[mm].loc["Mean", met]) for mm in models}
                        row.append(cell(vals[m], m == winner(vals, met)))
                L.append(" & ".join(row) + r" \\")
            if bi < len(budgets) - 1:
                L.append(r"\cmidrule(lr){2-%d}" % (3 + 4 * len(N_REPS)))
        L.append(r"\bottomrule" if pi == len(PROBLEMS) - 1 else r"\midrule")
    L += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def wrapper():
    inputs = []
    inputs.append("\\input{problem_figures.tex}")
    for sfx, _ in SETS:
        title = ("\\section*{Version 1 --- all four models}" if not sfx else
                 "\\clearpage\n\\section*{Version 2 --- without the Categorical GP}")
        inputs.append(title)
        inputs.append("\\input{summary%s.tex}\n\\clearpage" % sfx)
        if not sfx:
            inputs.append("\\input{conclusions.tex}\n\\clearpage")
            inputs.append("\\input{seed_summary.tex}\n\\clearpage")
        for prob, _, _ in PROBLEMS:
            for nr in N_REPS:
                inputs.append("\\input{%s_nrep%02d%s.tex}" % (prob, nr, sfx))
            inputs.append("\\clearpage")
    body = "\n".join(inputs)
    return r"""\documentclass[11pt]{article}
\usepackage[margin=0.7in,landscape]{geometry}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage[dvipsnames]{xcolor}
\usepackage{amsmath}
\usepackage{graphicx}
\title{Surrogate-model comparison --- heteroscedastic mixed-variable test problems\\
\large Separate GPs / Categorical GP / Standard LVGP (MATLAB) / H-LVGP (MATLAB)}
\author{}
\date{}
\begin{document}
\maketitle
\noindent\textbf{Legend.} Metrics on a ground-truth test set (200-pt grid per level for 1-D;
1024 Sobol points per level for Rastrigin 6-D): MAE and RRMSE (surface accuracy), 95\%
Gneiting--Raftery interval score IS (accuracy $+$ calibration, $\alpha=0.05$), and 95\% coverage
(target $0.95$; higher is \emph{not} better). \colorbox{green!20}{\textbf{Bold on green}} = best model
in the comparison group (winners are recomputed within each version's model set). Budgets $n$
are TOTAL design points; designs are SLHD with $n_{\mathrm{rep}}$ replicates per point.
Single seed.
\vspace{1em}
""" + body + "\n\\end{document}\n"


def main():
    for sfx, keys in SETS:
        open(os.path.join(OUT, f"summary{sfx}.tex"), "w").write(summary_table(keys, sfx))
        print(f"summary{sfx}.tex", flush=True)
        for prob, disp, budgets in PROBLEMS:
            for nr in N_REPS:
                f = f"{prob}_nrep{nr:02d}{sfx}.tex"
                open(os.path.join(OUT, f), "w").write(
                    per_level_table(prob, disp, budgets, nr, keys, sfx))
                print(f, flush=True)
    open(os.path.join(OUT, "tables_main.tex"), "w").write(wrapper())
    print("tables_main.tex")
    github_variant()




# ---------------------------------------------------------------------------
# GitHub variant: tables + test-function definitions/figures ONLY (no legend,
# no conclusions, no commentary). Self-contained under results_v2/latex/.
# ---------------------------------------------------------------------------
GITHUB_OUT = os.path.join(HERE, "latex_v2")

PROBLEM_DEFS = r"""\section*{Test problems}

\subsection*{Branin (heteroscedastic, 5 levels)}
$$f(x_1,\ell)=\Big(x_2(\ell)-\tfrac{5.1}{4\pi^2}x_1^2+\tfrac{5}{\pi}x_1-6\Big)^2
+10\Big(1-\tfrac{1}{8\pi}\Big)\cos x_1+10,\quad x_1\in[-5,10],\ x_2(\ell)\in\{15,2,8,0,10\}$$
$$\sigma(x_1,\ell)=\Big[0.5+0.15(x_1+5)+\tfrac{2.5}{1+e^{-1.2(x_1-2)}}\Big]\,m(\ell),
\quad m(\ell)\in\{0.5,1.0,1.8,2.5,3.5\}$$
\begin{center}\includegraphics[width=0.85\linewidth]{figs/branin_truth.pdf}\end{center}

\subsection*{Six-hump camel (5 levels: two pairs + bridge)}
$$f(x_1,\ell)=\Big(4-2.1x_1^2+\tfrac{x_1^4}{3}\Big)x_1^2+x_1 x_2(\ell)+(-4+4x_2(\ell)^2)x_2(\ell)^2,
\quad x_1\in[-2,2],\ x_2(\ell)\in\{-1.0,-0.85,0.0,0.85,1.0\}$$
$$\sigma(x_1,\ell)=0.05\,e^{(0.4x_1)^2}\,m(\ell),\quad m(\ell)\in\{2.0,3.5,1.5,5.0,2.5\}$$
\begin{center}\includegraphics[width=0.85\linewidth]{figs/camel_truth.pdf}\end{center}

\subsection*{Griewank 6-D (level-shifted pairs)}
$$f(\mathbf{x}_q,\ell)=\sum_{i=1}^{6}\tfrac{xs_i^2}{4000}-\prod_{i=1}^{6}\cos\big(\tfrac{xs_i}{\sqrt i}\big)+1+b(\ell),
\quad xs=[\mathbf{x}_q, v(\ell)],\ \mathbf{x}_q\in[-3,3]^5$$
$$v(\ell)\in\{0.5,1.3,2.7,3.1\},\quad b(\ell)\in\{0,0.15,0.6,0.75\},\quad
\sigma(\mathbf{x}_q,\ell)=0.02\sqrt{1+0.1\|\mathbf{x}_q\|^2/5}\;m(\ell),\quad
m(\ell)\in\{1.5,1.0,3.0,2.0\}$$
\begin{center}\includegraphics[width=0.85\linewidth]{figs/griewank_truth.pdf}\end{center}
\clearpage
"""


def github_variant():
    import shutil
    os.makedirs(GITHUB_OUT, exist_ok=True)
    figs = os.path.join(GITHUB_OUT, "figs")
    os.makedirs(figs, exist_ok=True)
    for f in ["branin_truth.pdf", "camel_truth.pdf", "griewank_truth.pdf"]:
        shutil.copy2(os.path.join(OUT, "figs", f), os.path.join(figs, f))
    table_files = ["summary.tex", "summary_nocat.tex"]
    for prob, _, _ in PROBLEMS:
        for nr in N_REPS:
            for sfx in ["", "_nocat"]:
                table_files.append(f"{prob}_nrep{nr:02d}{sfx}.tex")
    if os.path.exists(os.path.join(OUT, "seed_summary.tex")):
        table_files.insert(2, "seed_summary.tex")
    for f in table_files:
        shutil.copy2(os.path.join(OUT, f), os.path.join(GITHUB_OUT, f))
    open(os.path.join(GITHUB_OUT, "problems_def.tex"), "w").write(PROBLEM_DEFS)
    body = "\n".join(["\\input{problems_def.tex}"]
                     + [f"\\input{{{f}}}" for f in table_files])
    main = ("\\documentclass[11pt]{article}\n"
            "\\usepackage[margin=0.7in,landscape]{geometry}\n"
            "\\usepackage{booktabs}\n\\usepackage{multirow}\n"
            "\\usepackage[dvipsnames]{xcolor}\n\\usepackage{amsmath}\n"
            "\\usepackage{graphicx}\n"
            "\\begin{document}\n" + body + "\n\\end{document}\n")
    open(os.path.join(GITHUB_OUT, "main.tex"), "w").write(main)
    print("github variant ->", GITHUB_OUT)


if __name__ == "__main__":
    main()
