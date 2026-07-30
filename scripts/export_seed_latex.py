"""Generate the 30-seed replication table (seed_summary.tex) into latex/ and latex_v2/.

Reads results_v2/seed_summary.csv (written by results_v2/aggregate_seeds.py). Same winner
convention as export_latex.py's summary tables: best mean per (problem, budget) block in
bold on green (min for MAE/RRMSE/IS; Coverage closest to 0.95).
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(HERE, "results_v2", "seed_summary.csv")

DISP = {"branin_hetero": "Branin", "sixhump_camel": "Six-hump camel",
        "griewank_6d": "Griewank 6-D", "friedman_5d": "Friedman 5-D"}
SETS = [("", ["Separate GPs", "Categorical GP", "Standard LVGP", "H-LVGP"]),
        ("_nocat", ["Separate GPs", "Standard LVGP", "H-LVGP"])]
METRICS = ["MAE", "RRMSE", "IS", "Coverage"]


def fmt(v):
    if v >= 100: return f"{v:.1f}"
    if v >= 10:  return f"{v:.2f}"
    return f"{v:.3f}"


def cell(mean, std, win):
    s = f"${fmt(mean)} \\pm {fmt(std)}$"
    return r"\colorbox{green!20}{\boldmath" + s + "}" if win else s


def build(model_order, sfx):
    df = pd.read_csv(CSV, header=[0, 1], index_col=[0, 1, 2])
    probs = [p for p in DISP if p in df.index.get_level_values(0)]
    note = " Without the Categorical GP (3-model comparison; winners recomputed)." if sfx else ""
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{30-seed replication (fixed design, fresh noise replicates per seed): "
        r"mean $\pm$ std over seeds. Bold on green = best mean per (problem, budget) block "
        r"(coverage: closest to $0.95$)." + note + "}",
        rf"\label{{tab:seeds{sfx}}}", r"\begin{tabular}{lllrrrr}", r"\toprule",
        r"Problem & $n$ & Model & MAE & RRMSE & IS & Coverage \\", r"\midrule"]
    for pi, prob in enumerate(probs):
        if pi:
            lines.append(r"\midrule")
        budgets = sorted(set(df.loc[prob].index.get_level_values(0)))
        nrows = sum(len([m for m in model_order if (n, m) in df.loc[prob].index])
                    for n in budgets)
        first_p = True
        for bi, n in enumerate(budgets):
            if bi:
                lines.append(r"\cmidrule(lr){2-7}")
            models = [m for m in model_order if (n, m) in df.loc[prob].index]
            wins = {}
            for met in METRICS:
                vals = {m: df.loc[(prob, n, m), (met, "mean")] for m in models}
                wins[met] = (min(vals, key=lambda m: abs(vals[m] - 0.95))
                             if met == "Coverage" else min(vals, key=vals.get))
            for mi, m in enumerate(models):
                pcell = (rf"\multirow{{{nrows}}}{{*}}{{{DISP[prob]}}}"
                         if first_p else "")
                first_p = False
                ncell = rf"\multirow{{{len(models)}}}{{*}}{{{n}}}" if mi == 0 else ""
                row = [pcell, ncell, m]
                for met in METRICS:
                    row.append(cell(df.loc[(prob, n, m), (met, "mean")],
                                    df.loc[(prob, n, m), (met, "std")],
                                    wins[met] == m))
                lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    for sfx, model_order in SETS:
        tex = build(model_order, sfx)
        for sub in ("latex", "latex_v2"):
            out = os.path.join(HERE, sub, f"seed_summary{sfx}.tex")
            if os.path.isdir(os.path.dirname(out)):
                with open(out, "w") as fh:
                    fh.write(tex)
                print("wrote", out)


if __name__ == "__main__":
    main()
