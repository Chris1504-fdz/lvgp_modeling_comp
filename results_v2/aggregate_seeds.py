"""Aggregate the 30-seed fixed-design campaign (results_v2/pred_cache) into per-seed metric
tables and mean +/- std summaries. Outputs:
  results_v2/seed_metrics.csv  -- one row per (problem, budget, model, seed)
  results_v2/seed_summary.csv  -- mean/std/median over seeds per (problem, budget, model)
Convention: fixed design (seed 1's SLHD) + per-seed noise replicates => std measures
noise-realization + refit variability CONDITIONAL on the design."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

from utils import problems as P, study, metrics as MET

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_KEYS = ["separate_gp", "categorical_kernel", "standard_LVGP", "heter_LVGP",
              "heter_lvgp_torch"]
LABELS = {"separate_gp": "Separate GPs", "categorical_kernel": "Categorical GP",
          "standard_LVGP": "Standard LVGP", "heter_LVGP": "H-LVGP",
          "heter_lvgp_torch": "H-LVGP (torch)"}
CONFIGS = [("branin_hetero", 25), ("branin_hetero", 40),
           ("sixhump_camel", 25), ("sixhump_camel", 40),
           ("griewank_6d", 48), ("griewank_6d", 64),
           ("griewank_6d", 80), ("griewank_6d", 120), ("griewank_6d", 160),
           ("friedman_5d", 40), ("friedman_5d", 60), ("friedman_5d", 80)]
SEEDS = list(range(1, 31))
N_REP = 10


def main():
    rows = []
    for prob, ni in CONFIGS:
        spec = P.get(prob)
        Xt = study.test_points(spec)
        for k in MODEL_KEYS:
            for sd in SEEDS:
                try:
                    mdl = study.CachedPredictions(prob, k, sd, N_REP, ni, design_seed=1)
                except FileNotFoundError:
                    print(f"missing: {prob} n{ni} {k} seed{sd}", flush=True)
                    continue
                m = MET.metrics_table(spec, mdl, Xt).loc["Mean"]
                rows.append(dict(problem=prob, n=ni, model=LABELS[k], seed=sd,
                                 MAE=float(m.MAE), RRMSE=float(m.RRMSE),
                                 IS=float(m.IS), Coverage=float(m.Coverage)))
        print(f"aggregated {prob} n={ni}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "seed_metrics.csv"), index=False)
    summ = (df.groupby(["problem", "n", "model"])[["MAE", "RRMSE", "IS", "Coverage"]]
              .agg(["mean", "std", "median"]).round(5))
    summ.to_csv(os.path.join(HERE, "seed_summary.csv"))
    print(summ.to_string())
    print("\nwrote seed_metrics.csv + seed_summary.csv (rows:", len(df), ")")


if __name__ == "__main__":
    main()
