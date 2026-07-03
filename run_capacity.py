"""Multi-pattern capacity ensemble (H6/H6b), scale-free at the calibrated T.

5 replicates (fresh graph, dynamics RNG, pattern set each) x strategy x P.
Per cell: cued-pattern fidelity, best non-cued overlap, r_struct, r_mag.
P=1 must reproduce the H5 ensemble numbers (regression gate).

Writes data/ensemble_capacity.csv.
"""
from __future__ import annotations

import csv

import numpy as np

import dtm_col_infl_clamping as dtm

N = 196
N_SEEDS = 5
FRAC = 0.10
T = 6.9                     # 1.15 x Tc(scale_free), run_tc.py
RATIO = 2.0
P_GRID = (1, 2, 3, 5, 8, 12)


def one_cell(strat, P, seed):
    dtm.rng = np.random.default_rng(seed)
    G = dtm.make_graph("scale_free", N, seed=seed)
    return dtm.multi_pattern_recall(G, P, FRAC, strategy=strat, ratio=RATIO,
                                    T=T, pattern_seed=1000 + seed)


def main():
    rows = []
    print(f"scale_free N={N}, frac={FRAC}, T={T}, B*J/T={RATIO}, "
          f"{N_SEEDS} seeds")
    print("strategy   P   fid_cued      fid_other     r_struct      r_mag")
    for strat in ("random", "degree"):
        for P in P_GRID:
            ms = [one_cell(strat, P, s) for s in range(N_SEEDS)]
            agg = {k: (float(np.mean([m[k] for m in ms])),
                       float(np.std([m[k] for m in ms])))
                   for k in ("fid", "fid_other", "r_struct", "r_mag")}
            rows.append(dict(strategy=strat, P=P,
                             **{f"{k}_{s}": round(v, 3)
                                for k, (mu, sd) in agg.items()
                                for s, v in (("mean", mu), ("std", sd))}))
            print(f"{strat:8s}  {P:2d}  " +
                  "  ".join(f"{agg[k][0]:5.2f}+-{agg[k][1]:4.2f}"
                            for k in ("fid", "fid_other", "r_struct", "r_mag")))

    path = "data/ensemble_capacity.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
