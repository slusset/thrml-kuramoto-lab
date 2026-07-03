"""Ensemble pass over the calibrated placement sweep and Hopfield recall.

Everything in the 2026-07-03 sessions was single-seed. This runs N_SEEDS
replicates — new graph instance, new dynamics RNG, new stored pattern per
replicate — and reports mean +/- std so the tables become curves with error
bars. Tc per topology comes from run_tc.py (step 0); graph-instance Tc
wobble at N=196 is absorbed into the ensemble spread.

Writes data/ensemble_placement.csv and data/ensemble_hopfield.csv.
"""
from __future__ import annotations

import csv

import numpy as np

import dtm_col_infl_clamping as dtm

N = 196
N_SEEDS = 5
RATIO = 2.0                                  # B*J/T
TC = {"lattice": 2.5, "small_world": 4.0, "scale_free": 6.0}   # run_tc.py
TC_SAFETY = 1.15
FRACTIONS = (0.01, 0.037, 0.064, 0.091, 0.119, 0.146, 0.173, 0.20)
STRATEGIES = ("random", "degree", "ci")
HOP_FRACTIONS = (0.02, 0.06, 0.10, 0.16)


def one_placement_cell(topo, strat, frac, seed):
    dtm.rng = np.random.default_rng(seed)    # dynamics + random placement
    G = dtm.make_graph(topo, N, seed=seed)
    T = round(TC_SAFETY * TC[topo], 2)
    B = RATIO * T / 1.0
    target = np.ones(G.number_of_nodes(), dtype=int)
    clamp_idx = dtm.place_carriers(G, frac, strategy=strat)
    s, free_idx, trace = dtm.sample_equilibrium(
        G, target, clamp_idx, J=1.0, B=B, T=T, burn=40, record=60)
    return dtm.metrics(s, free_idx, target, trace)["fid"]


def one_hopfield_cell(strat, frac, seed):
    dtm.rng = np.random.default_rng(seed)
    G = dtm.make_graph("scale_free", N, seed=seed)
    T = round(TC_SAFETY * TC["scale_free"], 2)
    return dtm.hopfield_recall(G, frac, strategy=strat, ratio=RATIO, T=T,
                               pattern_seed=1000 + seed)


def main():
    place_rows, hop_rows = [], []

    for topo in TC:
        print(f"\n=== {topo}  (T = {TC_SAFETY} x {TC[topo]}, B*J/T = {RATIO}, "
              f"{N_SEEDS} seeds) ===")
        print("strategy  " + "  ".join(f"{f:11.3f}" for f in FRACTIONS))
        for strat in STRATEGIES:
            cells = []
            for frac in FRACTIONS:
                fids = [one_placement_cell(topo, strat, frac, s)
                        for s in range(N_SEEDS)]
                mu, sd = float(np.mean(fids)), float(np.std(fids))
                cells.append((mu, sd))
                place_rows.append(dict(topo=topo, strategy=strat, frac=frac,
                                       fid_mean=round(mu, 3), fid_std=round(sd, 3)))
            print(f"{strat:8s}  " +
                  "  ".join(f"{mu:5.2f}+-{sd:4.2f}" for mu, sd in cells))

    print(f"\n=== Hopfield recall  (scale_free, {N_SEEDS} seeds) ===")
    print("strategy  frac    fid           r_mag         r_struct")
    for strat in ("random", "degree"):
        for frac in HOP_FRACTIONS:
            ms = [one_hopfield_cell(strat, frac, s) for s in range(N_SEEDS)]
            agg = {k: (float(np.mean([m[k] for m in ms])),
                       float(np.std([m[k] for m in ms])))
                   for k in ("fid", "r_mag", "r_struct")}
            hop_rows.append(dict(strategy=strat, frac=frac,
                                 **{f"{k}_{s}": round(v, 3)
                                    for k, (mu, sd) in agg.items()
                                    for s, v in (("mean", mu), ("std", sd))}))
            print(f"{strat:8s}  {frac:.2f}  " +
                  "  ".join(f"{agg[k][0]:5.2f}+-{agg[k][1]:4.2f}"
                            for k in ("fid", "r_mag", "r_struct")))

    for name, rows in (("ensemble_placement", place_rows),
                       ("ensemble_hopfield", hop_rows)):
        path = f"data/{name}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
