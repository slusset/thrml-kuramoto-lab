"""H8a: the heterogeneity penalty across the Ising-to-clock/XY state dial.

For each q in Q_GRID, on scale-free N=196:
  Phase 1 (step 0, per q): unclamped clock model across a temperature grid;
    The finite-graph ordering crossover is where |<e^{i·theta}>| crosses 0.5.
    It moves with q, so every q gets its own operating-point calibration.
  Phase 2: fid vs carrier fraction at T = 1.15*Tc(q), random vs degree
    placement, N_SEEDS seeds. The heterogeneity penalty is the degree-minus-
    random fidelity gap; H8a predicts it shrinks as q grows.

q=2 is a strict Ising regression gate (clock_sweep reduces to Glauber, and
the fractions/protocol match ensemble_placement.csv).

As q grows this model approaches an equilibrium XY state space. It does not
become generic Kuramoto dynamics, which also include natural frequencies and
nonequilibrium phase evolution.

Writes data/ensemble_clock.csv.
"""
from __future__ import annotations

import csv

import numpy as np

import dtm_col_infl_clamping as dtm

N = 196
Q_GRID = (2, 4, 8, 16)
N_SEEDS = 3
RATIO = 2.0
TC_SAFETY = 1.15
FRACTIONS = (0.01, 0.037, 0.064, 0.091, 0.146, 0.20)
T_SCAN = np.linspace(1.0, 10.0, 10)


def spontaneous_order(G, q, T, seed, burn=80, record=80):
    """Kuramoto order parameter of the unclamped clock model at T."""
    dtm.rng = np.random.default_rng(seed)
    Nn = G.number_of_nodes()
    adj = [list(G.neighbors(i)) for i in range(Nn)]
    clamp_mask = np.zeros(Nn, dtype=bool)
    free_idx = np.arange(Nn)
    s = dtm.rng.integers(0, q, size=Nn)
    s = dtm.clock_sweep(s, adj, clamp_mask, free_idx, q, 1.0, 1.0, T, burn)
    ms = []
    for _ in range(record):
        s = dtm.clock_sweep(s, adj, clamp_mask, free_idx, q, 1.0, 1.0, T, 1)
        ms.append(abs(np.exp(2j * np.pi * s / q).mean()))
    return float(np.mean(ms))


def locate_tc(G, q):
    rows = [(T, spontaneous_order(G, q, T, seed=0)) for T in T_SCAN]
    crossing = next((T for T, m in rows if m < 0.5), T_SCAN[-1])
    curve = "  ".join(f"{m:.2f}" for _, m in rows)
    print(f"  q={q:2d}  |m| over T=1..10: {curve}   ->  crossover ~ {crossing:.1f}")
    return crossing


def one_cell(G, q, T, frac, strat, seed):
    dtm.rng = np.random.default_rng(seed)
    target = np.zeros(G.number_of_nodes(), dtype=int)   # uniform phase 0
    clamp_idx = dtm.place_carriers(G, frac, strategy=strat)
    B = RATIO * T / 1.0
    s, free_idx, trace = dtm.sample_clock_equilibrium(
        G, target, clamp_idx, q, J=1.0, B=B, T=T, burn=40, record=60)
    return dtm.clock_metrics(s, free_idx, target, trace, q)["fid"]


def main():
    rows = []
    print("=== step 0 per q: finite-graph ordering crossover (seed-0 graph) ===")
    G0 = dtm.make_graph("scale_free", N, seed=0)
    tc = {q: locate_tc(G0, q) for q in Q_GRID}

    print(f"\n=== fid vs fraction at 1.15*crossover(q), {N_SEEDS} seeds ===")
    for q in Q_GRID:
        T = round(TC_SAFETY * tc[q], 2)
        print(f"\nq={q}  (crossover~{tc[q]:.1f} -> T={T})")
        print("strategy  " + "  ".join(f"{f:10.3f}" for f in FRACTIONS))
        gaps = []
        for strat in ("random", "degree"):
            cells = []
            for frac in FRACTIONS:
                fids = [one_cell(dtm.make_graph("scale_free", N, seed=sd),
                                 q, T, frac, strat, sd)
                        for sd in range(N_SEEDS)]
                mu, sd_ = float(np.mean(fids)), float(np.std(fids))
                cells.append(mu)
                rows.append(dict(q=q, T=T, strategy=strat, frac=frac,
                                 fid_mean=round(mu, 3), fid_std=round(sd_, 3)))
            print(f"{strat:8s}  " + "  ".join(f"{m:10.2f}" for m in cells))
            gaps.append(cells)
        gap = float(np.mean(np.array(gaps[1]) - np.array(gaps[0])))
        print(f"mean degree-minus-random gap: {gap:+.3f}")
        rows.append(dict(q=q, T=T, strategy="GAP", frac=-1,
                         fid_mean=round(gap, 3), fid_std=0.0))

    print("\n=== H8a summary: heterogeneity penalty vs q ===")
    for r in rows:
        if r["strategy"] == "GAP":
            print(f"  q={r['q']:2d}   mean placement gap {r['fid_mean']:+.3f}")

    path = "data/ensemble_clock.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
