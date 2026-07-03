"""H9: the kinetic clock dial — does the escape threshold melt as q grows?

Phase 1: refine Tc(q) on scale-free for q>=4 (the H8a scan used a unit-step
grid; the kinetic protocol sits at 0.75*Tc so the estimate must be finer).
Phase 2: metastable-flip thresholds. For each (q, strategy, seed), scan the
carrier fraction upward and record h_c = smallest fraction whose fidelity to
the NEW (antipodal) phase exceeds 0.5 within HORIZON sweeps, at matched
relative trap depth T = DEPTH * Tc(q). Median over seeds.

H9 predicts h_c(random) falls with q; q=2 is the original Ising flip.
Writes data/ensemble_kinetic.csv.
"""
from __future__ import annotations

import csv

import numpy as np

import dtm_col_infl_clamping as dtm

N = 196
Q_GRID = (2, 4, 8, 16)
N_SEEDS = 3
DEPTH = 0.6                     # T / Tc(q): matched relative trap depth.
                                # 0.75 + amplified clamps floored every h_c
                                # to the grid bottom (first run) - deepened.
B_CLAMP = 1.0                   # plain pinned spins: the original protocol
HORIZON = 150
FRACTIONS = (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.26)
FLIP_CRIT = 0.5
TC_COARSE = {2: 6.0}            # q=2 from run_tc.py / run_clock.py
T_REFINE = np.linspace(1.5, 4.0, 6)


def refine_tc(G, q):
    rows = [(T, run_order(G, q, T)) for T in T_REFINE]
    crossing = next((T for T, m in rows if m < 0.5), T_REFINE[-1])
    curve = "  ".join(f"{m:.2f}" for _, m in rows)
    print(f"  q={q:2d}  |m| over T=1.5..4.0: {curve}   ->  Tc ~ {crossing:.1f}")
    return float(crossing)


def run_order(G, q, T, burn=80, record=80, seed=0):
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


def main():
    rows = []
    print("=== phase 1: refine Tc(q) on scale-free (seed-0 graph) ===")
    G0 = dtm.make_graph("scale_free", N, seed=0)
    tc = dict(TC_COARSE)
    for q in Q_GRID:
        if q not in tc:
            tc[q] = refine_tc(G0, q)

    print(f"\n=== phase 2: kinetic thresholds (T={DEPTH}*Tc(q), "
          f"B={B_CLAMP} pinned clamps, horizon={HORIZON}, "
          f"flip = fid_new > {FLIP_CRIT}) ===")
    print("q    T     strategy   h_c per seed        median")
    for q in Q_GRID:
        T = round(DEPTH * tc[q], 2)
        for strat in ("random", "degree"):
            hcs = []
            for seed in range(N_SEEDS):
                G = dtm.make_graph("scale_free", N, seed=seed)
                dtm.rng = np.random.default_rng(seed)
                hc = np.nan
                for frac in FRACTIONS:
                    fid = dtm.kinetic_flip(G, q, frac, strategy=strat,
                                           T=T, horizon=HORIZON, B=B_CLAMP)
                    rows.append(dict(q=q, T=T, strategy=strat, seed=seed,
                                     frac=frac, fid_new=round(fid, 3)))
                    if fid > FLIP_CRIT:
                        hc = frac
                        break
                hcs.append(hc)
            med = float(np.nanmedian(hcs))
            disp = "  ".join(">0.24" if np.isnan(h) else f"{h:.2f}" for h in hcs)
            print(f"{q:2d}  {T:5.2f}  {strat:8s}   {disp:18s}  "
                  f"{'>0.24' if np.isnan(med) else f'{med:.2f}'}")
            rows.append(dict(q=q, T=T, strategy=f"HC_{strat}", seed=-1,
                             frac=med if not np.isnan(med) else 999,
                             fid_new=-1))

    path = "data/ensemble_kinetic.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
