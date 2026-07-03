"""H8b: does the capacity wall recede as covenants get richer (q x q)?

Phase 1 (per q): locate Tc of the P=1 centered Potts-Hebb model on
scale-free — unclamped, order parameter = chance-corrected overlap with the
stored pattern's gauge (max over global rotations), crossing 0.5. The
coupling scale changes with q (centered deltas), so per-q calibration is
mandatory, as ever.

Phase 2: capacity sweep at T = 1.15*Tc(q), held fixed as P grows (the H6
protocol). frac=0.10 degree-placed carriers cue pattern 0; report the
chance-corrected cued overlap. q=2 must reproduce H6's knee (gate).

Writes data/ensemble_capacity_q.csv.
"""
from __future__ import annotations

import csv

import numpy as np

import dtm_col_infl_clamping as dtm

N = 196
N_SEEDS = 3
FRAC = 0.10
RATIO = 2.0
TC_SAFETY = 1.15
Q_P_GRID = {
    2: (1, 2, 3, 5, 8, 12),
    4: (1, 2, 3, 5, 8, 12),
    8: (1, 2, 4, 8, 16, 24),
}
T_SCANS = {2: np.linspace(0.4, 2.6, 8),
           4: np.linspace(0.2, 1.6, 8),
           8: np.linspace(0.1, 1.2, 8)}


def spontaneous_potts_order(G, q, T, seed=0, burn=80, record=40):
    """Unclamped P=1 Potts-Hebb order, gauge-relative to the stored pattern."""
    dtm.rng = np.random.default_rng(seed)
    Nn = G.number_of_nodes()
    pats = np.random.default_rng(1000).integers(0, q, size=(1, Nn))
    W = dtm.potts_hebb_weights(G, pats, q)
    nbrs = [[] for _ in range(Nn)]
    for (i, j), M in W.items():
        nbrs[i].append((j, M))
        nbrs[j].append((i, M.T))
    clamp_mask = np.zeros(Nn, dtype=bool)
    free_idx = np.arange(Nn)
    s = dtm.rng.integers(0, q, size=Nn)
    s = dtm.potts_sweep(s, nbrs, clamp_mask, free_idx, q, 1.0, T, burn)
    ms = []
    for _ in range(record):
        s = dtm.potts_sweep(s, nbrs, clamp_mask, free_idx, q, 1.0, T, 1)
        rel = (s - pats[0]) % q
        f = np.bincount(rel, minlength=q) / Nn
        ms.append((q * f.max() - 1.0) / (q - 1.0))
    return float(np.mean(ms))


def locate_tc(G, q):
    grid = T_SCANS[q]
    rows = [(T, spontaneous_potts_order(G, q, T)) for T in grid]
    crossing = next((T for T, m in rows if m < 0.5), grid[-1])
    curve = "  ".join(f"{m:.2f}" for _, m in rows)
    print(f"  q={q}  |m| over T={grid[0]:.1f}..{grid[-1]:.1f}: {curve}"
          f"   ->  Tc ~ {crossing:.2f}")
    return float(crossing)


def main():
    rows = []
    print("=== phase 1: Tc(q) of the P=1 Potts-Hebb model (scale-free) ===")
    G0 = dtm.make_graph("scale_free", N, seed=0)
    tc = {q: locate_tc(G0, q) for q in Q_P_GRID}

    print(f"\n=== phase 2: capacity at T=1.15*Tc(q), frac={FRAC}, degree, "
          f"{N_SEEDS} seeds ===")
    for q, p_grid in Q_P_GRID.items():
        T = round(TC_SAFETY * tc[q], 3)
        print(f"\nq={q}  (Tc~{tc[q]:.2f} -> T={T})")
        print("  P    m_cued        m_other")
        for P in p_grid:
            mc, mo = [], []
            for seed in range(N_SEEDS):
                G = dtm.make_graph("scale_free", N, seed=seed)
                dtm.rng = np.random.default_rng(seed)
                r = dtm.potts_recall(G, q, P, FRAC, T, strategy="degree",
                                     ratio=RATIO, pattern_seed=1000 + seed)
                mc.append(r["m_cued"])
                mo.append(r["m_other"])
            mu, sd = float(np.mean(mc)), float(np.std(mc))
            rows.append(dict(q=q, T=T, P=P,
                             m_cued_mean=round(mu, 3), m_cued_std=round(sd, 3),
                             m_other_mean=round(float(np.mean(mo)), 3)))
            print(f" {P:3d}   {mu:5.2f}+-{sd:4.2f}   {np.mean(mo):5.2f}")

    print("\n=== H8b summary: P at last m_cued >= 0.7 (the knee) ===")
    for q in Q_P_GRID:
        good = [r["P"] for r in rows if r["q"] == q and r["m_cued_mean"] >= 0.7]
        print(f"  q={q}   P_0.7 = {max(good) if good else '<1'}")

    path = "data/ensemble_capacity_q.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
