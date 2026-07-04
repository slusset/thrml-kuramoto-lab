"""Finite-size scaling of the H8b capacity exponent.

Does the q=8 knee (8-14x at N=196, vs dense theory's 28x) rise toward the
quadratic law as N grows, or is it pinned by the sparse substrate?

Fixed density (BA m=3, <k> ~ 6 at every N) isolates system size. Per (q, N):
recalibrate Tc of the P=1 Potts-Hebb model (hub growth with N can shift it),
then run the H6/H8b capacity protocol at 1.15*Tc and interpolate the knee
P_0.7 where the chance-corrected cued overlap crosses 0.7.

q=4 is the control (already near theory at N=196; should be N-flat if the
theory is per-connection). Writes data/ensemble_fss.csv.
"""
from __future__ import annotations

import csv

import numpy as np

import dtm_col_infl_clamping as dtm

N_GRID = (196, 400, 800)
N_SEEDS = 3
FRAC = 0.10
RATIO = 2.0
TC_SAFETY = 1.15
P_GRIDS = {4: (2, 4, 6, 8, 12, 16), 8: (4, 8, 12, 16, 24, 32)}
T_SCANS = {4: np.linspace(0.6, 2.0, 8), 8: np.linspace(0.5, 1.9, 8)}
KNEE = 0.7


def spontaneous_potts_order(G, q, T, seed=0, burn=60, record=30):
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
    return float(crossing)


def knee_from_curve(ps, ms):
    """Linear-interpolate the P where m_cued crosses KNEE (descending)."""
    for k in range(1, len(ps)):
        if ms[k - 1] >= KNEE > ms[k]:
            p0, p1, m0, m1 = ps[k - 1], ps[k], ms[k - 1], ms[k]
            return p0 + (m0 - KNEE) * (p1 - p0) / (m0 - m1)
    if ms[0] < KNEE:
        return float(ps[0]) * (ms[0] / KNEE)      # below range, crude
    return float(ps[-1])                           # above range: >= last P


def main():
    rows = []
    knees = {}
    for q, p_grid in P_GRIDS.items():
        for Nn in N_GRID:
            G0 = dtm.make_graph("scale_free", Nn, seed=0)
            tc = locate_tc(G0, q)
            T = round(TC_SAFETY * tc, 3)
            print(f"\nq={q}  N={Nn}  Tc~{tc:.2f} -> T={T}")
            print("  P    m_cued")
            mus = []
            for P in p_grid:
                mc = []
                for seed in range(N_SEEDS):
                    G = dtm.make_graph("scale_free", Nn, seed=seed)
                    dtm.rng = np.random.default_rng(seed)
                    r = dtm.potts_recall(G, q, P, FRAC, T, strategy="degree",
                                         ratio=RATIO, pattern_seed=1000 + seed)
                    mc.append(r["m_cued"])
                mu, sd = float(np.mean(mc)), float(np.std(mc))
                mus.append(mu)
                rows.append(dict(q=q, N=Nn, T=T, P=P,
                                 m_cued_mean=round(mu, 3),
                                 m_cued_std=round(sd, 3)))
                print(f" {P:3d}   {mu:5.2f}+-{sd:4.2f}")
            knee = knee_from_curve(p_grid, mus)
            knees[(q, Nn)] = knee
            print(f"  knee P_{KNEE} ~ {knee:.1f}")
            rows.append(dict(q=q, N=Nn, T=T, P=-1,
                             m_cued_mean=round(knee, 2), m_cued_std=-1))

    print("\n=== FSS summary: knee P_0.7 by (q, N) ===")
    print("        " + "  ".join(f"N={n:4d}" for n in N_GRID))
    for q in P_GRIDS:
        print(f"  q={q}   " + "  ".join(f"{knees[(q, n)]:6.1f}" for n in N_GRID))
    print("dense-theory ratio q=8/q=4: 28/6 ~ 4.7x")
    for n in N_GRID:
        r = knees[(8, n)] / max(knees[(4, n)], 1e-9)
        print(f"  observed q=8/q=4 at N={n}: {r:.1f}x")

    path = "data/ensemble_fss.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
