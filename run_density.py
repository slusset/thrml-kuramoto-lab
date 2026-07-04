"""The complement claim, tested head-on: capacity vs web density at fixed q.

H8b-FSS said covenant richness and web density are complements — the
quadratic richness payoff should switch on only where density supports it.
Sweep BA density m in (3, 6, 12, 24) (<k> ~ 6..48) at N=196 for q in (4, 8):

  Prediction 1 (per-connection capacity): the knee P_0.7 rises ~linearly
    with <k> at fixed q.
  Prediction 2 (the complement): the q=8/q=4 knee ratio rises from ~2-3x
    at <k>=6 toward dense theory's q(q-1)/2 ratio of 4.7x as <k> grows.

Tc recalibrated per (q, m) — Tc rides <k>, so scan windows scale with m.
P grids scale with m to keep the knee in range. Writes data/ensemble_density.csv.
"""
from __future__ import annotations

import csv

import numpy as np
import networkx as nx

import dtm_col_infl_clamping as dtm

N = 196
M_GRID = (3, 6, 12, 24)
N_SEEDS = 3
FRAC = 0.10
RATIO = 2.0
TC_SAFETY = 1.15
KNEE = 0.7
P_BASE = {4: (2, 4, 6, 8, 12, 16), 8: (4, 8, 12, 16, 24, 32)}
T_BASE = {4: (0.6, 2.0), 8: (0.5, 1.9)}


def make_graph_m(m, seed):
    G = nx.barabasi_albert_graph(N, m=m, seed=seed)
    return nx.convert_node_labels_to_integers(G)


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


def locate_tc(G, q, m):
    lo, hi = T_BASE[q]
    scale = m / 3.0
    grid = np.linspace(lo * scale, hi * scale, 8)
    rows = [(T, spontaneous_potts_order(G, q, T)) for T in grid]
    crossing = next((T for T, mm in rows if mm < 0.5), grid[-1])
    return float(crossing)


def recall_m(G, q, P, T, seed):
    dtm.rng = np.random.default_rng(seed)
    return dtm.potts_recall(G, q, P, FRAC, T, strategy="degree",
                            ratio=RATIO, pattern_seed=1000 + seed)


def knee_from_curve(ps, ms):
    for k in range(1, len(ps)):
        if ms[k - 1] >= KNEE > ms[k]:
            p0, p1, m0, m1 = ps[k - 1], ps[k], ms[k - 1], ms[k]
            return p0 + (m0 - KNEE) * (p1 - p0) / (m0 - m1)
    if ms[0] < KNEE:
        return float(ps[0]) * (ms[0] / KNEE)
    return float(ps[-1])


def main():
    rows, knees = [], {}
    for q in P_BASE:
        for m in M_GRID:
            p_grid = tuple(int(round(p * m / 3)) for p in P_BASE[q])
            G0 = make_graph_m(m, seed=0)
            k_mean = 2 * G0.number_of_edges() / N
            tc = locate_tc(G0, q, m)
            T = round(TC_SAFETY * tc, 3)
            print(f"\nq={q}  m={m}  <k>={k_mean:.1f}  Tc~{tc:.2f} -> T={T}")
            print("  P    m_cued")
            mus = []
            for P in p_grid:
                mc = [recall_m(make_graph_m(m, seed=sd), q, P, T, sd)["m_cued"]
                      for sd in range(N_SEEDS)]
                mu, sd_ = float(np.mean(mc)), float(np.std(mc))
                mus.append(mu)
                rows.append(dict(q=q, m=m, k_mean=round(k_mean, 1), T=T, P=P,
                                 m_cued_mean=round(mu, 3),
                                 m_cued_std=round(sd_, 3)))
                print(f" {P:4d}   {mu:5.2f}+-{sd_:4.2f}")
            knee = knee_from_curve(p_grid, mus)
            knees[(q, m)] = (knee, k_mean)
            print(f"  knee P_{KNEE} ~ {knee:.1f}   (per connection: "
                  f"{knee / k_mean:.2f})")
            rows.append(dict(q=q, m=m, k_mean=round(k_mean, 1), T=T, P=-1,
                             m_cued_mean=round(knee, 2), m_cued_std=-1))

    print("\n=== complement test: knee and q8/q4 ratio vs density ===")
    print("  m   <k>    knee q=4   knee q=8   ratio   knee/k q=4  knee/k q=8")
    for m in M_GRID:
        k4, km = knees[(4, m)]
        k8, _ = knees[(8, m)]
        print(f" {m:3d}  {km:5.1f}   {k4:7.1f}   {k8:7.1f}   {k8 / max(k4, 1e-9):4.1f}x"
              f"     {k4 / km:5.2f}       {k8 / km:5.2f}")
    print("dense-theory ratio: 4.7x")

    path = "data/ensemble_density.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
