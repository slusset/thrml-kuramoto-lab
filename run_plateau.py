"""Anatomy of the random-placement Hopfield recall plateau (~0.66, frac 0.10-0.16).

Four probes, scale-free N=196 at the calibrated T=6.9 (1.15*Tc), B*J/T=2.0:

1. GAUGE CHECK — sigma_i = s_i*t_i maps the single-pattern Hopfield model onto
   the ferromagnet exactly, so Hopfield/random must reproduce ferro/random at
   equal fraction and seed. If it does, "pattern interference" is ruled out by
   construction and the plateau is a fact about random placement, not memory.
2. DENSE GRID — 10 seeds x fractions 0.08-0.24: real plateau or slow climb?
3. HORIZON — burn 40 vs 160 vs 640 at frac 0.16: kinetic (fid climbs with
   time => domains still growing) or equilibrium (flat => structural)?
4. DOMAIN ANATOMY — for each stuck configuration look at the wrong-sign free
   nodes in the gauge picture: largest connected wrong-cluster, mean degree,
   and hop-distance to the nearest carrier, wrong vs right. Distinguishes
   "anchored wrong domains" from "uncovered low-degree periphery".
"""
from __future__ import annotations

import numpy as np
import networkx as nx

import dtm_col_infl_clamping as dtm

N = 196
T = 6.9                    # 1.15 x Tc(scale_free) from run_tc.py
RATIO = 2.0
B = RATIO * T / 1.0


def run(seed, frac, strategy="random", hopfield=True, burn=40, record=60):
    dtm.rng = np.random.default_rng(seed)
    G = dtm.make_graph("scale_free", N, seed=seed)
    n = G.number_of_nodes()
    if hopfield:
        t = np.random.default_rng(1000 + seed).choice([-1, 1], size=n).astype(int)
        pattern = t
    else:
        t = np.ones(n, dtype=int)
        pattern = None
    clamp_idx = dtm.place_carriers(G, frac, strategy=strategy)
    s, free_idx, trace = dtm.sample_equilibrium(
        G, t, clamp_idx, J=1.0, B=B, T=T, burn=burn, record=record,
        pattern=pattern)
    fid = dtm.metrics(s, free_idx, t, trace)["fid"]
    return G, t, s, clamp_idx, free_idx, fid


def anatomy(G, t, s, clamp_idx, free_idx):
    """Wrong-vs-right structure of the free nodes, in the gauge picture."""
    sigma = s * t
    free = set(free_idx.tolist())
    wrong = [i for i in free_idx if sigma[i] < 0]
    right = [i for i in free_idx if sigma[i] > 0]
    deg = dict(G.degree())
    dist = nx.multi_source_dijkstra_path_length(G, set(clamp_idx.tolist()))
    wrong_sub = G.subgraph(wrong)
    comps = sorted((len(c) for c in nx.connected_components(wrong_sub)),
                   reverse=True) or [0]
    d = lambda nodes: float(np.mean([deg[i] for i in nodes])) if nodes else 0.0
    r = lambda nodes: float(np.mean([dist.get(i, np.inf) for i in nodes])) if nodes else 0.0
    return dict(n_wrong=len(wrong), largest_wrong_cluster=comps[0],
                n_wrong_clusters=len([c for c in comps if c > 0]),
                deg_wrong=d(wrong), deg_right=d(right),
                dist_wrong=r(wrong), dist_right=r(right))


def main():
    print("=== 1. gauge check (hopfield vs ferromagnet, same seed/frac) ===")
    print("frac   seed   fid_hopfield   fid_ferro")
    for frac in (0.10, 0.16):
        for seed in range(5):
            *_, fh = run(seed, frac, hopfield=True)
            *_, ff = run(seed, frac, hopfield=False)
            print(f"{frac:.2f}   {seed}      {fh:6.3f}        {ff:6.3f}")

    print("\n=== 2. dense fraction grid, random placement, 10 seeds ===")
    fracs = (0.08, 0.12, 0.16, 0.20, 0.24)
    print("frac    fid (mean+-std)      [degree ref]")
    for frac in fracs:
        fids = [run(s, frac, "random")[-1] for s in range(10)]
        dref = [run(s, frac, "degree")[-1] for s in range(3)]
        print(f"{frac:.2f}    {np.mean(fids):5.2f}+-{np.std(fids):4.2f}"
              f"          {np.mean(dref):5.2f}+-{np.std(dref):4.2f}")

    print("\n=== 3. horizon test, frac=0.16 random, 5 seeds ===")
    print("burn    fid (mean+-std)")
    for burn in (40, 160, 640):
        fids = [run(s, 0.16, "random", burn=burn, record=30)[-1]
                for s in range(5)]
        print(f"{burn:4d}    {np.mean(fids):5.2f}+-{np.std(fids):4.2f}")

    print("\n=== 4. domain anatomy, frac=0.16 random, 5 seeds ===")
    print("seed  fid   n_wrong  biggest  #clusters  deg w/r      dist w/r")
    for seed in range(5):
        G, t, s, clamp_idx, free_idx, fid = run(seed, 0.16, "random")
        a = anatomy(G, t, s, clamp_idx, free_idx)
        print(f"{seed}    {fid:5.2f}  {a['n_wrong']:5d}    {a['largest_wrong_cluster']:5d}"
              f"    {a['n_wrong_clusters']:5d}   {a['deg_wrong']:4.1f}/{a['deg_right']:4.1f}"
              f"   {a['dist_wrong']:4.2f}/{a['dist_right']:4.2f}")


if __name__ == "__main__":
    main()
