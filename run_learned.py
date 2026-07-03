"""H7: learned couplings vs written (Hebb) couplings — the trainable-intent rung.

Train an IsingEBM's edge weights on P stored patterns by maximum likelihood.
For a fully-visible model the positive phase is exact:
    <s_i s_j>_data = mean_mu t^mu_i t^mu_j
so only the negative (model) phase is sampled — with THRML block Gibbs
(`estimate_moments` on a free-running graph-colored program), which is what
`estimate_kl_grad` reduces to when no nodes are hidden. Ascent:
    w_ij += lr * (<s_i s_j>_data - <s_i s_j>_model)
Biases are frozen at 0: the Hebb rule has none, so this compares two ways of
WRITING couplings within the same model class.

Training runs at beta = 1/T_EVAL, so learned weights are in the same units
as Hebb weights and both are evaluated with the identical numpy Glauber
harness (dtm.sample_equilibrium(weights=...)) used for H5/H6.

H7a: at P=1, learned couplings percolate at least as cheaply as Hebb
     (fid vs carrier fraction, degree placement).
H7b: learning pushes the capacity knee to higher P (it can decorrelate
     interference; Hebb just superposes).
H7c: does learning restore r_struct's tracking domain at higher P?

Writes data/ensemble_learned.csv.
"""
from __future__ import annotations

import csv
import time

import jax
import jax.numpy as jnp
import numpy as np

from thrml import Block, SamplingSchedule, SpinNode
from thrml.models import IsingEBM, IsingSamplingProgram, estimate_moments

import dtm_col_infl_clamping as dtm
from carrier_pgm.model import greedy_coloring

N = 196
T_EVAL = 6.9              # 1.15 x Tc(scale_free), run_tc.py
RATIO = 2.0
FRAC = 0.10
P_GRID = (1, 2, 3, 5, 8, 12)
FRAC_GRID = (0.02, 0.06, 0.10, 0.16)   # H7a sweep at P=1
N_SEEDS = 3
LR = 0.3
N_STEPS = 150
SCHED = SamplingSchedule(n_warmup=40, n_samples=120, steps_per_sample=1)


def train_weights(G, patterns, seed):
    """Maximum-likelihood edge weights for the (fully visible) pattern set."""
    idx_edges = list(G.edges())
    nodes = [SpinNode() for _ in range(N)]
    edges = [(nodes[a], nodes[b]) for a, b in idx_edges]
    adj = {i: set() for i in range(N)}
    for a, b in idx_edges:
        adj[a].add(b)
        adj[b].add(a)
    groups = greedy_coloring(adj, list(range(N)))
    blocks = [Block([nodes[i] for i in g]) for g in groups]

    # exact positive phase (all nodes visible)
    m_data = jnp.array([float(np.mean(patterns[:, a] * patterns[:, b]))
                        for a, b in idx_edges])

    w = jnp.zeros(len(idx_edges))
    biases = jnp.zeros(N)
    beta = jnp.array(1.0 / T_EVAL)
    key = jax.random.key(10_000 + seed)

    t0 = time.time()
    for step in range(N_STEPS):
        key, k_init, k_mom = jax.random.split(key, 3)
        ebm = IsingEBM(nodes, edges, biases, w, beta)
        program = IsingSamplingProgram(ebm, blocks, [])
        init = [jax.random.bernoulli(k_init, 0.5, (len(b.nodes),))
                for b in blocks]
        _, m_model = estimate_moments(k_mom, [], edges, program, SCHED, init, [])
        w = w + LR * (m_data - m_model)
        if (step + 1) % 50 == 0:
            gap = float(jnp.mean(jnp.abs(m_data - m_model)))
            print(f"    step {step+1:3d}  moment gap {gap:.3f}  "
                  f"mean|w| {float(jnp.mean(jnp.abs(w))):.2f}  "
                  f"({time.time()-t0:.0f}s)")
    W = np.zeros((N, N))
    w_np = np.asarray(w)
    for e, (a, b) in enumerate(idx_edges):
        W[a, b] = W[b, a] = w_np[e]
    return W


def recall(G, W, pats, frac, strategy="degree"):
    """Same protocol as multi_pattern_recall, with externally supplied W."""
    cue = pats[0]
    clamp_idx = dtm.place_carriers(G, frac, strategy=strategy)
    B = RATIO * T_EVAL
    s, free_idx, trace = dtm.sample_equilibrium(
        G, cue, clamp_idx, J=1.0, B=B, T=T_EVAL, burn=40, record=60, weights=W)
    m = dtm.metrics(s, free_idx, cue, trace)
    others = [abs(float((s[free_idx] * pats[mu][free_idx]).mean()))
              for mu in range(1, len(pats))]
    m["fid_other"] = max(others) if others else 0.0
    m["r_struct"] = float(np.mean([np.sign(W[i, j]) * s[i] * s[j]
                                   for i, j in G.edges()]))
    return m


def main():
    rows = []
    # -- H7b/H7c: capacity sweep, learned vs Hebb, frac fixed ------------
    print(f"capacity sweep (frac={FRAC}, degree, {N_SEEDS} seeds)")
    print("  P    fid learned    fid hebb      r_struct lrn  r_struct hebb")
    for P in P_GRID:
        cells = {k: [] for k in ("fl", "fh", "rl", "rh")}
        for seed in range(N_SEEDS):
            G = dtm.make_graph("scale_free", N, seed=seed)
            pats = np.random.default_rng(1000 + seed).choice(
                [-1, 1], size=(P, N)).astype(int)
            print(f"  training P={P} seed={seed} ...")
            W = train_weights(G, pats, seed)
            dtm.rng = np.random.default_rng(seed)
            ml = recall(G, W, pats, FRAC)
            dtm.rng = np.random.default_rng(seed)
            mh = dtm.multi_pattern_recall(G, P, FRAC, strategy="degree",
                                          ratio=RATIO, T=T_EVAL,
                                          pattern_seed=1000 + seed)
            cells["fl"].append(ml["fid"]); cells["fh"].append(mh["fid"])
            cells["rl"].append(ml["r_struct"]); cells["rh"].append(mh["r_struct"])
            if P == 1:                       # reuse P=1 training for H7a
                for frac in FRAC_GRID:
                    dtm.rng = np.random.default_rng(seed)
                    a = recall(G, W, pats, frac)
                    dtm.rng = np.random.default_rng(seed)
                    b = dtm.multi_pattern_recall(
                        G, 1, frac, strategy="degree", ratio=RATIO,
                        T=T_EVAL, pattern_seed=1000 + seed)
                    rows.append(dict(kind="h7a", P=1, frac=frac, seed=seed,
                                     fid_learned=round(a["fid"], 3),
                                     fid_hebb=round(b["fid"], 3)))
        s = {k: (float(np.mean(v)), float(np.std(v))) for k, v in cells.items()}
        rows.append(dict(kind="h7b", P=P, frac=FRAC, seed=-1,
                         **{f"{n}_mean": round(s[k][0], 3)
                            for k, n in (("fl", "fid_learned"), ("fh", "fid_hebb"),
                                         ("rl", "rs_learned"), ("rh", "rs_hebb"))},
                         **{f"{n}_std": round(s[k][1], 3)
                            for k, n in (("fl", "fid_learned"), ("fh", "fid_hebb"),
                                         ("rl", "rs_learned"), ("rh", "rs_hebb"))}))
        print(f" {P:3d}   {s['fl'][0]:5.2f}+-{s['fl'][1]:4.2f}   "
              f"{s['fh'][0]:5.2f}+-{s['fh'][1]:4.2f}    "
              f"{s['rl'][0]:5.2f}+-{s['rl'][1]:4.2f}   "
              f"{s['rh'][0]:5.2f}+-{s['rh'][1]:4.2f}")

    print("\nH7a: P=1 fraction sweep (degree placement)")
    print("frac    fid learned (per seed)    fid hebb (per seed)")
    for frac in FRAC_GRID:
        a = [r["fid_learned"] for r in rows if r["kind"] == "h7a" and r["frac"] == frac]
        b = [r["fid_hebb"] for r in rows if r["kind"] == "h7a" and r["frac"] == frac]
        print(f"{frac:.2f}    {np.mean(a):5.2f}+-{np.std(a):4.2f}"
              f"                {np.mean(b):5.2f}+-{np.std(b):4.2f}")

    path = "data/ensemble_learned.csv"
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
