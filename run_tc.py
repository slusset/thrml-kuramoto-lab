"""Locate a finite-graph ordering crossover for each topology.

Phase 1 — for each topology, run the Glauber field with ZERO clamps across a
temperature grid and measure spontaneous order:
    <|m|>  — magnetization (order parameter)
    chi    — susceptibility N*(<m^2> - <|m|>^2)/T  (peaks near Tc)
The operating crossover is estimated from the |m|=0.5 crossing; susceptibility
is printed as a diagnostic. This is a pseudocritical calibration for the finite
graphs, not a thermodynamic critical-temperature estimate. Carrier experiments
must operate above it if the intended regime is ordering from disorder.

Phase 2 — rerun the placement sweep (random/degree/ci) at T = TC_SAFETY * Tc
per topology, fixed B*J/T, so fidelity is carrier-driven and placements can
actually separate.
"""
from __future__ import annotations

import numpy as np

import dtm_col_infl_clamping as dtm

TC_SAFETY = 1.15          # run carrier experiments at this multiple of Tc
BURN, RECORD = 80, 80

T_GRIDS = {
    "lattice":     np.linspace(1.5, 4.0, 11),
    "small_world": np.linspace(2.0, 7.0, 11),
    "scale_free":  np.linspace(3.0, 12.0, 10),
}


def spontaneous_order(G, T, burn=BURN, record=RECORD):
    """<|m|> and susceptibility of the unclamped field at temperature T."""
    N = G.number_of_nodes()
    adj = [list(G.neighbors(i)) for i in range(N)]
    clamp_mask = np.zeros(N, dtype=bool)
    free_idx = np.arange(N)
    s = dtm.rng.choice([-1, 1], size=N).astype(int)
    s = dtm.glauber_sweep(s, adj, clamp_mask, free_idx, J=1.0, B=1.0, T=T, n_sweeps=burn)
    ms = []
    for _ in range(record):
        s = dtm.glauber_sweep(s, adj, clamp_mask, free_idx, J=1.0, B=1.0, T=T, n_sweeps=1)
        ms.append(s.mean())
    ms = np.array(ms)
    m_abs = np.abs(ms).mean()
    chi = N * (np.mean(ms**2) - m_abs**2) / T
    return m_abs, chi


def locate_tc(N=196):
    tc = {}
    for topo, grid in T_GRIDS.items():
        G = dtm.make_graph(topo, N, seed=1)
        rows = [(T, *spontaneous_order(G, T)) for T in grid]
        chi_peak_T = max(rows, key=lambda r: r[2])[0]
        crossing = next((T for T, m, _ in rows if m < 0.5), grid[-1])
        # The |m|=0.5 crossing is the primary estimate: at N~200 single-seed,
        # the chi peak is noisy (it put the lattice at 1.75 vs the exact 2.27;
        # the crossing gave 2.50). chi is printed as a diagnostic only.
        tc[topo] = crossing
        print(f"\n{topo}  (mean degree {2*G.number_of_edges()/N:.1f})")
        print("   T      <|m|>    chi")
        for T, m, chi in rows:
            print(f"  {T:5.2f}   {m:5.3f}   {chi:7.2f}")
        print(f"  ordering crossover: |m| crosses 0.5 at T={crossing:.2f}"
              f"   (chi peak at T={chi_peak_T:.2f}, diagnostic)")
    return tc


def calibrated_sweep(tc, N=196, ratio=2.0):
    """Placement sweep with each topology sitting just above its own Tc."""
    fractions = tuple(np.round(np.linspace(0.01, 0.20, 8), 3))
    print(f"\n=== calibrated placement sweep (T = {TC_SAFETY} x Tc, B*J/T = {ratio}) ===")
    for topo, Tc in tc.items():
        T = round(TC_SAFETY * Tc, 2)
        B = ratio * T / 1.0
        G = dtm.make_graph(topo, N, seed=1)
        target = np.ones(G.number_of_nodes(), dtype=int)
        print(f"\n{topo}  Tc={Tc:.2f}  ->  T={T}   (fid by carrier fraction)")
        print("frac:    " + "  ".join(f"{f:5.3f}" for f in fractions))
        for strat in ("random", "degree", "ci"):
            row = []
            for frac in fractions:
                clamp_idx = dtm.place_carriers(G, frac, strategy=strat)
                s, free_idx, trace = dtm.sample_equilibrium(
                    G, target, clamp_idx, J=1.0, B=B, T=T, burn=40, record=60)
                row.append(dtm.metrics(s, free_idx, target, trace)["fid"])
            print(f"{strat:8s} " + "  ".join(f"{v:5.2f}" for v in row))


if __name__ == "__main__":
    tc = locate_tc()
    print("\nfinite-graph ordering crossovers:", {k: round(v, 2) for k, v in tc.items()})
    calibrated_sweep(tc)
