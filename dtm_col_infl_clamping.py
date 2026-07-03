# %% [markdown]
# # Collective-Influence Clamping inside a Denoising Thermodynamic Step
#
# **What this notebook tests:** whether placing a *sparse* set of clamped
# ("carrier") nodes by **collective influence** (Morone–Makse) lowers the
# carrier-fraction threshold for coherence — and whether **staging** that
# clamping across a chain of shallow steps (the DTM / denoising move) beats a
# single monolithic landscape. It is the union of two things we already have:
#
# - **Yesterday (THRML):** ~4% carrier-fraction percolation threshold; the
#   `B·J` term (a clamp exerts an effective field ~`B·J` on its neighbors);
#   clamp *placement* matters by neighbor structure. Control parameter is the
#   dimensionless ratio `B·J / T`.
# - **The paper (Extropic DTM/DTCA, npj Unconv. Computing):** the
#   mixing–expressivity tradeoff (one deep landscape is un-samplable), escaped
#   by *chaining shallow, well-mixed steps*; the Adaptive Correlation Penalty
#   (a homeostatic loop that watches lag-autocorrelation `r_yy` and holds the
#   sampler in the mixable regime).
#
# ---
#
# ## Framing: three nested contrasts (read before running)
#
# ### 1. Deterministic vs. dissipative  —  *ATOM vs. us*
# `mprahboamey/atom` (optical attention in holographic crystals) and this work
# converge on the **meta-thesis** ("the weights are the computer") but sit on
# opposite sides of the thermodynamic axis. We are firmly on the right column.
#
# | axis            | ATOM (optical attention)      | this notebook (thermodynamic)     |
# |-----------------|-------------------------------|-----------------------------------|
# | dynamics        | ballistic, single-shot pass   | relaxational, settles over time   |
# | thermodynamics  | conservative, **reversible**  | dissipative, **irreversible**     |
# | what it computes| a deterministic linear map    | a **distribution** by equilibration|
# | role of noise   | the **enemy** (no noise model)| the **substrate** (compute medium)|
# | layer           | one operator (attention head) | the **field** (many coupled units)|
#
# The optical cousin of *our* substrate is not ATOM; it is the **coherent Ising
# machine**. ATOM is a candidate *organ* inside the mesh, not a competitor.
#
# ### 2. Monolithic field vs. staged aggregated fields  —  *the MET escape*
# Do not ask one landscape to carry the whole telos. Stage it across small,
# well-mixed sub-fields that align. This is the *same structural move* as the
# guild / carrier-cluster result (spatial) and the denoising chain (temporal):
# **telos is composed from small aligned fields, never from one monolith.**
#
# ### 3. Coherence-as-order-parameter vs. coherence-as-fidelity  —  *the caveat*
# We measure BOTH, separately, because they diverge:
# - **homeostatic coherence** `r_mag` = |mean(s)| — the system agreeing with *itself*
# - **teleological fidelity** `fid` = mean(s · target) — agreeing with the *right thing*
#
# The thermostat-burning-the-house failure: `r_mag → 1` while `fid` stays low
# (locked onto the wrong target). Aligning small fields multiplies whatever the
# setpoints point at — it needs a *true sign* to multiply.

# %%
from __future__ import annotations
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

rng = np.random.default_rng(7)

# %% [markdown]
# ## 0. The sampler seam
#
# The cell below is a minimal, self-contained **Glauber (Ising) sampler** so the
# whole notebook runs today with zero external deps. It reproduces yesterday's
# setup exactly: clamped nodes are fixed to the target; each clamp exerts an
# effective field `B·J` on its free neighbors (interior edges carry `J`);
# temperature `T` enters the acceptance. The dimensionless knob is `B·J / T`.
#
# **THRML swap-in:** replace `glauber_sweep` with your THRML block-Gibbs step,
# feeding clamped nodes as the boundary and reading the settled configuration.
# The metrics and sweep harness below are sampler-agnostic.

# %%
def _neighbor_fields(s, adj, clamp_mask, J, B):
    """Local field on every node: interior edges = J, clamp-incident edges = B*J."""
    h = np.zeros_like(s, dtype=float)
    w_clamped = B * J
    for i, nbrs in enumerate(adj):
        if not nbrs:
            continue
        contrib = 0.0
        for j in nbrs:
            w = w_clamped if clamp_mask[j] else J
            contrib += w * s[j]
        h[i] = contrib
    return h


def glauber_sweep(s, adj, clamp_mask, free_idx, J, B, T, n_sweeps, pattern=None,
                  weights=None):
    """In-place-ish Glauber dynamics over free nodes; clamped nodes held fixed.

    `pattern` switches on Hopfield couplings J_ij = J * t_i * t_j (edges that
    should agree stay ferromagnetic, edges that should differ become
    anti-ferromagnetic), so `pattern` is an energy minimum. None = ferromagnet.

    `weights` (NxN array, takes precedence) allows arbitrary per-edge
    couplings — used for multi-pattern Hebb sums, where J_ij is no longer
    +-J and the pattern shortcut can't express it. Clamp-incident edges are
    amplified by B as before.
    """
    s = s.copy()
    for _ in range(n_sweeps):
        order = rng.permutation(free_idx)
        for i in order:
            h = 0.0
            for j in adj[i]:
                if weights is not None:
                    w = weights[i, j] * (B if clamp_mask[j] else 1.0)
                else:
                    w = (B * J) if clamp_mask[j] else J
                    if pattern is not None:
                        w *= pattern[i] * pattern[j]
                h += w * s[j]
            p_up = 1.0 / (1.0 + np.exp(-2.0 * h / max(T, 1e-9)))
            s[i] = 1 if rng.random() < p_up else -1
    return s


def sample_equilibrium(G, target, clamp_idx, J=1.0, B=1.0, T=1.0,
                       burn=40, record=60, pattern=None, weights=None):
    """Settle the field with `clamp_idx` fixed to target; return final state and
    a magnetization trace (for the r_yy mixing sensor)."""
    N = G.number_of_nodes()
    adj = [list(G.neighbors(i)) for i in range(N)]
    clamp_mask = np.zeros(N, dtype=bool)
    clamp_mask[clamp_idx] = True
    s = rng.choice([-1, 1], size=N).astype(int)
    s[clamp_idx] = target[clamp_idx]                 # boundary = the program
    free_idx = np.array([i for i in range(N) if not clamp_mask[i]])

    s = glauber_sweep(s, adj, clamp_mask, free_idx, J, B, T, burn,
                      pattern=pattern, weights=weights)   # burn-in
    trace = []
    for _ in range(record):
        s = glauber_sweep(s, adj, clamp_mask, free_idx, J, B, T, 1,
                          pattern=pattern, weights=weights)
        s[clamp_idx] = target[clamp_idx]
        trace.append(s[free_idx].mean())
    return s, free_idx, np.array(trace)

# %% [markdown]
# ## 1. Topologies
# Sweep placement across lattice / small-world / scale-free so any threshold can
# be checked against known percolation behavior per topology (yesterday's plan).

# %%
def make_graph(kind, N, seed=0):
    if kind == "lattice":
        side = int(round(np.sqrt(N)))
        G = nx.grid_2d_graph(side, side, periodic=True)
        G = nx.convert_node_labels_to_integers(G)
    elif kind == "small_world":
        G = nx.watts_strogatz_graph(N, k=6, p=0.1, seed=seed)
    elif kind == "scale_free":
        G = nx.barabasi_albert_graph(N, m=3, seed=seed)
    else:
        raise ValueError(kind)
    return nx.convert_node_labels_to_integers(G)

# %% [markdown]
# ## 2. Placement strategies
# `random`, `degree` (highest-degree), `ci` (collective influence, radius ℓ=2).
# CI_ℓ(i) = (k_i − 1) · Σ_{j on the ℓ-frontier of i} (k_j − 1). The static
# top-m version is used here; the rigorous Morone–Makse variant removes the max
# and recomputes (adaptive) — a TODO once the static result is in.

# %%
def collective_influence(G, ell=2):
    deg = dict(G.degree())
    ci = {}
    for i in G.nodes():
        dist = nx.single_source_shortest_path_length(G, i, cutoff=ell)
        frontier = [j for j, d in dist.items() if d == ell]
        ci[i] = (deg[i] - 1) * sum(deg[j] - 1 for j in frontier)
    return ci


def place_carriers(G, fraction, strategy="ci", ell=2):
    N = G.number_of_nodes()
    m = max(1, int(round(fraction * N)))
    if strategy == "random":
        return rng.choice(N, size=m, replace=False)
    if strategy == "degree":
        order = sorted(G.nodes(), key=lambda i: G.degree(i), reverse=True)
        return np.array(order[:m])
    if strategy == "ci":
        ci = collective_influence(G, ell=ell)
        order = sorted(G.nodes(), key=lambda i: ci[i], reverse=True)
        return np.array(order[:m])
    raise ValueError(strategy)

# %% [markdown]
# ## 3. The two coherence metrics + the mixing sensor
# - `r_mag`  — homeostatic coherence (|magnetization| over free nodes)
# - `fid`    — teleological fidelity (overlap of free nodes with the target)
# - `r_yy`   — lag-1 autocorrelation of the magnetization trace = the ACP
#              homeostasis sensor. High `r_yy` ⇒ sluggish mixing ⇒ untrustworthy.

# %%
def metrics(s, free_idx, target, trace):
    r_mag = abs(s[free_idx].mean())
    fid = float((s[free_idx] * target[free_idx]).mean())   # in [-1, 1]
    x = trace - trace.mean()
    denom = (x * x).sum()
    r_yy = float((x[:-1] * x[1:]).sum() / denom) if denom > 1e-12 else 0.0
    return dict(r_mag=r_mag, fid=fid, r_yy=r_yy)

# %% [markdown]
# ### Modeling notes (read once — these are the knobs to get right in THRML)
# - **Symmetry breaking.** With `J>0` below `Tc` the field orders *spontaneously*,
#   which would masquerade as a carrier effect. Default `T=2.5` keeps the interior
#   in the disordered phase (2D lattice `Tc≈2.27 J`) so ordering is *carrier-driven*.
#   Small-world / scale-free have higher effective `Tc` (denser) — locate each
#   topology's critical point first; that is step 0 of any real run.
# - **`fid` vs `r_mag` in the coherent-target model.** With `target = all +1`,
#   fidelity and magnetization coincide in the ordered phase *by construction* —
#   so the main sweep tests percolation (H1–H3), and the **fid≠r_mag contrast
#   lives in the H4 control** (coherent-but-wrong broadcast). Keep them apart.
# - **Arbitrary targets ⇒ Hopfield.** To recall a structured (non-uniform)
#   pattern, set edge couplings `J_ij = target_i · target_j` (associative memory)
#   so the pattern is an energy minimum. Then `fid` and `r_mag` genuinely diverge
#   without needing the wrong-target trick. TODO cell once the uniform case is in.
#
# ## 4. Monolithic sweep — carrier fraction × (B·J / T) × placement × topology
# The core measurement. For each cell we record fidelity, homeostatic coherence,
# and the mixing sensor. Collapse the fidelity threshold curves against `B·J/T`
# to recover the master curve (the "4%" is one slice through it).

# %%
@dataclass
class Sweep:
    N: int = 256
    topologies: tuple = ("lattice", "small_world", "scale_free")
    strategies: tuple = ("random", "degree", "ci")
    fractions: tuple = tuple(np.round(np.linspace(0.01, 0.20, 8), 3))
    BJ_over_T: tuple = (0.5, 1.0, 2.0)     # dimensionless control
    J: float = 1.0
    T: float = 2.5     # sit near/above the interior Tc so order is carrier-DRIVEN,
                       # not spontaneous symmetry breaking (2D lattice Tc≈2.27 J)
    burn: int = 40
    record: int = 60
    results: list = field(default_factory=list)


def run_monolithic(cfg: Sweep):
    cfg.results.clear()
    for topo in cfg.topologies:
        G = make_graph(topo, cfg.N, seed=1)
        N = G.number_of_nodes()
        # Coherent target = the all-aligned state (matches yesterday's Kuramoto
        # phase-lock). With a ferromagnet (J>0) this is a genuine attractor, so
        # `fid` rising with carrier fraction measures percolation-of-recall.
        # (For arbitrary targets, swap to a Hopfield coupling — see modeling notes.)
        target = np.ones(N, dtype=int)                     # the "true sign"
        for strat in cfg.strategies:
            for frac in cfg.fractions:
                clamp_idx = place_carriers(G, frac, strategy=strat)
                for ratio in cfg.BJ_over_T:
                    B = ratio * cfg.T / cfg.J                # B·J/T = ratio
                    s, free_idx, trace = sample_equilibrium(
                        G, target, clamp_idx, J=cfg.J, B=B, T=cfg.T,
                        burn=cfg.burn, record=cfg.record)
                    m = metrics(s, free_idx, target, trace)
                    cfg.results.append(dict(topo=topo, strat=strat,
                                            frac=frac, ratio=ratio, **m))
    return cfg.results

# %% [markdown]
# ## 5. Staged clamping — the MET escape (proxy for the DTM denoising chain)
# Instead of one settle, run `K` shallow steps. Each step re-places a small
# carrier set by CI and carries the free-node state forward. We compare the
# **total clamp budget** and end-to-end fidelity against the monolithic run.
#
# > **Proxy warning.** This is *re-place-and-carry*, not a learned forward/
# > reverse diffusion process. Replace with the paper's denoising steps (each a
# > shallow EBM conditioned on the previous, noised toward the target) for the
# > real DTM. The point of the proxy is to see whether staging lowers the
# > per-step carrier budget while `r_yy` stays low at every step.

# %%
def run_staged(G, target, K=4, frac_per_step=0.02, strategy="ci",
               J=1.0, B=2.0, T=1.0, sweeps_per_step=15):
    N = G.number_of_nodes()
    adj = [list(G.neighbors(i)) for i in range(N)]
    s = rng.choice([-1, 1], size=N).astype(int)
    per_step_ryy = []
    for _ in range(K):
        clamp_idx = place_carriers(G, frac_per_step, strategy=strategy)
        clamp_mask = np.zeros(N, dtype=bool); clamp_mask[clamp_idx] = True
        free_idx = np.array([i for i in range(N) if not clamp_mask[i]])
        s[clamp_idx] = target[clamp_idx]
        trace = []
        for _ in range(sweeps_per_step):
            s = glauber_sweep(s, adj, clamp_mask, free_idx, J, B, T, 1)
            s[clamp_idx] = target[clamp_idx]
            trace.append(s[free_idx].mean())
        trace = np.array(trace); x = trace - trace.mean()
        d = (x * x).sum()
        per_step_ryy.append(float((x[:-1] * x[1:]).sum() / d) if d > 1e-12 else 0.0)
    free_all = np.arange(N)
    fid = float((s * target).mean())
    total_budget = K * frac_per_step
    return dict(fid=fid, total_budget=total_budget,
                mean_ryy=float(np.mean(per_step_ryy)), per_step_ryy=per_step_ryy)

# %% [markdown]
# ## 6. Hypotheses & falsification (state before looking at plots)
#
# **H1 (placement).** CI placement reaches `fid → 1` at a strictly lower carrier
# fraction than degree, and degree lower than random, on small-world and
# scale-free. *Falsified if* CI ≈ degree ≈ random (placement doesn't matter →
# yesterday's neighbor finding was a lattice artifact).
#
# **H2 (collapse).** Fidelity-threshold curves for a fixed topology collapse onto
# one master curve when plotted against `B·J/T`. *Falsified if* the curves stay
# separated → the threshold is not governed by the dimensionless ratio alone.
#
# **H3 (staging).** Staged CI clamping reaches equal-or-higher `fid` at lower
# *total* budget than monolithic, with per-step `r_yy` staying low. *Falsified if*
# staging needs ≥ the monolithic budget, or `r_yy` climbs across steps
# (the MET escape doesn't hold in this proxy).
#
# **H4 (the caveat, control).** Point clamps at a *wrong* target: `r_mag` should
# still rise (system self-locks) while `fid` stays ~0. *Falsified if* `fid` rises
# without correct clamps → the metric isn't separating the two coherences.

# %%
# ---- quick visualization helper (fidelity vs fraction, per strategy) ----
def plot_thresholds(results, topo, ratio):
    plt.figure(figsize=(6, 4))
    for strat in sorted({r["strat"] for r in results}):
        pts = [(r["frac"], r["fid"]) for r in results
               if r["topo"] == topo and r["strat"] == strat and r["ratio"] == ratio]
        pts.sort()
        if pts:
            xs, ys = zip(*pts)
            plt.plot(xs, ys, marker="o", label=strat)
    plt.axhline(0.9, ls="--", lw=0.8, color="gray")
    plt.xlabel("carrier fraction"); plt.ylabel("teleological fidelity")
    plt.title(f"{topo}   B·J/T = {ratio}"); plt.legend(); plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 7. Hopfield: structured targets — intent in the couplings
#
# The ferromagnet has exactly two valleys (all-up, all-down), so the only
# intent it can hold is *uniformity* — and with a uniform target, fidelity and
# magnetization coincide by construction (the H4 wrong-target trick was needed
# to pry them apart). Hopfield's move: **sculpt the valley**. For any pattern
# `t`, set `J_ij = J·t_i·t_j` and `t` becomes an energy minimum. Carriers then
# clamp a few nodes of the pattern and relaxation *completes the memory* —
# associative recall, the percolation question upgraded from "can carriers
# make everyone conform?" to "can carriers make the mesh remember?"
#
# Gauge note: `s_i → s_i·t_i` maps the Hopfield model back to the ferromagnet,
# so the single-pattern Tc and threshold physics carry over from step 0
# (`run_tc.py`) unchanged. What does NOT carry over is what the monitors see:
#
# - `r_mag` (magnetization) is **blind** to structured order — a balanced
#   pattern recalled perfectly has `mean(s) ≈ 0`.
# - `r_struct` = mean over edges of `sign(J_ij)·s_i·s_j` — *satisfaction of
#   the local wiring* — is target-blind (each node needs only its own edge
#   signs) and DOES track recall. The honest homeostatic monitor for
#   structured intent is coupling-satisfaction, not consensus.
#
# **H5 (structured recall).** With Hopfield couplings on scale-free at the
# calibrated T, `fid → 1` at a sparse carrier fraction (degree placement)
# while `r_mag` stays ≈ 0 and `r_struct` tracks `fid`. *Falsified if* fid
# fails to rise (the pattern is not an attractor — implementation wrong) or
# r_mag tracks fid (the metric split is illusory).

# %%
def hopfield_recall(G, frac, strategy="degree", J=1.0, ratio=2.0, T=6.9,
                    burn=40, record=60, pattern_seed=11):
    """Clamp `frac` carriers to a stored random pattern; measure recall."""
    N = G.number_of_nodes()
    t = np.random.default_rng(pattern_seed).choice([-1, 1], size=N).astype(int)
    clamp_idx = place_carriers(G, frac, strategy=strategy)
    B = ratio * T / J
    s, free_idx, trace = sample_equilibrium(G, t, clamp_idx, J=J, B=B, T=T,
                                            burn=burn, record=record, pattern=t)
    m = metrics(s, free_idx, t, trace)
    m["r_struct"] = float(np.mean([t[i] * t[j] * s[i] * s[j]
                                   for i, j in G.edges()]))
    return m

# %% [markdown]
# ## 8. Multi-pattern capacity — when memories compete
#
# Store several patterns at once with the Hebb rule, `J_ij = Σ_μ t^μ_i·t^μ_j`
# (unnormalized: each new covenant is laid on top of the existing ones; the
# alternative — normalizing by P — trades interference for weakened signal.
# Flag: this choice matters and is made explicitly). Each stored pattern is a
# valley; they also interfere — for the cued pattern, the other P−1 contribute
# random noise of scale √(P−1) per edge. Dense-network theory says capacity
# α_c ≈ 0.138·(connections per node); on a sparse graph (⟨k⟩ ≈ 6 here) that
# predicts **a handful of patterns at most**. Sparse meshes are cheap to
# steer but poor libraries.
#
# The gauge transform that made single-pattern Hopfield "just the ferromagnet"
# does NOT survive superposition — this is genuinely new physics for the lab.
# We measure at the calibrated single-pattern operating point (T, B·J/T fixed),
# so what we find is *operating capacity at the steerable point*, not the
# zero-temperature theoretical maximum.
#
# **H6 (capacity).** Cued recall (carriers clamp fragments of pattern 0)
# degrades as P grows, with a knee at small P. Degree placement sustains
# recall to higher P than random. *Falsified if* recall is P-independent
# (interference doesn't bite at this size) or random ≥ degree.
#
# **H6b (the monitor prediction — charter-relevant).** At capacity collapse,
# `r_struct` stays elevated while cued fidelity drops: the mesh settles into
# *some* stored (or mixture) state, honoring its couplings, just not the one
# that was cued. Coupling-satisfaction certifies "serving a stored intent,"
# not "serving the intent you asked for." *Falsified if* r_struct tracks
# fid_cued down — then the monitor is stronger than we fear.

# %%
def build_hopfield_weights(G, patterns):
    """Hebb rule on the graph's edges: W[i,j] = Σ_μ t^μ_i · t^μ_j."""
    N = G.number_of_nodes()
    W = np.zeros((N, N))
    for i, j in G.edges():
        w = float(np.sum(patterns[:, i] * patterns[:, j]))
        W[i, j] = W[j, i] = w
    return W


def multi_pattern_recall(G, P, frac, strategy="degree", ratio=2.0, T=6.9,
                         burn=40, record=60, pattern_seed=11):
    """Store P patterns, cue pattern 0 via carriers; measure who gets served."""
    N = G.number_of_nodes()
    pats = np.random.default_rng(pattern_seed).choice(
        [-1, 1], size=(P, N)).astype(int)
    W = build_hopfield_weights(G, pats)
    cue = pats[0]
    clamp_idx = place_carriers(G, frac, strategy=strategy)
    B = ratio * T / 1.0
    s, free_idx, trace = sample_equilibrium(G, cue, clamp_idx, J=1.0, B=B, T=T,
                                            burn=burn, record=record, weights=W)
    m = metrics(s, free_idx, cue, trace)
    others = [abs(float((s[free_idx] * pats[mu][free_idx]).mean()))
              for mu in range(1, P)]
    m["fid_other"] = max(others) if others else 0.0   # best NON-cued overlap
    m["r_struct"] = float(np.mean([np.sign(W[i, j]) * s[i] * s[j]
                                   for i, j in G.edges()]))
    return m

# %% [markdown]
# ## 9. Clock phases — the dial between Ising and Kuramoto
#
# The q-state clock model puts each node's state on a circle,
# `θ_a = 2πa/q`, with coupling `J·cos(θ_i − θ_j)` per edge. `q=2` IS the
# Ising model (regression gate); `q→∞` approaches the XY model — Kuramoto's
# phase space. This is the dial for finding 6's open question: the Gibbs
# substrate punished degree heterogeneity ~8× harder than Kuramoto, and the
# suspected mechanism was that binary spins can't *partially* align — an
# unclamped hub anchors the old phase absolutely, while soft phases let
# intent leak through it gradually.
#
# **H8a (heterogeneity penalty).** At matched operating points (1.15·Tc(q),
# located per q — Tc moves as states are added), the random-vs-degree
# placement gap on scale-free shrinks monotonically as q grows: partial
# alignment extends the influence radius, so coverage matters less.
# *Falsified if* the gap is q-independent (the 8× was never about softness)
# or grows.
#
# Metrics generalize gradedly: `fid` = mean cos(θ_i − θ_target,i) over free
# nodes; `r_mag` = |mean e^{iθ}| (the Kuramoto order parameter, exactly);
# the trace for `r_yy` is mean cos(θ) over free nodes. At q=2 all three
# reduce to their Ising forms.

# %%
def clock_sweep(s, adj, clamp_mask, free_idx, q, J, B, T, n_sweeps):
    """Heat-bath Gibbs for the q-state clock model; clamped nodes held fixed.

    cos_diff[b, a] = cos(2π(a − b)/q), so a neighbor in state b contributes
    the row cos_diff[b] to the local field over candidate states a.
    """
    ang = 2.0 * np.pi * np.arange(q) / q
    cos_diff = np.cos(ang[None, :] - ang[:, None])   # [q, q]
    s = s.copy()
    for _ in range(n_sweeps):
        order = rng.permutation(free_idx)
        for i in order:
            h = np.zeros(q)
            for j in adj[i]:
                w = (B * J) if clamp_mask[j] else J
                h += w * cos_diff[s[j]]
            p = np.exp((h - h.max()) / max(T, 1e-9))
            p /= p.sum()
            s[i] = rng.choice(q, p=p)
    return s


def sample_clock_equilibrium(G, target, clamp_idx, q, J=1.0, B=1.0, T=1.0,
                             burn=40, record=60):
    """Clock-model analog of sample_equilibrium; returns state + cos trace."""
    N = G.number_of_nodes()
    adj = [list(G.neighbors(i)) for i in range(N)]
    clamp_mask = np.zeros(N, dtype=bool)
    clamp_mask[clamp_idx] = True
    s = rng.integers(0, q, size=N)
    s[clamp_idx] = target[clamp_idx]
    free_idx = np.array([i for i in range(N) if not clamp_mask[i]])

    s = clock_sweep(s, adj, clamp_mask, free_idx, q, J, B, T, burn)
    trace = []
    for _ in range(record):
        s = clock_sweep(s, adj, clamp_mask, free_idx, q, J, B, T, 1)
        s[clamp_idx] = target[clamp_idx]
        trace.append(np.cos(2 * np.pi * s[free_idx] / q).mean())
    return s, free_idx, np.array(trace)


def clock_metrics(s, free_idx, target, trace, q):
    theta = 2 * np.pi * s[free_idx] / q
    theta_t = 2 * np.pi * target[free_idx] / q
    fid = float(np.cos(theta - theta_t).mean())
    r_mag = float(abs(np.exp(1j * theta).mean()))
    x = trace - trace.mean()
    denom = (x * x).sum()
    r_yy = float((x[:-1] * x[1:]).sum() / denom) if denom > 1e-12 else 0.0
    return dict(r_mag=r_mag, fid=fid, r_yy=r_yy)

# %% [markdown]
# ## 10. Run
# Defaults are sized to run on a laptop CPU in a couple of minutes. Scale `N`,
# `record`, and `sweeps_per_step` up once you swap in the THRML sampler and have
# a GPU under it.

# %%
if __name__ == "__main__":
    cfg = Sweep(N=196)                     # 14x14 lattice-compatible
    res = run_monolithic(cfg)
    print(f"monolithic cells: {len(res)}")

    # H1/H2 view
    plot_thresholds(res, topo="scale_free", ratio=2.0)

    # H3: staged vs monolithic at the SAME total clamp budget on one graph.
    # Coherent target — a random pattern is not an attractor of a ferromagnet
    # (see modeling notes: arbitrary targets need Hopfield couplings J_ij=t_i·t_j).
    G = make_graph("scale_free", cfg.N, seed=1)
    N = G.number_of_nodes()
    target = np.ones(N, dtype=int)
    staged = run_staged(G, target, K=4, frac_per_step=0.02, strategy="ci", B=2.0)
    clamp_idx = place_carriers(G, staged["total_budget"], strategy="ci")
    s, free_idx, trace = sample_equilibrium(G, target, clamp_idx, B=2.0, T=1.0,
                                            burn=30, record=30)
    mono = metrics(s, free_idx, target, trace)
    print("staged:    ", {k: round(staged[k], 3) for k in ("fid", "total_budget", "mean_ryy")})
    print("monolithic:", {"fid": round(mono["fid"], 3),
                          "total_budget": staged["total_budget"],
                          "r_yy": round(mono["r_yy"], 3)})

    # H4: wrong-target control. Carriers broadcast a COHERENT but wrong signal
    # (all -1); we score against the intended target (all +1). Expect the mesh to
    # phase-lock (|r_mag| high) onto the wrong sign (fid -> negative). This is the
    # thermostat-burning-the-house failure made numerical: coherence without fidelity.
    intended = np.ones(G.number_of_nodes(), dtype=int)
    wrong = -np.ones(G.number_of_nodes(), dtype=int)
    clamp_idx = place_carriers(G, 0.10, strategy="ci")
    s, free_idx, trace = sample_equilibrium(G, wrong, clamp_idx, B=5.0, T=2.5)
    ctrl = metrics(s, free_idx, intended, trace)   # score against the TRUE sign
    print("wrong-target control (want r_mag high, fid negative):",
          {k: round(v, 3) for k, v in ctrl.items()})

    # H5: Hopfield recall of a structured pattern, scale-free at calibrated T
    # (Tc ~ 6.0 from run_tc.py -> T = 6.9). Expect fid to rise with fraction
    # while r_mag stays ~0; r_struct should track fid.
    print("\nH5 Hopfield recall (scale_free, T=6.9, B*J/T=2.0):")
    print("strategy  frac    fid   r_mag  r_struct   r_yy")
    for strat in ("random", "degree"):
        for frac in (0.02, 0.06, 0.10, 0.16):
            h5 = hopfield_recall(G, frac, strategy=strat)
            print(f"{strat:8s}  {frac:.2f}  {h5['fid']:5.2f}  {h5['r_mag']:5.2f}"
                  f"     {h5['r_struct']:5.2f}  {h5['r_yy']:5.2f}")

    # H6/H6b: multi-pattern capacity at the calibrated operating point.
    print("\nH6 multi-pattern capacity (scale_free, T=6.9, frac=0.10, degree):")
    print("   P   fid_cued  fid_other  r_struct")
    for P in (1, 2, 3, 5, 8, 12):
        h6 = multi_pattern_recall(G, P, 0.10, strategy="degree")
        print(f"  {P:2d}    {h6['fid']:5.2f}     {h6['fid_other']:5.2f}"
              f"     {h6['r_struct']:5.2f}")