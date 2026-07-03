# thrml-kuramoto-lab

The `homeostasis_telos` carrier-fraction experiment, re-expressed as a
**clamped probabilistic graphical model** in [THRML](https://docs.thrml.ai)
(Extropic's JAX library for block Gibbs sampling of energy-based models).

The question is the same one the Kuramoto oracle asks: *at what carrier
fraction `h` does intent percolate across the mesh faster than purpose
drifts?* The oracle finds ~0.04. Here the same question becomes: at what
clamp density do pinned spins nucleate a global flip out of a metastable
state?

## The mapping

| Kuramoto oracle (`homeostasis_telos/main.py`) | Clamped PGM (this repo) |
|---|---|
| oscillator on a ring, horizon-6 neighborhood | `SpinNode` on the same ring, edges within ±6 |
| mutual coherence pull `K·sin(ψ−θ)` | ferromagnetic coupling `J` per edge |
| carrier reconstructs the target locally | node **clamped** to the target spin |
| intent diffuses via neighbors | block Gibbs relaxation of free spins |
| the target moves (novel perturbation) | clamp values **flip** mid-run |
| holographic depth `h` | fraction of nodes clamped |
| homeostatic score = mean local `r` | mean `sᵢsⱼ` over free–free edges (target-blind) |
| teleological score = `cos(θ − target)` | mean `sᵢ · target` |

The honest-failure rule survives the translation: below the critical `h`
the free lattice stays metastably in the OLD state after the flip —
every edge agreeing, globally wrong. **FROZEN: locked, off-target.**
Above it, the carriers nucleate the flip. The homeostatic score never
references the target.

One physics note: the knob pair `(β, J)` sets how deep the metastable
trap is (the analog of the oracle's noise level, inverted). Too shallow
and even `h=0` drifts across by fluctuation; too deep and nothing flips
on your watch timescale. The interesting regime is in between — that's
the first thing worth exploring.

## Setup

```bash
cd thrml-kuramoto-lab
uv venv && uv pip install -e ".[notebook]"
```

(or plain `pip install -e ".[notebook]"` in a venv; Python ≥ 3.10.
Runs fine on CPU; JAX will use a GPU if one is visible.)

## Run

```bash
uv run python run_sweep.py            # oracle-style h table + critical scan
uv run python run_sweep.py --beta 1.2 --j 0.4
```

Or start with the notebook:

```bash
uv run jupyter lab notebooks/01_carrier_fraction_clamped_pgm.ipynb
```

## Layout

```
├── run_sweep.py                  # CLI mirror of the oracle's __main__
├── src/carrier_pgm/
│   ├── model.py                  # ring EBM, graph coloring, clamped blocks
│   └── experiment.py             # two-phase protocol, scores, h-sweep
└── notebooks/
    └── 01_carrier_fraction_clamped_pgm.ipynb
```

## Early findings (β=1.0, J=0.15, N=120, seed 7)

Threshold vs topology, random carrier placement — the naive
"shortcuts drop h_c" prediction is **wrong** in this regime:

| topology | h_c (random placement) |
|---|---|
| ring (oracle's graph) | ≈ 0.015 (oracle itself: ≈ 0.025–0.03) |
| small_world (p=0.1) | between 0.02 and 0.05 — *higher* |
| scale_free (m=6) | > 0.10 — frozen everywhere tested |

Why: this is nucleation in a metastable state, not bond percolation.
On the ring a flipped droplet has a cheap boundary (two domain walls)
and grows; shortcuts and hubs give every node more neighbors
reinforcing the old state, deepening the trap.

But **placement beats fraction** (pinning control): on the scale-free
graph, hub-placed carriers flip the network at h=0.02 where random
placement stays frozen at h=0.10 (`--placement hub`). For the mesh
design rule, that reads: a few percent *placed on the hubs*, not a few
percent anywhere.

## Rigor status (`run_rigor.py`)

Three checks run before trusting the numbers (seeds / equil / size):

- **Seeds:** ring h_c wobbles across placements: ≈ 0.015–0.03. This
  *overlaps* the oracle's 0.025–0.03 — the substrates agree more, not
  less, once seed variance is honest. Quote h_c ≈ 0.02 ± 0.01.
- **Equilibration:** above threshold, fidelity is stable at 1×/2×/4×
  Gibbs steps. Below threshold it's noisy — that's marginal nucleation,
  expected. Pass.
- **Finite size (N=60/120/240):** the controlling variable is carrier
  *density*, not count — equal spacing flips equally (N=120,k=3 ≈
  N=240,k=6). But N=240 at low k shows partial flips at a fixed watch
  window: domains were still growing. **The threshold is kinetic, not
  an equilibrium constant** — any h>0 flips eventually; h_c is "flips
  within the watch window." Always quote h_c with its time horizon.
  This is the charter's own framing made quantitative: intent must
  percolate *faster than purpose drifts* — a race, not a state.

## The master curve (`run_master.py`)

Because β multiplies the whole energy, the Gibbs dynamics depend only
on the product β·J — the data collapse is exact by construction. The
law itself (ring, random placement, 300-sweep horizon, median of 3
seeds, `data/master_curve.png`):

| β·J | 0.08 | 0.10 | 0.125 | 0.15 | 0.175 | 0.20 | 0.25 | 0.30 |
|---|---|---|---|---|---|---|---|---|
| h_c | 0 (disordered) | 0.04* | 0.02 | 0.02 | 0.03 | 0.03 | 0.04 | 0.08 |

*near-critical, weakly ordered — noisiest point on the curve.

The shape is a U with its minimum just above the ordering onset:
disorder holds nothing (nothing to steer), deep order is expensive to
redirect (β·J=0.30 needs 4× the carriers of 0.15). **Steering is
cheapest near criticality.** The charter's homeostatic band
[r_lo, r_hi] — deliberately short of full lock — sits exactly where
this curve says a steerable mesh should live. "Full lock is the
brittle, frozen mode" now has a price tag.

Threshold criterion is relative (fidelity > 0.6 × coherence), since an
absolute cut conflates "flipped" with "ordered" in shallow traps.

## Placement law, tested at scale (`--placement random|hub|ci`)

Collective-influence (CI_2, Morone & Makse) was predicted to beat
degree placement, with the gap widening on scale-free graphs. Tested at
N=2000 (1s/run — JAX shrugs at the scale-up):

| graph (N=2000) | k_c ci | k_c hub | k_c random |
|---|---|---|---|
| scale-free (m=2, j=0.5) | 15 | 15 | ≫15 |
| Erdős–Rényi (m=3, j=0.35) | 100–120 | 100–120 | 200–250 |

**CI = degree, everywhere we could measure.** On scale-free the top-k
rankings are literally the same nodes. On ER the rankings genuinely
diverge (75% overlap at k=120) — and the thresholds *still* tie. The
defensible reading: CI optimizes network *fragmentation*; kinetic
nucleation cares about local field injection, and for that degree is
the right heuristic. (CI's regime may also be k/N ≪ 1% on N≳10⁵
graphs; untested here. ER numbers are single-seed.)

What did hold from the prediction: the smart-vs-random gap widens with
degree heterogeneity — ≥13× on scale-free (15 vs ≫200), ~2× on ER,
zero on the ring by symmetry. And a scaling gift: on scale-free the
hub threshold *fraction* falls with N (h_c: 0.025 at N=120 → 0.0075
at N=2000) — hubs grow with the graph, so hub-seated intent gets
cheaper at scale. For the mesh design rule: heavy-tailed connectivity
plus hub-seated carriers is the economical regime, and plain degree is
all the placement math you need.

## The adversarial reading: infiltration (`run_contest.py`)

The same result read backwards is a vulnerability bound. Attackers
capture the top-k hubs (clamped anti-target); defenders hold loyal
clamped nodes. Scale-free, defaults, fidelity to the TRUE target:

| captured | no defense | 5% random | 10% random | 5% next-hubs |
|---|---|---|---|---|
| 2 | HELD | HELD | HELD | HELD |
| 3 | **LOST** | LOST | HELD | torn |
| 5 | LOST | LOST | LOST | torn |

Three captured hubs steal the network. Random defense barely helps;
even next-best-hub defense only degrades takeover into civil war.
Seats beat numbers in both directions — whoever holds the top hubs
wins, and the defense that works is *capture-resistance of the hubs
themselves*, not loyal mass elsewhere.

The nastiest detail is what the target-blind coherence monitor sees:

```
healthy               hom +0.78   tel +0.82
contested (held)      hom +0.57   tel +0.64   <- alarm fires
clean takeover        hom +0.75   tel -0.81   <- alarm SILENT
torn (civil war)      hom +0.44   tel -0.53   <- alarm fires
```

Coherence only dips *while the fight is on*. A successful takeover
restores it — every local check green again, pointed the wrong way.
Homeostatic monitoring gives you a detection *window*, not a detection
*guarantee*: if the flip completes faster than your monitoring cadence,
it never happened, as far as the mesh can tell. (The charter's FROZEN
trap, now with an adversary steering it.)

## The substrate-invariance test (`run_substrate.py`)

The closer. Same graph instances, same carrier sets (matched RNG),
two entirely different physics: the oracle's Kuramoto dynamics
(`homeostasis_telos/substrate_invariance.py`, verified against
`main.py` on the ring before being trusted) vs block Gibbs on the
clamped EBM. N=120, seed 7, kinetic horizons fixed per substrate:

| condition | h_c Kuramoto | h_c Gibbs |
|---|---|---|
| ring / random | 0.02 | 0.02 |
| small_world / random | 0.02 | 0.04 |
| scale_free / random | 0.03 | **0.16** |
| scale_free / hub | **0.01** | 0.03 |

**Every effect has the same sign in both substrates.** Degree
heterogeneity penalizes random placement in both; hub placement wins
in both (3× in Kuramoto, 5× in Gibbs); the ring is the easy case in
both. The orderings agree. What differs is the gain: Gibbs punishes
heterogeneity ~8× harder than Kuramoto (binary spins can't partially
align, so unclamped hubs anchor the old phase absolutely; soft phases
let intent leak through a hub gradually).

So the claim lands in its honest form: **the topology owns the
ordering; the substrate sets the scale.** Constants don't transfer —
design rules do. "Percolation-aware placement of a few percent of
carriers" survived translation between a deterministic oscillator
swarm and a stochastic sampler; "4%" did not, and was never going to.

## The DTM bridge (`dtm_col_infl_clamping.py`)

A self-contained notebook-style script (pure numpy/networkx Glauber, no
THRML dep — the sampler is an explicit seam for the block-Gibbs swap-in)
that unions this repo's carrier-placement results with the Extropic
DTM/DTCA paper's move: escape the mixing–expressivity tradeoff by
*chaining shallow, well-mixed steps* instead of asking one deep
landscape to carry the whole telos.

What it adds that the repo doesn't have:

- **Staged clamping (H3)** — K shallow settle steps with re-placed
  carriers, vs one monolithic settle at the same total clamp budget.
  The temporal twin of the guild/carrier-cluster move: telos composed
  from small aligned fields.
- **The `r_yy` mixing sensor** — lag-1 autocorrelation of the
  magnetization trace, a minimal stand-in for the paper's Adaptive
  Correlation Penalty. High `r_yy` = sluggish mixing = don't trust the
  sample.
- **A different protocol.** This repo's runs are *nucleation out of a
  metastable trap* (field ordered in the OLD state, clamps flip
  mid-run). The DTM script runs *ordering from disorder* (T above the
  interior Tc, so any order is carrier-driven). Same percolation
  question, different physics regime — thresholds are not comparable
  across the two without care.

Two honest collisions with the results above:

- Its **H1 predicts CI beats degree** — but finding 4 (N=2000, both
  regimes' placement math) already showed CI = degree everywhere
  measured. Treat H1 as a re-test in the new regime, not an open
  question.
- A companion chat produced a **Sheaf-ADMM extension** (three-vertex
  framing, `sheaf_admm_consensus`, deterministic-vs-thermodynamic
  coordination cell) that was never landed here — the disk copy is the
  pre-ADMM version. If that comparison matters, it needs to be
  re-created or exported from that chat.

First run status (2026-07-03, N=196, T=2.5, seed 7): **the sweep is
saturated** — fid ≈ 0.9–1.0 for every strategy at every fraction down
to 0.01, on small-world and scale-free at both B·J/T ratios. The
script's own modeling note names the cause: T=2.5 clears the 2D-lattice
Tc but small-world/scale-free are denser, so they sit *below* their
effective Tc and order spontaneously; clamps just pick the sign. (One
cell even ordered to the wrong sign: ci/0.01/ratio-1.) The companion
chat's "CI wins on thermo fidelity" (0.912/0.982/1.000) reproduces here
as 0.91/0.98/0.99 — saturation noise, not a placement effect. **Step 0
(locate each topology's critical point) is prerequisite to every
hypothesis in the file.** H4 does confirm cleanly (r_mag=1.0,
fid=−1.0), and staged-vs-monolithic at equal budget (fixed to use a
coherent target) reads 1.0 vs 0.989 — also at saturation, so H3 is
untested, not supported.

**Step 0 done (`run_tc.py`, same day).** Unclamped Glauber across a
temperature grid per topology; Tc estimated where spontaneous |m|
crosses 0.5 (the susceptibility peak is too noisy at N=196 single-seed
— it put the lattice at 1.75 vs the exact 2.27; the crossing gave 2.5):

| topology | Tc (N=196) | sweep now runs at 1.15·Tc |
|---|---|---|
| lattice | ≈ 2.5 (exact: 2.27) | 2.88 |
| small_world (k=6) | ≈ 4.0 | 4.6 |
| scale_free (m=3) | ≈ 6.0 | 6.9 |

So the original T=2.5 sat at ~0.6·Tc for small-world and ~0.4·Tc for
scale-free — deep in the ordered phase, hence the saturation.
Re-swept at 1.15·Tc, B·J/T=2.0, H1 resolves into the same design rule
the metastable-flip protocol found:

- **scale_free** — clean separation: degree/CI reach fid ≈ 0.8 by 4%
  carriers while random is still at 0.57 and never clears 0.8 by 20%.
  **CI = degree within noise** (finding 4 again, new regime).
- **small_world** — no consistent placement separation; mild
  heterogeneity, little for placement to exploit.
- **lattice** — *random wins*; with all degrees equal, "top-degree"
  degenerates to node order, which clumps carriers into one region;
  spreading beats clumping on homogeneous graphs.

Placement gain grows with degree heterogeneity, and degree is all the
placement math you need — now confirmed in both regimes (nucleation
out of a metastable trap AND ordering from disorder). All single-seed;
threshold curves want ~5 seeds before quoting numbers.

## Where to take it next

- **Clock/Potts phases.** `CategoricalNode` with a cosine coupling factor
  gets you discretized phases — much closer to Kuramoto than ±1 spins,
  and the target shift can be the oracle's actual 1.9 rad.
- **Annealed β.** Sweep β during phase B (simulated annealing) — the
  hardware-relevant schedule, per THRML's codon-optimization example.
- **Trainable intent.** `IsingTrainingSpec` / `estimate_kl_grad` let the
  lattice *learn* couplings from target-aligned data, then you ask
  whether learned intent percolates at lower h than clamped intent.
- **Close the loop with the oracle.** Port `small_world`/`scale_free`
  topologies into `homeostasis_telos/main.py` and test whether the
  Kuramoto thresholds *track* the PGM thresholds across topologies —
  that's the substrate-invariance claim, tested for real.
