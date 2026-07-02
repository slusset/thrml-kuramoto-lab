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

## Placement law, tested (`--placement random|hub|ci`)

Collective-influence (CI_2, Morone & Makse) placement was predicted to
beat degree placement. On this graph it can't be tested: at N=120,
m=6, the top-12 CI and top-12 degree nodes are the *same 12 nodes*
(rankings only diverge on larger, sparser graphs — N≳1000, m=2–3).
What is established: (ci = hub) h_c = 0.03 vs random > 0.10 on
scale-free — smart placement is worth >3× in carrier budget, and the
CI-vs-degree question is queued behind a bigger graph.

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
