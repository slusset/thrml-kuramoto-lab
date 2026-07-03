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
out of a metastable trap AND ordering from disorder).

**Hopfield extension (H5, section 7 of the file).** Couplings
`J_ij = J·t_i·t_j` make an arbitrary pattern `t` an energy minimum, so
carriers clamping a few nodes of the pattern trigger associative
recall — intent stored in the relationships, not broadcast. (A gauge
transform maps it back to the ferromagnet, so the step-0 Tc carries
over.) Scale-free, T=6.9, B·J/T=2.0:

| strategy | frac | fid | r_mag | r_struct |
|---|---|---|---|---|
| random | 0.02 | 0.41 | 0.04 | 0.43 |
| random | 0.16 | 0.79 | 0.07 | 0.82 |
| degree | 0.02 | 0.73 | 0.01 | 0.71 |
| degree | 0.06 | 0.90 | 0.05 | 0.91 |
| degree | 0.16 | 0.94 | 0.02 | 0.95 |

Three confirmations in one table: recall rises with carrier fraction
and degree placement beats random (the design rule survives structured
targets); **`r_mag` ≈ 0 throughout** — magnetization-based homeostatic
monitoring is completely blind to structured order, so the fid/r_mag
split is now real rather than manufactured by the H4 wrong-target
trick; and `r_struct` (mean edge coupling-satisfaction
`sign(J_ij)·s_i·s_j` — target-blind, locally computable) tracks
fidelity almost exactly. **The honest homeostatic monitor for
structured intent is coupling-satisfaction, not consensus.** This
finding is now recorded in the `homeostasis_telos` CHARTER ("The
relational finding") together with its flag: the monitor is honest
exactly insofar as the couplings are — the defense budget moves from
hub capture-resistance to covenant integrity.

**Ensemble pass (`run_ensemble.py`, 5 seeds — new graph, dynamics RNG,
and pattern per replicate; CSVs in `data/`).** Everything above
survives with error bars:

- **scale_free** — degree/CI ≫ random at every fraction (1% carriers:
  0.59±0.06 vs 0.31±0.07; 9%: 0.91±0.01 vs 0.64±0.05). CI = degree to
  within ±0.01 across the whole curve.
- **small_world** — 5 seeds resolve what single-seed noise hid: a
  modest smart-placement edge at sparse fractions (1%: degree
  0.37±0.11, CI 0.45±0.17 vs random 0.16±0.11), converging by ~9%.
  Consistent with "gain grows with heterogeneity" — mild heterogeneity,
  mild gain.
- **lattice** — random wins decisively (20%: 0.94±0.02 vs 0.55±0.13);
  degree and CI produce literally identical rows (equal degrees ⇒ same
  tie-broken ranking ⇒ same clumped carrier set).
- **Hopfield (H5)** — degree recall 0.95±0.02 at 16% with r_mag
  0.04±0.03; r_struct tracks fid within ~0.01 in every cell. One new
  wrinkle: random-placement recall plateaus (~0.66 from 10% to 16%) —
  resolved below.

**The random-placement "plateau", dissected (`run_plateau.py`).** Four
probes, scale-free at the calibrated T:

1. *Gauge check.* Hopfield/random and ferromagnet/random at equal
   fraction and seed are statistically identical (as the gauge
   transform `σ_i = s_i·t_i` demands) — pattern interference is ruled
   out by construction. The plateau is a fact about random placement,
   not about memory.
2. *Dense grid, 10 seeds.* Not a true plateau — a slow climb with a
   soft shoulder: 0.63 → 0.69 → 0.70 → 0.79 → 0.81 across 8–24%
   (degree reference: 0.86 → 0.99). The 5-seed "0.67 → 0.65" was noise
   sitting on the shoulder.
3. *Horizon test.* fid at burn 40/160/640: 0.69/0.73/0.70 — flat.
   **Not kinetic.** Unlike the metastable-flip regime (where every
   threshold was a race), the shortfall here is an equilibrium
   property; waiting doesn't help.
4. *Domain anatomy.* The unrecalled mass is not a stuck domain: ~30
   wrong free nodes scatter into 13–24 clusters of size mostly 1–5.
   Wrong nodes have lower degree (≈4.0 vs 6.6) and sit farther from
   the nearest carrier (≈2.0 vs 1.4 hops) than right nodes.

So the honest name for it is an **influence-radius / coverage law**:
just above Tc the field is paramagnetic on its own, and a free node
holds the target only insofar as clamp influence reaches it (finite
correlation length). Random carriers — mostly low-degree, because most
nodes are — each cover a small ball, and new ones increasingly land in
already-covered territory; the residue is flickering low-degree
periphery at distance ≥2 from every carrier. Hub placement wins
because hubs put most of the graph within ~1 hop of a carrier:
coverage per carrier is maximal. Design-rule phrasing: **in the
steerable regime, intent has a finite influence radius — placement is
a covering problem, and heavy-tailed graphs let a few hubs cover
almost everything.**

**Multi-pattern capacity (H6/H6b, `run_capacity.py`, 5 seeds).** Hebb
superposition `J_ij = Σ_μ t^μ_i·t^μ_j` (unnormalized — each covenant
laid on top of the others), cue pattern 0 with 10% carriers, scale-free
at the calibrated T. The gauge trick does not survive superposition, so
this is new physics for the lab. P=1 reproduces the H5 ensemble exactly
(regression gate passed).

| P | fid_cued (degree) | fid_cued (random) | r_struct obs | r_struct ceiling* |
|---|---|---|---|---|
| 1 | 0.94±0.03 | 0.67±0.04 | 0.94 | 1.00 |
| 2 | 0.73±0.07 | 0.51±0.05 | 0.44 | 0.50 |
| 3 | 0.59±0.06 | 0.46±0.08 | 0.47 | 0.49 |
| 5 | 0.46±0.10 | 0.34±0.09 | 0.42 | 0.38 |
| 8 | 0.38±0.10 | 0.24±0.15 | 0.36 | 0.29 |
| 12 | 0.27±0.09 | 0.15±0.07 | 0.36 | 0.23 |

*ceiling = r_struct of the PURE cued state — interference means even
perfect recall can't satisfy every edge.

- **H6 confirmed, brutally.** The knee is at P=2: a second stored
  pattern costs ~20 points of recall. Operating capacity at fid>0.7 is
  ~1–2 patterns — matching sparse-Hopfield theory (P_c ≈ 0.14·⟨k⟩ ≈ 1
  at ⟨k⟩≈6). **Sparse meshes are cheap to steer but poor libraries**;
  library size rides on degree. Placement still helps at every P
  (degree > random throughout) but cannot rescue capacity.
- **H6b confirmed in a form sharper than predicted.** Not only does
  `r_struct` stay elevated while cued fidelity collapses — from P≈3 the
  settled state satisfies MORE couplings than perfect recall would
  (0.36 observed vs 0.23 ceiling at P=12, fid_cued 0.27). The mesh
  serves the *tangle*, not the cued intent. And `fid_other` stays low
  (≤0.19): it doesn't defect to another single memory, it fragments
  into a mixture/glass state. So the covenant monitor's honest scope
  is: **below capacity, coupling-satisfaction tracks fidelity; at
  overload, it actively rewards the drift** — the mesh looks
  *increasingly* well-covenanted as it loses the plot.
- The saving nuance: overload itself is locally detectable —
  interference shows up in the wiring (zero-weight and contested edges,
  coupling-magnitude variance) before any recall is attempted. A node
  can know *that* the covenant web is overloaded even though no local
  signal can say *which* intent is being served. Structural
  self-knowledge survives; teleological self-knowledge does not.

**Learned couplings (H7, `run_learned.py`, 3 seeds).** The
trainable-intent rung: train the edge weights by maximum likelihood
(exact positive phase — all nodes visible; THRML block-Gibbs
`estimate_moments` for the negative phase, β = 1/T_eval so learned and
Hebb weights share units and the identical eval harness). Biases
frozen at 0: same model class as Hebb, two ways of *writing* the
couplings.

- **H7a — learned intent percolates ~4× cheaper, with a catch.** At
  P=1, learned couplings give fid = 1.00±0.00 at *every* fraction
  tested, including 2% carriers where Hebb reads 0.73. But look at how:
  training on noiseless patterns grows couplings until the model
  matches ⟨s_i·s_j⟩ = 1 (mean |w| ≈ 3.9 vs Hebb's 1.0) — it deepens
  the valley until the pattern is overwhelming at the operating
  temperature. Per the master curve, depth is the price of
  redirectability: learned-on-noiseless-data intent holds cheaper and
  will steer dearer. (Training on a noised target distribution would
  set the depth honestly — flagged, not run.)
- **H7b — learning does NOT move the capacity wall.** fid_cued,
  learned vs Hebb: P=2 0.81/0.75, P=3 0.61/0.58, P=5 0.49/0.50,
  P=8 0.31/0.39, P=12 0.28/0.28. Same knee, same collapse; past the
  wall learning is if anything slightly worse (it spreads error across
  patterns where Hebb stores each equally). **The covenant budget is a
  property of the substrate — graph sparsity × pairwise couplings —
  not of the writing rule.** No cleverer covenant-writing buys
  capacity; only densifying the web or higher-order covenants can.
- **H7c — and the wall announces itself during learning.** The
  training diagnostic is its own overload detector: at P=1 the
  moment gap converges to 0.014; at P≥2 it *sticks* at 0.2–0.4 no
  matter how long you train — the model class simply cannot make its
  equilibrium look like all the things it's being asked to remember.
  That is an earlier and cleaner overload signal than wiring conflicts:
  the mesh can measure its own insufficiency at covenant-writing time,
  before any recall is attempted, let alone failed.

**Clock phases (H8a, `run_clock.py`, 3 seeds).** The q-state clock
model (`J·cos(θ_i−θ_j)`; q=2 IS Ising, q→∞ approaches XY/Kuramoto) is
the dial for finding 6's open mechanism question: does the Gibbs
substrate punish heterogeneity harder *because* binary spins can't
partially align? Per-q calibration first (the step-0 lesson pays
immediately: Tc drops from 6.0 at q=2 to ~3.0 for all q≥4 — running
q=16 at the Ising temperature would have measured pure disorder). Then
fid-vs-fraction at 1.15·Tc(q), random vs degree, scale-free. The q=2
row reproduces the Ising ensemble (gate passed). Result — the mean
degree-minus-random gap:

| q | 2 | 4 | 8 | 16 |
|---|---|---|---|---|
| placement gap | +0.26 | +0.24 | +0.25 | +0.23 |

**H8a falsified in this regime** — the gap is flat in q (a ~13% drift,
non-monotonic, within 3-seed noise), not the predicted shrink.
Softness does not dissolve the covering problem: near Tc the influence
radius is set by the correlation length, not by whether alignment is
graded, so **placement-as-covering is invariant across the whole
Ising→XY dial** in the equilibrium (ordering-from-disorder) protocol.
The design rule gets *more* universal; the mechanism hypothesis
relocates: finding 6's 8× gain difference was measured in the
*metastable-flip kinetic* protocol, so "hubs anchor the old phase
absolutely when spins are binary" is a claim about escaping traps, not
about coverage. Testing the dial there — clock model below Tc, clamps
flipped mid-run — is the honest follow-up.

Where this leaves the lab: the design rules are closed for this model
class (placement = covering — now q-invariant, capacity = substrate,
monitor = covenant-satisfaction below capacity). Open doors: the
kinetic-regime clock dial (finding 6's mechanism), Potts capacity
(H8b: does recall capacity grow ~q² with richer covenants?), denser
webs / higher-order factors, and the THRML-hardware questions —
annealed β and the real DTM denoising chain behind the section-5
proxy.

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
