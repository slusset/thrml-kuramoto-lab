# RESULTS — carrier percolation, from oscillators to samplers

*One page. Every number below is single-configuration unless noted;
every threshold is kinetic (quote it with its time horizon). Details,
caveats, and reproduction commands are in the README; scripts:
`run_sweep.py`, `run_rigor.py`, `run_master.py`, `run_contest.py`,
`run_substrate.py`.*

## The question

The `homeostasis_telos` oracle found that a Kuramoto swarm holding
perfect internal coherence loses its purpose unless a small fraction
of nodes (~4%) carry the intent invariant — and that fraction is
startlingly small. Is that a fact about oscillators, or a fact about
networks?

## The six findings

**1. The phenomenon transfers.** Re-expressed as a clamped Ising PGM
in THRML (carriers → clamped spins, target shift → clamp flip, intent
diffusion → block Gibbs), the same structure appears: FROZEN at h=0
(locked, off-target, every local check green), sharp recovery above a
sparse threshold. Ring h_c ≈ 0.02 ± 0.01 across seeds — overlapping
the oracle's 0.025–0.03. The substrates agree within error bars.

**2. The threshold is kinetic, not a constant.** Finite-size scaling
(N=60→240) shows carrier *density* controls, not count — but partial
flips at fixed watch windows expose the truth: any h>0 flips
eventually; h_c means "flips within the horizon." This is the
charter's claim made quantitative: a race between intent percolation
and purpose drift, not a state.

**3. Steering is cheapest near criticality.** Since β multiplies the
whole energy, dynamics depend only on β·J — the data collapse is
exact. The master curve h_c(β·J) is a U: disorder holds nothing,
deep order resists redirection (β·J=0.30 costs 4× the carriers of
0.15), minimum just above ordering onset. The charter's viable band
[r_lo, r_hi] — deliberately short of full lock — sits where this
curve says a steerable mesh should live.

**4. Placement beats fraction, and degree is the placement math.**
Scale-free, random placement: h_c > 0.10. Same graph, carriers on
hubs: 0.02. At N=2000 the hub threshold *fraction* falls to 0.0075 —
hubs grow with the graph, so seated intent gets cheaper at scale.
Collective influence (Morone–Makse CI_2) never beat plain degree,
even on Erdős–Rényi where the rankings genuinely diverge: nucleation
rewards local field injection, and degree measures exactly that.

**5. The same law read backwards is an attack surface.** Three
captured hubs steal a 120-node network; random defense barely helps;
next-best-hub defense yields civil war, not recovery. Worst: the
target-blind coherence monitor dips only *during* the contest
(0.78 → 0.57) and recovers after a successful takeover (0.75) —
a completed flip erases its own evidence. Detection is a window,
not a guarantee. Harden the hubs; monitor the derivative.

**6. Substrate invariance, in its honest form.** Same graph
instances, same carrier sets, two physics (verified Kuramoto port vs
block Gibbs):

| condition | h_c Kuramoto | h_c Gibbs |
|---|---|---|
| ring / random | 0.02 | 0.02 |
| small_world / random | 0.02 | 0.04 |
| scale_free / random | 0.03 | 0.16 |
| scale_free / hub | 0.01 | 0.03 |

Every effect has the same sign; the orderings agree; only the gain
differs (binary spins punish heterogeneity ~8× harder than soft
phases). **The topology owns the ordering; the substrate sets the
scale.**

## The design rule (current form)

For a mesh that must stay steerable: heavy-tailed connectivity,
intent carriers seated on the hubs (plain degree suffices), coupling
held just above the ordering transition, coherence-derivative
monitoring for capture events, and hub capture-resistance as the
security budget's first line. Sparse suffices — a few percent,
placed, not a majority — but "4%" was one slice through a law, not
the law.

## What would falsify the next step

Ensemble statistics (all adversarial and ER numbers are single-seed);
CI vs degree at N≥10⁵ where k/N ≪ 1%; clock-model phases
(CategoricalNode) to test whether soft-vs-binary explains the gain
difference; and the drift-speed axis — every threshold here assumes
the target moves once, not continuously.
