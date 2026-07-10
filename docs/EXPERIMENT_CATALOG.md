# Experiment catalog

This catalog organizes the lab by research question rather than by the order
in which scripts were written. It is the navigation layer between the
[research program](RESEARCH_PROGRAM.md), the runnable scripts, and the evidence
in [RESULTS.md](../RESULTS.md).

## Recommended reading and run order

The most coherent path through the lab is:

1. **Calibrate the uncontrolled substrate.** Establish what the system does
   without carriers.
2. **Test acquisition of alignment.** Start from disorder and introduce a
   reference.
3. **Test adaptation.** Start from a coherent old state and change the target.
4. **Test placement and conflict.** Change where reference authority sits and
   whether multiple authorities disagree.
5. **Test relational memory and capacity.** Store structured patterns in the
   couplings and increase load.
6. **Test model boundaries.** Change state granularity and compare against
   actual oscillator dynamics.

That order prevents several interpretation errors: mistaking spontaneous order
for a carrier effect, comparing equilibrium and kinetic thresholds, or treating
an associative-memory result as generic synchronization.

## Track A — alignment and adaptation

### A0. Uncontrolled baseline and operating crossover

| field | value |
|---|---|
| scripts | `run_tc.py` |
| question | What order appears without pinned nodes on each finite topology? |
| purpose | Prevent spontaneous order from masquerading as control-induced alignment. |
| outputs | console table; operating crossover constants consumed by later scripts |
| current status | implemented; finite-graph pseudocritical calibration |
| architectural reading | Establish the baseline before crediting a governance or reference mechanism. |
| next gate | Estimate uncertainty across graph instances and record calibration data in a machine-readable artifact. |

### A1. Ordering from disorder and placement

| field | value |
|---|---|
| scripts | `run_ensemble.py`, placement portions of `dtm_col_infl_clamping.py` |
| question | Can sparse pinned references induce target alignment in an otherwise disordered spin system? |
| purpose | Test whether influence coverage and placement matter when alignment is not already present. |
| outputs | `data/ensemble_placement.csv` |
| current status | five-seed ensemble; degree/CI advantage on heterogeneous graphs; random dispersion advantage on the lattice |
| architectural reading | Reference implementations and standards need strategic coverage; degree is not a universal placement rule. |
| next gate | Sweep pin strength independently; compare degree, random, farthest-point, `k`-center, and spectral placement. |

### A2. Coverage anatomy

| field | value |
|---|---|
| scripts | `run_plateau.py` |
| question | Is incomplete alignment caused by slow domain growth or by limited influence coverage? |
| purpose | Distinguish a kinetic failure from a structural reach problem. |
| outputs | console diagnostics; no dedicated CSV yet |
| current status | longer burn-in was flat; wrong nodes were peripheral and farther from carriers |
| architectural reading | Waiting or applying more process does not repair an interface-coverage gap. |
| next gate | Persist node-level distance, degree, and outcome data; fit coverage models across graph instances. |

### A3. Metastable reversal

| field | value |
|---|---|
| scripts | `run_sweep.py`, `run_rigor.py`, `run_master.py`; core protocol in `src/carrier_pgm/experiment.py` |
| question | Can sparse pins redirect a system that is internally coherent around an old target? |
| purpose | Measure the tradeoff between stability and redirectability. |
| outputs | `data/master_curve.csv`, `data/master_curve.png`, console threshold tables |
| current status | operational ring threshold near `h=0.02` at the default finite horizon; U-shaped escape-cost curve over trap depth |
| architectural reading | Deeply reinforced local conventions can be reliable and still resist a legitimate change in purpose. |
| next gate | Replace crossing-only results with first-passage distributions over time, size, placement, and pin strength. |

### A4. Competing reference authorities

| field | value |
|---|---|
| scripts | `run_contest.py`, `src/carrier_pgm/infiltration.py` |
| question | What happens when trusted and adversarial pinned sets assert conflicting targets? |
| purpose | Test target-blind monitoring and high-leverage capture risk. |
| outputs | console contest table |
| current status | single-instance adversarial scenario; useful counterexample, not a security bound |
| architectural reading | A healthy local system can serve a captured or stale reference; provenance and authority need their own controls. |
| next gate | Add ensembles, multiple conflict geometries, authority strengths, detection latency, and recovery protocols. |

## Track B — relational memory and domain capacity

### B1. Single-pattern associative recall

| field | value |
|---|---|
| scripts | Hopfield portions of `run_ensemble.py` and `dtm_col_infl_clamping.py` |
| question | Can a structured target encoded in pairwise relationships be reconstructed from a partial cue? |
| purpose | Separate relational consistency from simple consensus. |
| outputs | `data/ensemble_hopfield.csv` |
| current status | five-seed gauge-equivalent regression result; degree cues outperform random cues |
| architectural reading | Executable boundary relationships can help reconstruct domain identity, but structural satisfaction cannot prove target identity. |
| next gate | Include the inverse-pattern control, target provenance, and asymmetric or typed relationships. |

### B2. Multi-pattern interference

| field | value |
|---|---|
| scripts | `run_capacity.py` |
| question | How does recall degrade as more patterns share the same sparse pairwise substrate? |
| purpose | Probe responsibility and covenant overload. |
| outputs | `data/ensemble_capacity.csv` |
| current status | five-seed binary Hebbian capacity curve; mixture/glassy interpretation remains provisional |
| architectural reading | A bounded context can satisfy many local relationships while losing the specific outcome it was asked to serve. |
| next gate | Add energy, replica overlap, cluster, and convergence diagnostics to classify overload states. |

### B3. Learned versus written couplings

| field | value |
|---|---|
| scripts | `run_learned.py` |
| question | Does learned structure improve recall or capacity relative to Hebbian writing? |
| purpose | Test whether stronger adaptation solves a structural capacity problem. |
| outputs | `data/ensemble_learned.csv` |
| current status | deeper learned one-pattern basin; no measured capacity-knee improvement |
| architectural reading | Learned conventions can improve fidelity by becoming harder to change; implementation cleverness may not remove a boundary problem. |
| next gate | Add `h=0`, matched-weight-scale, noised-target training, and post-training redirection tests. |

### B4. Richer relational state and capacity scaling

| field | value |
|---|---|
| scripts | `run_capacity_q.py`, `run_fss.py`, `run_density.py` |
| question | How do state richness, graph size, and mean degree change associative capacity? |
| purpose | Test whether richer pairwise covenants or denser connectivity postpone overload. |
| outputs | `data/ensemble_capacity_q.csv`, `data/ensemble_fss.csv`, `data/ensemble_density.csv` |
| current status | capacity rises with `q` and mean degree in the measured finite protocol; exact law is provisional |
| architectural reading | Richer contracts and more relationships can increase capacity, but neither substitutes for a coherent boundary. |
| next gate | Temperature sweep, fitted uncertainty, uncensored grids, and comparison with matching theoretical conventions. |

## Track C — model boundaries and transfer

### C1. Equilibrium clock granularity

| field | value |
|---|---|
| scripts | `run_clock.py` |
| question | Does a finer circular state space reduce the placement penalty during ordering from disorder? |
| purpose | Test whether graded alignment increases influence reach. |
| outputs | `data/ensemble_clock.csv` |
| current status | prediction falsified; placement gap was nearly flat across tested `q` |
| architectural reading | Graded states do not automatically repair missing coverage. |
| next gate | Repeat with independent calibration ensembles and normalized network coupling. |

### C2. Kinetic clock granularity

| field | value |
|---|---|
| scripts | `run_kinetic.py` |
| question | Does a finer circular state space reduce the cost of leaving an old ordered state? |
| purpose | Isolate state granularity in the adaptation regime. |
| outputs | `data/ensemble_kinetic.csv` |
| current status | coarse operational threshold step between `q=4` and `q=8`; rotation mechanism unmeasured |
| architectural reading | Explicit transitional states may make legitimate change less discontinuous. |
| next gate | Record angular trajectories and classify paths before naming the mechanism. |

### C3. Matched oscillator/Gibbs comparison

| field | value |
|---|---|
| scripts | `run_substrate.py`; requires sibling `homeostasis_telos` checkout |
| question | Do topology and placement rank conditions similarly under two different dynamics? |
| purpose | Test qualitative transfer without assuming numerical thresholds transfer. |
| outputs | console comparison table |
| current status | one matched instance across four conditions; cross-substrate hypothesis only |
| architectural reading | A design heuristic is more credible when it survives model changes, but four matched cells are not universality. |
| next gate | Multi-instance paired comparison with identical graph and placement artifacts saved for both models. |

### C4. Direct finite-strength Kuramoto pacemaker experiment — proposed

| field | value |
|---|---|
| scripts | not implemented |
| question | Can a few finite-strength pacemakers entrain an initially incoherent oscillator network, and does order persist after their removal? |
| purpose | Test the motivating Dyson-swarm question without hard clamps or spin-state substitution. |
| outputs | proposed phase, target-alignment, frequency-locking, first-passage, and persistence data |
| current status | next decisive experiment |
| architectural reading | Distinguish ongoing governance/reference support from genuinely internalized, self-sustaining alignment. |
| next gate | Implement the protocol specified in `docs/CONCEPTS.md`, with normalized and unnormalized coupling. |

## Cross-cutting controls

Every experiment family should eventually include:

- a no-carrier or no-control baseline;
- independent control fraction and control-strength axes;
- explicit initial-condition and observation-horizon metadata;
- graph, placement, model, and dynamics seeds recorded separately;
- both target-blind and target-relative measurements where applicable;
- multiple system sizes when a scaling or criticality claim is made;
- a negative control that could expose a misleading metric;
- a manifest linking code version, configuration, output data, and result note.
