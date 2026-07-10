# Current results

Current through commit `ac97342`. This document integrates the experiment arc
recorded in the generated CSVs and run scripts. It replaces the original
six-finding snapshot, which is preserved at
[`docs/RESULTS_INITIAL_ARC.md`](docs/RESULTS_INITIAL_ARC.md).

## How to read these results

The lab contains several protocols with different initial conditions and
different meanings of “threshold.” Numerical values are not transferable
between protocols without matching system size, topology, temperature,
control strength, update rule, criterion, and time horizon.

Evidence labels used below:

- **implementation check** — expected identity or regression gate reproduced;
- **observed** — present in a recorded finite protocol;
- **ensemble-supported** — repeated over the stated small ensemble;
- **mechanism hypothesis** — interpretation that still needs a discriminating
  experiment;
- **cross-substrate hypothesis** — qualitative agreement across model classes,
  not a universal law.

See [`docs/CONCEPTS.md`](docs/CONCEPTS.md) for definitions and reporting rules.

## 1. Metastable reversal

### Sparse pins can redirect an ordered Ising state

**Evidence: ensemble-supported within the finite-horizon protocol.**

The THRML experiment starts the free spins in the old ordered state, flips the
clamped spins to the new state, and observes the free system. On the
`N=120`, horizon-6 ring at `beta=1`, `J=0.15`, the operational crossing is near
`h=0.02`; seed-to-seed placement variation puts it roughly in the
`0.015–0.03` range.

This is a kinetic first-passage result. It does not establish a universal
critical carrier fraction.

### Escape cost depends on trap depth

**Evidence: observed over three seeds per grid point.**

For the finite ring and a 300-sweep horizon, the recorded master curve is:

| `beta*J` | 0.08 | 0.10 | 0.125 | 0.15 | 0.175 | 0.20 | 0.25 | 0.30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| operational `h_c` | 0.00* | 0.04 | 0.02 | 0.02 | 0.03 | 0.03 | 0.04 | 0.08 |

`*` Classified as disordered by the protocol rather than as a successful
controlled reversal.

The U-shaped finite-size curve supports a practical tradeoff: shallow order is
easy to disturb, while deeper ordered states are more expensive to redirect.
“Steering is cheapest near criticality” should currently be read as
“steering is cheapest near the measured finite-system ordering crossover.”

### Matched Kuramoto and Gibbs conditions have the same qualitative ordering

**Evidence: cross-substrate hypothesis; single matched graph instance.**

| condition | Kuramoto `h_c` | Gibbs `h_c` |
|---|---:|---:|
| ring / random | 0.02 | 0.02 |
| small-world / random | 0.02 | 0.04 |
| scale-free / random | 0.03 | 0.16 |
| scale-free / hub | 0.01 | 0.03 |

Both model classes penalized random placement on the heterogeneous graph and
benefited from hub placement. The scale differed substantially. This supports
qualitative robustness of the placement ordering, not substrate invariance as
a proved law. `run_substrate.py` also depends on the sibling
`homeostasis_telos` checkout.

## 2. Ordering from disorder and placement

### The first placement sweep was below the relevant ordering crossover

**Evidence: implementation correction.**

The initial `T=2.5` sweep saturated on the denser small-world and scale-free
graphs because those systems ordered without meaningful help from the pins.
Finite-graph magnetization crossings were then used to define operating
crossovers:

| topology, `N=196` | measured ordering crossover | carrier-run temperature |
|---|---:|---:|
| periodic square lattice | about 2.5 | 2.88 |
| small-world, mean degree about 6 | about 4.0 | 4.60 |
| BA graph, mean degree about 6 | about 6.0 | 6.90 |

These are pseudocritical operating points, not high-precision thermodynamic
critical temperatures.

### Placement advantage depends on heterogeneity and dispersion

**Evidence: ensemble-supported over five graph/dynamics seeds.**

- On the scale-free graphs, degree and static CI placement strongly
  outperformed random placement. At 1% pins, fidelity was about `0.59±0.06`
  for degree/CI versus `0.31±0.07` for random; at 9%, about `0.91±0.01`
  versus `0.64±0.05`.
- On small-world graphs the advantage was smaller and noisiest at sparse
  fractions.
- On the regular lattice, random placement outperformed degree/CI because
  equal-score ties resolved by node label and produced a clumped set.

The bounded result is that degree is effective on the tested heterogeneous
graphs. The general placement problem also includes spatial coverage and
control-set dispersion. Static CI did not beat degree, but the code does not
implement the full adaptive CI selection algorithm.

### The random-placement shoulder is consistent with influence coverage

**Evidence: observed with a 10-seed fraction sweep and horizon probes.**

Increasing burn-in from 40 to 160 to 640 sweeps did not remove the recall
shortfall. Incorrect nodes were typically low-degree peripheral nodes farther
from the nearest carrier and appeared in many small components rather than one
growing domain.

The working interpretation is finite-correlation-length influence coverage,
not a stalled macroscopic domain. Exact equilibrium and mixing have not been
established with independent-chain or effective-sample-size diagnostics.

## 3. Structured recall and monitoring

### A partial cue recalls a single structured Hopfield pattern

**Evidence: ensemble-supported over five seeds; gauge-equivalent regression
check.**

On the scale-free graph at the calibrated operating point:

| placement | pin fraction | target overlap | magnetization | bond satisfaction |
|---|---:|---:|---:|---:|
| random | 0.02 | 0.39±0.07 | 0.05±0.03 | 0.36±0.06 |
| random | 0.16 | 0.65±0.08 | 0.05±0.03 | 0.66±0.06 |
| degree | 0.02 | 0.73±0.03 | 0.08±0.05 | 0.71±0.05 |
| degree | 0.10 | 0.94±0.03 | 0.03±0.02 | 0.94±0.03 |

Magnetization is blind to a balanced structured pattern. Bond satisfaction
tracks overlap in this one-pattern protocol because of the gauge equivalence.
It certifies relational consistency, not target identity: the global inverse
of a binary pattern satisfies the same pairwise signs.

### Pairwise consistency becomes misleading under overload

**Evidence: ensemble-supported over five seeds.**

With 10% degree-placed cues and unnormalized binary Hebbian superposition:

| stored patterns `P` | cued overlap | observed bond satisfaction | satisfaction of pure cued state |
|---:|---:|---:|---:|
| 1 | 0.94±0.03 | 0.94 | 1.00 |
| 2 | 0.73±0.07 | 0.44 | 0.50 |
| 3 | 0.59±0.06 | 0.47 | 0.49 |
| 5 | 0.46±0.10 | 0.42 | 0.38 |
| 8 | 0.38±0.10 | 0.36 | 0.29 |
| 12 | 0.27±0.09 | 0.36 | 0.23 |

At higher load, the settled state can satisfy more pairwise signs than the
pure cued pattern while losing cued overlap. Low overlap with any single
non-cued pattern is consistent with a mixture or glassy state, but classifying
the state requires stronger diagnostics than the current overlap table.

### Learned couplings deepen the one-pattern basin but do not move the measured capacity knee

**Evidence: observed over three training seeds.**

Maximum-likelihood training on noiseless patterns produced much larger weights
than the Hebbian baseline and perfect measured P=1 recall at every tested pin
fraction, including 2%. This is not yet evidence of intrinsically cheaper
control: the learned model has a deeper energy landscape and `h=0` was not part
of the reported fraction grid.

For multiple patterns, learned and Hebbian systems lost cued recall at roughly
the same load. A noised-target training distribution and a matched-weight-scale
comparison remain necessary.

## 4. State-space granularity

### Equilibrium coverage was nearly flat across clock resolution

**Evidence: observed over three seeds.**

At per-`q` calibrated operating points on the scale-free graph, the mean
degree-minus-random placement gap was approximately:

| clock states `q` | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|
| placement gap | 0.26 | 0.24 | 0.25 | 0.23 |

The prediction that finer phases would remove the equilibrium coverage penalty
was falsified in this protocol.

### Kinetic escape became cheaper between `q=4` and `q=8`

**Evidence: observed over three seeds on a coarse threshold grid.**

For a scale-free graph at `0.6` times the measured per-`q` crossover, `B=1`,
and a 150-sweep horizon:

| `q` | random operational `h_c` | degree operational `h_c` |
|---|---:|---:|
| 2 | 0.12 | 0.02 |
| 4 | 0.12 | 0.02 |
| 8 | 0.02 | 0.01 |
| 16 | 0.02 | 0.01 |

This supports a granularity-dependent escape cost in the clock heat-bath
model. A nucleation-to-gradual-rotation crossover is a mechanism hypothesis;
the current code does not record or classify successful angular paths.

## 5. Potts associative capacity

### Richer pairwise state tables raise the measured capacity

**Evidence: ensemble-supported over three seeds.**

Using centered Potts-Hebb edge tables, 10% degree-placed cues, and a per-`q`
calibrated operating point, the `m_cued >= 0.7` knee moved from approximately
one pattern at `q=2`, to about five at `q=4`, to the 8–16 range at `q=8` on
`N=196`, mean-degree-six graphs.

This establishes the direction of the effect within the protocol. Dense,
zero-temperature Potts capacity formulas do not directly predict these finite,
sparse, finite-temperature, externally cued measurements.

### Capacity is approximately proportional to mean degree over the measured grid

**Evidence: observed scaling pattern; not yet a fitted universal law.**

Interpolated `m_cued=0.7` knees were:

| mean degree | `q=4` knee | `q=8` knee | ratio |
|---:|---:|---:|---:|
| 5.9 | 4.3 | 12.1 | 2.8 |
| 11.6 | 12.3 | 32.2 | 2.6 |
| 22.5 | 18.9 | 56.4 | 3.0 |
| 42.1 | 34.3 | 108.8 | 3.2 |

The observed knees are roughly linear in mean degree over these four points.
The `q=8/q=4` ratio did not approach the borrowed dense-theory ratio as degree
increased. Finite-temperature entropy is a current mechanism hypothesis; a
temperature sweep is the discriminating experiment.

## 6. What is established, provisional, and open

### Most defensible current statements

- Sparse pinned controls can strongly change finite-horizon escape and recall.
- Placement matters most on the tested heterogeneous graphs; spatial dispersion
  matters on homogeneous graphs.
- Target-blind order and target-relative alignment must be measured separately.
- Pairwise structural satisfaction has a limited scope and can reward overloaded
  mixture states.
- State granularity affects kinetic escape in the clock model and associative
  capacity in the Potts model.

### Provisional interpretations

- topology sets a substrate-independent ordering of control difficulty;
- the clock threshold step is specifically a nucleation-to-rotation crossover;
- capacity equals an exact `alpha(q) * mean_degree` law;
- high-degree pinning is universally optimal;
- model takeover tables imply a real security bound.

### Highest-value next experiments

1. Direct Kuramoto ordering from incoherence with finite-strength pacemakers,
   heterogeneous natural frequencies, and a no-control baseline.
2. A two-dimensional sweep of pin fraction and pin strength, including `B=1`.
3. Degree-normalized versus unnormalized coupling on heterogeneous graphs.
4. Placement comparisons against graph coverage and spectral pinning methods.
5. Angular trajectory logging to test the proposed rotation mechanism.
6. Temperature sweep for the `q=4` and `q=8` Potts capacity coefficient.
7. Noised-target learning and an `h=0` learned-coupling control.
8. Larger ensembles with graph, pattern, placement, initial-condition, and
   dynamics randomness separated in the output data.
