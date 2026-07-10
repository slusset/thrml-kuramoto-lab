# Concepts and methods

This document defines the lab's models, protocols, measurements, and naming
conventions. It separates standard technical terms from project metaphors and
sets the minimum conditions that should accompany a reported result.

The motivation and architectural interpretation of these models are documented
separately in [`RESEARCH_PROGRAM.md`](RESEARCH_PROGRAM.md); the runnable
experiment map is in [`EXPERIMENT_CATALOG.md`](EXPERIMENT_CATALOG.md).

## 1. System vocabulary

### Oscillator population

Use **oscillator population**, **oscillator ensemble**, or **oscillator
network** for nodes with continuously evolving phases. A networked Kuramoto
model has the general form

\[
\dot{\theta_i}
= \omega_i
+ K\sum_j A_{ij}\sin(\theta_j-\theta_i)
+ u_i\kappa\sin(\phi-\theta_i)
+ \sqrt{2D}\,\xi_i(t).
\]

Here `theta_i` is phase, `omega_i` is natural frequency, `A` is the adjacency
matrix, `K` is mutual coupling, `u_i` selects controlled nodes, `kappa` is
control strength, and `phi` is the reference phase.

The word **field** is acceptable as a project metaphor, but it can also mean a
spatially distributed physical field or an external field in statistical
mechanics. Prefer the model-specific term in technical claims.

### Spin system

Use **spin system** for the Ising, clock, and Potts experiments.

- Ising spins have two states, conventionally `-1` and `+1`.
- Clock spins occupy one of `q` ordered angles on a circle.
- Potts variables occupy one of `q` categorical states; unlike clock states,
  Potts states do not have an intrinsic circular distance.

The `q`-state clock model approaches an equilibrium XY state space as `q`
increases. It does not by that fact become generic Kuramoto dynamics, which
also depend on natural-frequency heterogeneity and nonequilibrium phase
evolution.

### Carrier, pinned node, and pacemaker

**Carrier** is the lab's project term for a node that holds a reference state.
Use a more precise term beside it:

| implementation | preferred term | meaning |
|---|---|---|
| state fixed exactly | clamped node or pinned node | an external boundary condition |
| node receives finite reference feedback | controlled node or leader node | finite-strength pinning control |
| oscillatory node supplies phase/frequency reference | pacemaker | a dynamical reference source |
| spin has a strong but finite local bias | biased seed | stable with high probability, not fixed |

A hard clamp is not evidence that a node is autonomously stable. It is an
externally maintained condition.

## 2. The experimental regimes

### A. Metastable reversal

Protocol:

1. Initialize the free system in an ordered old state.
2. Hold carriers at the old state during phase A.
3. Switch carriers to the new state.
4. Observe whether the free system escapes the old basin within a fixed
   horizon.

Appropriate language:

- metastable phase reversal;
- pinning-induced nucleation;
- domain growth;
- kinetic escape;
- first-passage time or escape probability.

The threshold is kinetic. It is not an equilibrium material constant.

### B. Ordering from disorder

Protocol:

1. Initialize spins randomly.
2. Operate near or above the unforced ordering crossover.
3. Hold a subset of nodes at a reference state.
4. Measure induced order and target alignment.

Appropriate language:

- pinning response;
- symmetry breaking by a reference boundary;
- induced magnetization;
- finite-correlation-length influence;
- spatial or graph coverage.

This is the regime closest to the motivating phrase “seeds organize an
incoherent population,” but the current seeds are hard clamps and often have
amplified incident coupling.

### C. Associative recall

Protocol:

1. Store one or more patterns in pairwise couplings.
2. Initialize the system away from a stored pattern.
3. Clamp a partial cue.
4. Measure overlap with the cued pattern and competing patterns.

Appropriate language:

- associative recall;
- content-addressable memory;
- basin completion;
- storage load and cross-talk;
- mixture or glassy state.

For one binary pattern, the gauge transformation
`sigma_i = s_i * target_i` maps the structured system to a ferromagnet. Recall
of one pattern is therefore a valuable regression check, not independent
evidence of a new phase mechanism.

### D. Competing pinning control

Protocol:

1. Start from a target-aligned ordered state.
2. Pin one set of nodes to the target and another set to an anti-target.
3. Measure alignment, local order, and fragmentation.

Appropriate language:

- competing boundary conditions;
- adversarial pinning;
- takeover or loss of target alignment;
- fragmented or contested state.

Security and organizational interpretations are analogies unless separately
validated in a real system.

## 3. Measurements

No single score means “stability.” State which property is being measured.

### Global phase coherence

For phases,

\[
r(t)=\left|\frac{1}{N}\sum_i e^{\mathrm{i}\theta_i(t)}\right|.
\]

`r` measures concentration of phases. It does not identify the phase around
which the population concentrates.

### Target-phase alignment

For reference phase `phi`,

\[
m_\phi(t)=\operatorname{Re}\left[
e^{-\mathrm{i}\phi}\frac{1}{N}\sum_i e^{\mathrm{i}\theta_i(t)}
\right].
\]

This distinguishes coherent-on-target from coherent-off-target states.

### Frequency synchronization

Phase coherence and frequency synchronization are different. Report a
long-window dispersion of mean frequencies, a locked fraction, or both:

\[
\Omega_i = \frac{\theta_i(t_1)-\theta_i(t_0)}{t_1-t_0}.
\]

### Binary target overlap

For spins and target `t_i`,

\[
m_t = \frac{1}{N_f}\sum_{i\in\mathrm{free}}s_i t_i.
\]

The code calls this `fid`. It is target-relative and ranges from `-1` to `+1`.

### Magnetization

\[
m = \frac{1}{N_f}\sum_{i\in\mathrm{free}}s_i.
\]

`|m|` detects uniform binary order. It is not a general coherence measure for
balanced structured patterns.

### Local correlation and bond satisfaction

For ferromagnetic edges, mean `s_i s_j` measures local agreement. For signed or
matrix-valued couplings, use an explicitly defined bond-satisfaction or energy
measure.

Bond satisfaction is target-blind. In a single binary Hopfield pattern, both
the pattern and its global inverse satisfy the same pairwise signs. Structural
consistency therefore cannot certify target identity.

### Dynamical measurements

When the claim concerns stabilization or escape, include at least one temporal
quantity:

- time to first enter the target region;
- probability of entry within the observation horizon;
- residence time after entry;
- recovery time after perturbation;
- survival of order after control is removed;
- integrated autocorrelation time or effective sample size for stochastic
  traces.

## 4. Thresholds and criticality

### Operational threshold

Use **operational threshold** when a parameter is defined by crossing a chosen
score within a finite protocol. Write it as a conditional quantity:

\[
h_c = h_c(N, G, T, J, \kappa, t_{\mathrm{obs}}, c, \text{placement}),
\]

where `c` is the crossing criterion.

Every threshold report should include:

- system size and topology;
- coupling normalization;
- temperature or inverse temperature;
- clamp or pacemaker strength;
- initial condition;
- placement algorithm;
- score and crossing criterion;
- burn-in, observation horizon, and update rule;
- number and type of independent replicates.

### Pseudocritical temperature

For a finite graph, a magnetization crossing or susceptibility maximum defines
a **pseudocritical temperature** or **ordering crossover**. Use
“critical temperature” only when the thermodynamic interpretation is supported
by an appropriate finite-size analysis and model family.

### Kinetic versus equilibrium statements

- “Flips within 150 sweeps” is a kinetic statement.
- “The measured distribution is stationary under longer burn-in and multiple
  chains” is evidence toward an equilibrium statement.
- A flat result at three burn-in lengths is useful, but does not prove exact
  equilibrium or mixing.

At positive temperature, a finite irreducible spin chain can eventually cross
between basins even without pins. The meaningful object is the change in escape
time or probability, not the word “eventually.”

## 5. Control strength and topology

### Separate number from authority

The NumPy experiments multiply clamp-incident edges by `B`. The control input
therefore has two independent quantities:

- pinning fraction `h`;
- incident control amplification `B`.

Holding `B*J/T` fixed across temperatures is a legitimate protocol choice, but
it does not isolate the effect of carrier count. A control study should sweep
`h` and `B` independently and include `B=1`.

### Coupling normalization

On heterogeneous graphs, unnormalized coupling gives a hub greater total input
and output simply because it has more neighbors. Compare at least:

- edge coupling `K A_ij`;
- degree-normalized coupling `K A_ij/k_i`;
- optionally symmetric normalization `K A_ij/sqrt(k_i k_j)`.

If the placement advantage changes under normalization, part of the result is
degree-amplified control authority rather than topology alone.

### Placement algorithms

Current baselines:

- random placement;
- highest degree;
- static radius-2 collective influence.

Important additions:

- evenly spaced or farthest-point placement;
- graph `k`-center or dominating-set approximations;
- grounded-Laplacian or other spectral pinning criteria;
- adaptive collective influence with rescoring after each selection.

Degree ties must not silently resolve by node label when spatial dispersion is
part of the phenomenon.

## 6. Evidence levels

Use these labels in result summaries:

| level | meaning |
|---|---|
| implementation check | expected identity or regression gate reproduced |
| observed | present in the recorded finite protocol |
| ensemble-supported | repeated over stated graph, pattern, and dynamics seeds |
| mechanism hypothesis | plausible explanation requiring a discriminating test |
| scaling claim | supported by fitted scaling, uncertainty, and uncensored range |
| cross-substrate hypothesis | qualitative agreement observed across model classes |

Small standard deviations across three seeds do not by themselves establish a
scaling law. Prefer paired analyses when the same graph or random seed is used
across conditions, and report confidence intervals or the individual replicate
values when ensembles are small.

## 7. Direct oscillator experiment

The next experiment that directly matches the motivating question should:

1. initialize phases uniformly on `[0, 2*pi)`;
2. draw heterogeneous natural frequencies from a declared distribution;
3. verify that the no-control population remains incoherent at the chosen `K`;
4. give selected nodes finite pacemaker coupling `kappa`, not only hard clamps;
5. sweep pinning fraction and `kappa` independently;
6. compare random, degree, coverage, and spectral placement;
7. run both normalized and unnormalized network coupling;
8. measure phase coherence, target alignment, frequency locking, and settling
   time;
9. remove the pacemaker input after convergence and measure persistence;
10. repeat over graph, natural-frequency, placement, initial-condition, and
    noise realizations at several system sizes.

If order requires the control to remain active, call the result **pinning
synchronization** or **entrainment**. If order persists after removal, call it
**basin capture**, **hysteresis**, or **self-sustained synchronization**, as
supported by the observed dynamics.

## References

- Dörfler, F. and Bullo, F., [On the Critical Coupling for Kuramoto
  Oscillators](https://arxiv.org/abs/1011.3878) — distinguishes several notions
  of synchronization and phase cohesiveness.
- Sorrentino, F. et al., [Pinning-controllability of complex
  networks](https://arxiv.org/abs/cond-mat/0701073) — control of network
  synchronization through a selected subset of nodes.
- Wang, Y. and Doyle, F., [Exponential synchronization rate of Kuramoto
  oscillators in the presence of a pacemaker](https://arxiv.org/abs/1209.0811)
  — pacemaker terminology and finite reference strength.
- Morone, F. and Makse, H., [Influence maximization in complex networks through
  optimal percolation](https://www.nature.com/articles/nature14604) — the
  original scope of Collective Influence.
- Hopfield, J., [Neural networks and physical systems with emergent collective
  computational abilities](https://pubmed.ncbi.nlm.nih.gov/6953413/) —
  content-addressable binary associative memory.
- Kanter, I., [Potts-glass models of neural
  networks](https://pubmed.ncbi.nlm.nih.gov/9900002/) — multi-state associative
  memory and its dense-model capacity result.
- [THRML documentation](https://docs.thrml.ai/) — the block-Gibbs and energy
  model API used by the core implementation.
