# thrml-kuramoto-lab

An experimental lab for studying how a sparse set of reference nodes can
induce, redirect, or recall collective order in networked dynamical systems.

The original experiment translated a Kuramoto-inspired carrier-fraction model
into a clamped Ising probabilistic graphical model in
[THRML](https://docs.thrml.ai). The lab now contains several related but
distinct protocols: metastable reversal, ordering from disorder, associative
recall, storage-capacity experiments, and competing pinning controls.

The project term **carrier** means a node that holds a reference state. In the
implemented spin models carriers are externally **clamped** or **pinned**; they
are not ordinary nodes that merely happen to be dynamically stable.

## Research question

The broad question is:

> Under what conditions can a sparse set of reference nodes produce
> target-aligned collective order, and how do topology, control strength,
> temperature, state-space granularity, and placement change the result?

The cleanest form of the motivating question—whether a few autonomous
pacemaker nodes can synchronize an initially incoherent Kuramoto population—
remains a next experiment. Most current results concern spin and clock models
with externally pinned nodes.

## Experimental regimes

These protocols answer different questions and should not share an
unqualified threshold.

| regime | initial condition | intervention | standard interpretation |
|---|---|---|---|
| metastable reversal | ordered in an old state | pins switch to a new state | pinning-induced nucleation, kinetic escape, first-passage dynamics |
| ordering from disorder | random state near or above an ordering crossover | fixed reference nodes | symmetry breaking, pinning response, influence coverage |
| associative recall | random state with learned or Hebbian couplings | partial pattern cue | basin completion, recall fidelity, interference |
| competing control | ordered state | loyal and adversarial pins disagree | competing boundary conditions, takeover, fragmentation |
| capacity | multiple patterns stored in pairwise couplings | one pattern is partially cued | storage load, cross-talk, mixture or glassy states |

See [Concepts and methods](docs/CONCEPTS.md) for definitions of the models,
metrics, thresholds, and terminology.

## Current findings

The current evidence supports the following bounded conclusions:

1. In the finite-horizon Ising reversal protocol, a sparse pinned set can
   redirect an ordered state. The observed carrier threshold is an
   **operational first-passage threshold**, conditional on system size,
   coupling, temperature, placement, and observation horizon.
2. In heterogeneous graphs, high-degree placement strongly outperforms random
   placement in the tested regimes. On homogeneous graphs, spatial coverage
   matters and degree ranking can produce a pathologically clumped control
   set. Degree is therefore a useful heuristic, not a universal placement law.
3. Above the unforced ordering crossover, the random-placement shoulder is
   consistent with finite-range influence coverage: unrecalled nodes tend to
   be low-degree and farther from a pinned node.
4. A single Hopfield pattern can be recalled from a partial cue, while ordinary
   magnetization remains near zero for balanced patterns. Local bond
   satisfaction measures structural consistency, but cannot by itself certify
   which of a pattern and its global inverse was recalled.
5. Finer clock states reduce kinetic escape costs in the tested heat-bath
   model, while Potts-valued pairwise couplings increase associative capacity.
   These are results about discrete equilibrium-model dynamics; they are not
   yet a direct demonstration for generic Kuramoto dynamics.
6. Capacity grows with graph degree and state richness within the measured
   protocol. Exact scaling laws and dense-theory comparisons remain
   provisional because the runs use finite graphs, finite temperature,
   external cues, coarse grids, and small ensembles.

Numbers, conditions, confidence levels, and open falsification tests are in
[RESULTS.md](RESULTS.md).

## Setup

Python 3.10 or newer is required.

```bash
uv venv
uv pip install -e ".[notebook]"
```

The core THRML experiment runs on CPU. JAX will use an available accelerator.

## Run the lab

Start with the smallest reproducible experiment:

```bash
uv run python run_sweep.py
```

Useful experiment groups:

```bash
# Original clamped-Ising reversal and basic rigor checks
uv run python run_sweep.py
uv run python run_rigor.py seeds
uv run python run_rigor.py equil
uv run python run_rigor.py size
uv run python run_master.py --bj 0.10 0.15 0.20

# Ordering temperature, placement, and structured recall
uv run python run_tc.py
uv run python run_ensemble.py
uv run python run_plateau.py

# Associative capacity and learned couplings
uv run python run_capacity.py
uv run python run_learned.py
uv run python run_capacity_q.py
uv run python run_fss.py
uv run python run_density.py

# Clock-state equilibrium and kinetic reversal
uv run python run_clock.py
uv run python run_kinetic.py

# Competing pinned controls
uv run python run_contest.py
```

`run_substrate.py` also requires the sibling `homeostasis_telos` checkout at
`../homeostasis_telos`; it compares matched Kuramoto and Gibbs graph instances.

The notebook interface remains available at:

```bash
uv run jupyter lab notebooks/01_carrier_fraction_clamped_pgm.ipynb
```

## Documentation

- [RESULTS.md](RESULTS.md) — current integrated findings, limitations, and next
  falsification tests.
- [Concepts and methods](docs/CONCEPTS.md) — canonical terminology, protocol
  definitions, metrics, and claim boundaries.
- [Experiment log](docs/EXPERIMENT_LOG.md) — the original chronological README
  narrative, preserved as the detailed lab record.
- [`integrations/`](integrations/) — reflective and cross-project integration
  notes; these are context, not numerical evidence.

## Repository layout

```text
├── src/carrier_pgm/
│   ├── model.py          # THRML Ising graph, pinned blocks, placement
│   ├── experiment.py     # two-phase metastable reversal protocol
│   ├── infiltration.py   # competing loyal/adversarial pins
│   └── topology.py       # ring, small-world, preferential-attachment, ER
├── dtm_col_infl_clamping.py  # NumPy/NetworkX Glauber, clock, Hopfield, Potts
├── run_*.py                  # reproducible experiment entry points
├── data/                     # generated CSV evidence and master-curve plot
├── notebooks/                # exploratory notebook interface
├── docs/                     # concepts and chronological experiment record
└── integrations/             # reflective integration notes
```

## Evidence conventions

- Always report an operational threshold with its system size, temperature or
  inverse temperature, control strength, placement rule, criterion, and time
  horizon.
- Use **pseudocritical temperature** or **ordering crossover** for a finite graph
  unless a finite-size scaling analysis supports a thermodynamic transition.
- Keep target-blind organization separate from target-relative alignment.
- Treat security, organizational, and teleological readings as interpretations
  of the model, not direct empirical claims about real systems.
- Preserve failed hypotheses and superseded mechanisms in the experiment log;
  keep only current bounded claims in the README and results summary.

## Next decisive experiment

Implement a networked Kuramoto protocol that starts from uniformly random
phases, includes heterogeneous natural frequencies, and gives seed nodes a
finite-strength pacemaker coupling rather than a hard clamp. Sweep seed
fraction and seed strength independently, then remove the forcing to distinguish
sustained pinning synchronization from self-sustaining basin capture.
