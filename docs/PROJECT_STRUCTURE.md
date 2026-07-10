# Project structure roadmap

The current repository is appropriate for an exploratory lab: scripts are easy
to run, evidence is visible, and the chronological path is preserved. Its main
structural limitation is that experiment logic, reusable models, configuration,
analysis, and narrative are not yet cleanly separated. The root-level
`run_*.py` scripts and the large `dtm_col_infl_clamping.py` file reflect the
order of discovery more than the current research model.

The recommended change is incremental. Do not move every script at once or
break the existing reproduction commands.

## Current strengths

- `src/carrier_pgm/` already separates the core THRML model from entry points.
- Each major result has a runnable script and a CSV where applicable.
- Failed hypotheses and interpretation changes remain visible.
- `README.md`, `RESULTS.md`, and `docs/CONCEPTS.md` now serve distinct purposes.

## Current friction

- Experiment families are implicit in script names such as `run_fss.py` and
  historical labels such as H7/H8/H9.
- The NumPy/Glauber, Hopfield, clock, and Potts implementation lives in one
  notebook-style module.
- Configuration is embedded as module constants, so provenance requires reading
  code at the matching commit.
- Randomness is sometimes shared across graph generation, placement, patterns,
  and dynamics rather than recorded as distinct inputs.
- Several console-only experiments lack a durable machine-readable result.
- There is no automated regression suite for the mathematical identities that
  currently act as manual gates.
- `data/` mixes durable evidence with generated presentation artifacts and does
  not include a manifest per run.

## Target structure

```text
thrml-kuramoto-lab/
├── README.md
├── RESULTS.md
├── pyproject.toml
├── src/
│   └── thrml_lab/
│       ├── models/
│       │   ├── ising.py
│       │   ├── clock.py
│       │   ├── hopfield.py
│       │   └── potts.py
│       ├── protocols/
│       │   ├── ordering.py
│       │   ├── reversal.py
│       │   ├── recall.py
│       │   └── competing_control.py
│       ├── graphs/
│       │   ├── topology.py
│       │   └── placement.py
│       ├── metrics/
│       │   ├── alignment.py
│       │   ├── structure.py
│       │   └── kinetics.py
│       └── analysis/
│           ├── thresholds.py
│           ├── ensembles.py
│           └── scaling.py
├── experiments/
│   ├── alignment/
│   ├── adaptation/
│   ├── memory/
│   ├── capacity/
│   └── cross_model/
├── configs/
│   ├── baselines/
│   └── published/
├── data/
│   ├── published/
│   └── README.md
├── artifacts/                 # ignored generated plots and ad hoc run output
├── tests/
│   ├── unit/
│   ├── regression/
│   └── smoke/
├── docs/
├── specs/
└── integrations/
```

The exact package name is less important than the boundaries: models describe
dynamics, protocols describe initial conditions and interventions, metrics
describe observation, and experiment definitions compose those pieces without
reimplementing them.

## Experiment manifests

Each durable run should produce or be associated with a manifest such as:

```yaml
experiment_id: A3-metastable-reversal
code_commit: ac97342
model: ising
protocol: reversal
topology:
  kind: ring
  n: 120
  parameters:
    horizon: 6
control:
  placement: random
  fraction_grid: [0.0, 0.01, 0.02, 0.03]
  strength: 1.0
dynamics:
  beta: 1.0
  coupling: 0.15
  update_rule: block_gibbs
  lock_steps: 100
  observation_steps: 300
seeds:
  graph: [1, 3, 7]
  placement: [1, 3, 7]
  dynamics: [1, 3, 7]
metrics:
  - local_correlation
  - target_overlap
criterion:
  metric: target_overlap
  crossing: 0.6
outputs:
  - data/published/A3-metastable-reversal/results.csv
```

The manifest should be copied into the output directory so a CSV never becomes
detached from its assumptions.

## Recommended migration phases

### Phase 1 — navigation and provenance

Status: started by the current documentation work.

- Keep all current run commands working.
- Use `docs/EXPERIMENT_CATALOG.md` as the stable experiment registry.
- Add a short `data/README.md` that identifies which files are published
  evidence and which scripts generate them.
- Add manifests for new experiments before backfilling older runs.
- Give new experiments semantic IDs such as `A3` or `B2`; retain H labels only
  as historical aliases.

### Phase 2 — extract reusable scientific components

- Move graph construction and placement into a shared package.
- Extract Glauber, clock, Hopfield, and Potts dynamics from
  `dtm_col_infl_clamping.py` without changing their numerical behavior.
- Extract metrics so every protocol uses the same target-overlap,
  autocorrelation, bond-satisfaction, and first-passage definitions.
- Leave thin compatibility wrappers at the current `run_*.py` paths.

### Phase 3 — configuration-driven execution

- Introduce one experiment runner that accepts a versioned YAML configuration.
- Record separate seeds for graph, placement, target/pattern, initial state, and
  dynamics.
- Write tidy per-replicate rows before aggregation; derive means and plots in a
  separate analysis step.
- Treat threshold crossings outside a scanned grid as censored observations,
  not exact endpoints.

### Phase 4 — regression and statistical evidence

Add automated tests for:

- deterministic graph edge counts and topology invariants;
- graph coloring independence within Gibbs blocks;
- exact `q=2` clock/Ising equivalence under matched randomness;
- single-pattern Hopfield/ferromagnet gauge equivalence;
- target inversion and bond-satisfaction ambiguity;
- fixed-seed reproducibility;
- no-carrier and all-carrier boundary conditions;
- threshold interpolation and censored-grid behavior;
- schema validation for result manifests.

Then add analysis support for confidence intervals, paired condition
comparisons, first-passage distributions, and finite-size scaling fits.

## What not to do yet

- Do not introduce a workflow engine or database merely to organize a small
  number of local simulations.
- Do not rewrite the working THRML protocol while extracting the NumPy models.
- Do not move historical notes into the evidence path.
- Do not hide failed hypotheses; mark them superseded in the catalog and
  preserve their original record.
- Do not make the architectural metaphor a dependency of the numerical code.
  The simulation should remain interpretable as physics/network research on
  its own terms.

## Immediate next structural change

The best next implementation-sized refactor is:

1. add `src/thrml_lab/graphs.py`, `placement.py`, and `metrics.py` from the
   duplicated stable functions;
2. add regression tests for those functions;
3. leave existing scripts as wrappers;
4. implement the direct Kuramoto pacemaker experiment against those shared
   graph, placement, and metric interfaces.

That sequence improves scientific comparability before introducing a larger
runner abstraction.
