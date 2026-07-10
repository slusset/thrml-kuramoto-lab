# Research program: coherent domains in a shared field

## Why this lab exists

This lab began with an architectural intuition inspired by a Dyson swarm: a
large system composed of many autonomous units, each capable of maintaining
itself locally, interacting through a shared environment, and contributing to
a larger purpose without requiring a single controller to dictate every local
action.

The software-architecture analogue is a **sovereign domain**: a bounded context
that owns its model, execution, data lifecycle, infrastructure declarations,
clinical reasoning within its declared scope, operational health, and boundary
contracts. Sovereignty here does not mean isolation, opposition, or exemption
from shared obligations. It means local authority paired with explicit
accountability to the surrounding clinical and organizational environment.

The central question is therefore larger than synchronization:

> How can locally coherent, independently operable domains remain aligned with
> a purpose that is defined, observed, and sometimes changed at a larger scale?

The oscillator and spin experiments are deliberately simplified probes of
that question. They are not evidence that organizations or clinical systems
literally behave like Ising spins. Their value is to make architectural
assumptions explicit, create counterexamples, and identify design tradeoffs
that can then be validated in real systems.

## Homeostasis and telos are different measurements

The lab separates two properties that are often collapsed into one:

- **Homeostasis or internal coherence**: does the node or network remain
  internally consistent, stable, and capable of acting according to its own
  declared model?
- **Telos alignment**: is that coherent action oriented toward the currently
  intended external or global outcome?

Internal coherence is target-blind. A system can be beautifully consistent and
still serve an obsolete, captured, or locally convenient purpose. External
alignment without internal coherence is also insufficient: the system may be
correct momentarily but unable to sustain or explain its behavior.

| | externally aligned | externally misaligned |
|---|---|---|
| **internally coherent** | healthy autonomy: stable and serving the intended outcome | frozen or captured: consistent, efficient, and wrong |
| **internally incoherent** | fragile alignment: correct by accident or constant intervention | disordered failure: neither stable nor purposeful |

This is why the experiments retain separate target-blind and target-relative
metrics. The distinction is the architectural thesis, not merely a plotting
choice.

## What “sovereign domain” means

A sovereign domain should be able to declare and operate the following within
its bounded context:

- its ubiquitous language and domain model;
- its clinical or business decision logic within an explicitly governed scope;
- its data ownership, retention, provenance, and quality rules;
- its APIs, events, and compatibility commitments;
- its deployable lifecycle and infrastructure-as-code declarations;
- its observability, SLOs, failure modes, and recovery procedures;
- its local tests and evidence that declared behavior is actually delivered.

Sovereignty does **not** imply that every domain owns dedicated physical
infrastructure or invents its own enterprise standards. A shared platform may
provide clusters, identity, networking, secrets, policy enforcement, and
telemetry transport. The domain remains sovereign when it controls its
declared configuration and lifecycle on that substrate and is not coupled to
another domain's internal implementation.

In a clinical setting, local ownership also does not mean unilateral ownership
of medical truth. Clinical reasoning must remain bounded by evidence,
regulation, safety governance, shared semantic agreements, human authority,
and measured outcomes. The domain owns implementation and accountable
execution within that frame.

## The correspondence under exploration

| model concept | architectural interpretation | important limit |
|---|---|---|
| node | bounded context, service, team, or locally accountable capability | these are not interchangeable in every analysis |
| internal coupling | local model consistency, tests, workflows, and reinforcing dependencies | strong coupling can mean either useful cohesion or brittleness |
| graph edge | API, event, data, policy, or human coordination relationship | real interfaces are typed, directional, and governed |
| pinned carrier | reference implementation, policy anchor, exemplar domain, or trusted decision source | a hard clamp is stronger than ordinary influence |
| pacemaker | finite-strength source of shared timing, intent, or policy | the reference itself can be wrong or stale |
| target | declared outcome, clinical intent, policy, or environmental fitness criterion | a real telos is plural, negotiated, and time-varying |
| target-blind coherence | internal health, consistency, contract satisfaction, local SLOs | cannot prove outcome alignment |
| target overlap | measured conformance to a declared external objective | depends on the legitimacy and observability of the target |
| temperature/noise | uncertainty, local variation, operational disturbance, or freedom to explore | no single organizational variable corresponds exactly |
| topology | distribution of influence and dependency across domains | organizational graphs are multiplex and change over time |

The correspondence is a thinking instrument. Any architectural recommendation
must be restated in ordinary software and clinical terms and validated outside
the simulation.

## Why the experiments follow this sequence

### 1. Metastable reversal: can a coherent domain change direction?

The first protocol starts with an ordered system serving an old target and asks
whether sparse reference nodes can redirect it within a finite horizon.

Architectural question:

> Does strong internal coherence make a domain resilient, or does it make the
> domain unable to respond when its purpose changes?

Possible design implications:

- optimize for recoverable coherence, not maximum lock;
- preserve migration seams, rollback, feature flags, and versioned policies;
- measure time-to-adapt after a decision or environmental change;
- test whether a domain can replace a formerly correct invariant.

### 2. Ordering from disorder: how does a new capability acquire direction?

This protocol starts without stable global order and introduces sparse
reference nodes.

Architectural question:

> Can shared intent propagate through examples, interfaces, and local
> interaction without prescribing every internal implementation?

Possible design implications:

- prefer reference implementations and executable contracts over prose-only
  mandates;
- place alignment mechanisms where they cover important interfaces;
- separate the number of reference points from their authority or enforcement
  strength;
- verify that apparent order does not arise without the proposed intervention.

### 3. Placement and topology: where should alignment capability live?

The placement experiments compare random, high-degree, and influence-based
control sets.

Architectural question:

> Which integration points can spread alignment efficiently, and when does
> concentrating influence create a fragile choke point?

Possible design implications:

- invest in high-leverage interfaces such as identity, terminology, event
  envelopes, and clinical-policy distribution;
- harden and audit these interfaces because influence and capture risk share
  the same topology;
- on homogeneous networks, optimize coverage and dispersion rather than
  centrality;
- avoid translating “hub placement worked” into “centralize command.” A hub
  may distribute a standard while local domains retain implementation
  authority.

### 4. Associative recall: can relationships carry domain identity?

The Hopfield experiments store a structured pattern in pairwise couplings and
ask whether a partial cue reconstructs it.

Architectural question:

> Can a domain's identity be recovered from coherent relationships and
> contracts rather than from a central copy of its entire state?

Possible design implications:

- make boundary agreements executable and locally observable;
- treat contract conformance as evidence of relational integrity;
- retain an independent target or provenance check, because internally
  satisfied relationships can support the global inverse or another wrong
  interpretation.

### 5. Capacity: how much responsibility can one context coherently hold?

The multi-pattern experiments increase the number of intents written into a
fixed pairwise substrate.

Architectural question:

> When does a bounded context become overloaded by too many meanings,
> workflows, policies, or cross-domain obligations?

Possible design implications:

- treat context boundaries as a capacity decision, not only a naming decision;
- split a context when new responsibilities create conflicting invariants;
- monitor contradiction and unresolved policy interference at design time;
- increase interaction richness or domain separation rather than assuming a
  smarter implementation can remove structural overload.

### 6. State granularity: does binary governance make change expensive?

The clock and Potts experiments vary how many states a node or relationship can
represent.

Architectural question:

> Do coarse pass/fail states create unnecessary barriers where graded,
> versioned, or explicitly transitional states would support safe adaptation?

Possible design implications:

- represent migration, uncertainty, and partial compatibility explicitly;
- avoid forcing every cross-domain relationship into compliant/noncompliant;
- use richer contracts only where teams can still understand and operate them;
- test the cost of changing a policy, not only the fidelity with which it is
  enforced.

### 7. Competing control: who can change the telos?

The infiltration protocol gives different pinned sets conflicting targets.

Architectural question:

> How does a domain distinguish legitimate change from capture, stale policy,
> or incompatible authorities?

Possible design implications:

- version and sign policy or clinical-decision artifacts;
- make authority, provenance, and effective dates observable;
- define conflict-resolution and safe-degradation behavior before conflict;
- monitor target changes as events, not only steady-state service health.

## Architecture decisions the lab can inform

The simulations can support hypotheses and tradeoff discussions around these
decisions:

| decision area | evidence the lab can contribute | evidence still required in practice |
|---|---|---|
| domain autonomy | shows why local coherence and global alignment need separate controls | team cognitive load, deployment independence, incident history |
| vertical ownership | motivates domain-owned lifecycle and observability | platform capabilities, security model, operating cost |
| integration style | compares local coupling, coverage, and high-leverage reference points | API/event latency, failure propagation, semantic compatibility |
| governance | supports outcome-and-interface constraints over internal micromanagement | regulatory, clinical, security, and organizational authority |
| observability | demonstrates that internal green signals can coexist with target failure | real clinical/process outcome measures and alert validation |
| bounded-context size | provides a structural overload analogy | domain event analysis, policy conflicts, change coupling, team topology |
| change strategy | exposes the tradeoff between deep stability and redirectability | migration lead time, rollback success, adoption and safety data |
| shared standards | suggests strategic, trusted reference points rather than universal hard clamps | conformance tests, exception governance, version compatibility |

The lab should not be used to justify a precise carrier percentage, a
scale-free enterprise topology, or removal of human and clinical governance.
Its legitimate contribution is a disciplined vocabulary, counterexamples to
naive centralization or coherence claims, and experimentally generated
hypotheses for architecture evaluation.

## A practical sovereign-domain pattern

A domain shaped by this research would expose three distinct surfaces:

1. **Local control surface** — domain model, reasoning, persistence,
   deployment, SLOs, and recovery owned by the domain team.
2. **Boundary surface** — versioned APIs/events, semantic contracts, security
   policy, compatibility tests, and explicit failure behavior.
3. **Outcome surface** — measures that connect local operation to the external
   clinical or organizational purpose, including drift and unintended effects.

Enterprise governance supplies shared constraints and observes the boundary
and outcome surfaces. It should not need to reach through those surfaces to
micromanage the domain's internal model. Conversely, the domain cannot claim
success from internal coherence alone.

## Research discipline

For every architectural interpretation:

1. name the exact experiment and measured result;
2. state whether the result is observed, ensemble-supported, or hypothetical;
3. translate it into ordinary architecture language;
4. name where the analogy can fail;
5. identify the real-world evidence that would accept or reject the decision;
6. prefer a reversible pilot over an enterprise-wide conclusion.

The goal is not to persuade by metaphor. It is to make the proposed operating
model visible through interfaces, tests, telemetry, and small working systems.
