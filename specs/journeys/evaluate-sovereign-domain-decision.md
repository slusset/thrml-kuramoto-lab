---
id: evaluate-sovereign-domain-decision
type: journey
refs:
  persona: specs/personas/sovereign-domain-architect.md
---

# Journey: Evaluate a Sovereign-Domain Architecture Decision

## Actor

The sovereign domain architect is considering a design that gives a bounded
context greater control over its infrastructure, reasoning, data, and operating
lifecycle while retaining explicit external obligations.

Source Persona: `specs/personas/sovereign-domain-architect.md`

## Trigger

A proposed boundary, integration, governance, or ownership decision encounters
uncertainty or resistance and needs a clearer evidence-based rationale.

## Preconditions

- The domain's scope and intended external outcome can be stated.
- The proposed decision is reversible or can be piloted at bounded scale.
- Simulation results will be treated as hypothesis-generating evidence, not as
  direct proof about clinical or organizational behavior.

## Flow

### 1. Declare local identity and external telos

- **User intent**: State what the domain owns and what outcome or constraint it
  remains accountable to.
- **System response**: The decision record separates local coherence measures
  from external outcome-alignment measures.
- **Next**: Select the relevant experiment family.

### 2. Select the model analogy

- **User intent**: Identify whether the decision concerns acquisition of
  alignment, adaptation, placement, conflicting authority, relational memory,
  or capacity.
- **System response**: The experiment catalog identifies the closest protocol,
  measured result, evidence level, and known limit.
- **Next**: Translate the result into ordinary architecture language.

### 3. Bound the inference

- **User intent**: Avoid turning a model result into a universal architecture
  claim.
- **System response**: The decision record names where the analogy can fail and
  which real-world evidence is still missing.
- **Next**: Formulate a reversible architecture hypothesis.

### 4. Design a pilot

- **User intent**: Show the architecture through a working boundary and
  observable behavior.
- **System response**: The proposal specifies interfaces, ownership, contract
  tests, internal SLOs, external outcome measures, failure behavior, and a
  rollback path.
- **Next**: Operate and observe the pilot.

### 5. Integrate evidence

- **User intent**: Decide whether to retain, revise, or reject the proposed
  architecture.
- **System response**: Simulation rationale and operational evidence remain
  distinguishable and traceable in the decision record.
- **Next**: Share the bounded decision and its evidence.

## Outcomes

- **Success**: The architecture decision is expressed as a bounded,
  evidence-informed, reversible hypothesis with separate local-health and
  external-outcome measures.
- **Failure modes**: The metaphor substitutes for evidence; sovereignty is used
  to avoid shared obligations; centralized governance reaches through the
  boundary without a declared contract; or internal coherence is mistaken for
  outcome success.

## Related Stories

- `specs/stories/research/trace-experiment-to-decision.md`

## E2E Coverage

- Documentation traceability only; no automated E2E test yet.

