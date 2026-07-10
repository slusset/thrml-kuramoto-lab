# Integration packet — beat of 2026-07-04

*Raw materials for the one-page note. Not conclusions. You write; I object.*
*Genre exemplar: RESULTS.md — every number bounded, falsification section at end.*

## A. Findings landed since the last integration note

RESULTS.md ends at the previous arc (findings 1–6, oscillators→samplers). Everything
below lives only in README prose + CSVs — nothing yet written to be defended.

| # | finding (one factual line) | commit | data |
|---|---|---|---|
| 1 | Per-topology Tc located (2.5/4.0/6.0 lattice/SW/SF at N=196); prior sweep was sub-Tc → saturation; recalibrated: placement separates on SF, CI=degree again, random wins on lattice (degree-rank clumps) | ddbf4eb | run_tc.py, ensemble_placement.csv |
| 2 | Hopfield recall (J_ij=t_i·t_j): fid 0.95±0.02 at 16% degree carriers; r_mag ≈ 0 (consensus monitor blind); r_struct tracks fid within ~0.01 | 3b02393, 6b56210 | ensemble_hopfield.csv |
| 3 | Random-placement "plateau" = equilibrium coverage law: not kinetic (burn 40/160/640 flat), wrong nodes are low-degree periphery ≥2 hops from carriers; gauge check Hopfield≡ferromagnet | 805af03 | run_plateau.py output |
| 4 | Multi-pattern (Hebb, q=2): knee at P=2; at overload the mesh satisfies MORE couplings than perfect recall would (0.36 vs 0.23 ceiling at P=12) while cued fid collapses — mixture state, not rival-memory takeover | 458d52a | ensemble_capacity.csv |
| 5 | Learned couplings (THRML max-likelihood): fid 1.00 at 2% carriers vs Hebb 0.73 — via valley depth (mean|w| 3.9 vs 1.0); capacity knee unmoved; training moment-gap = built-in overload detector (0.014 at P=1, stuck 0.2–0.4 at P≥2) | 49dae57 | ensemble_learned.csv |
| 6 | Equilibrium placement gap is q-flat (clock model, q=2..16, per-q Tc) — "softness extends influence radius" falsified in coverage regime | ed6f158 | ensemble_clock.csv |
| 7 | Kinetic escape threshold steps 6× between q=4 and q=8 (h_c random 0.12→0.02, pinned clamps, 0.6·Tc, horizon 150): nucleation→rotation crossover; first run floored at 0.75·Tc with amplified clamps (methods lesson) | f4c3afd | ensemble_kinetic.csv |
| 8 | Potts q×q covenants: capacity knee 1→5→8-16 across q=2/4/8 at ⟨k⟩≈6 | efcee04 | ensemble_capacity_q.csv |
| 9 | Knee is N-flat (196→800, fixed density) — undershoot of dense theory is not finite-size | f1a2deb | ensemble_fss.csv |
| 10 | Capacity = α(q)·⟨k⟩ across ⟨k⟩ 6→42; α(4)≈0.85 ≈ Kanter 0.83; α(8)≈2.5 ≈ 65% of dense 3.86 at EVERY density → averaging-starvation mechanism refuted; live suspect: finite-T entropy | ac97342 | ensemble_density.csv |

Charter additions (homeostasis_telos): relational finding 161ef99 · boundary +
covenant budget 0473190 · granularity e5cfe32 · H8b measured 9f9f6e6 · exponent
settled ee9748e · mechanism retracted bc2551b.

## B. Orbiting ledger — each item needs a disposition: ACT / SCHEDULE (against a named wall) / PARK (with reason)

1. **Sheaf-ADMM rebuild** with real restriction maps (lost chat's section 7; code cloned at ~/PycharmProjects/sheaf-admm, paper PDF alongside). Orbiting since 7/3.
2. **Noised-target training** — the honest depth-setter for finding 5's "4× cheaper"; flagged in README, never run.
3. **β sweep** — the entropy suspect's trial: does α(8) recover toward 3.86 cold? Named wall exists; not yet scheduled.
4. **Hansen–Ghrist note** — in your journal; not visible to this repo. (The original orbiter.)
5. **XTR-0 assessment** — throughput-not-capacity conclusion delivered; does it produce an action (partner application? THRML-only plan?) or get parked?
6. **RESULTS.md is stale** — the strongest existing integration artifact doesn't know this arc happened.
7. **Kuramoto-side closure** (old README next-step): port SW/SF topologies into homeostasis_telos/main.py; test whether Kuramoto thresholds track the PGM across topologies.
8. **The oracle's 1.9-rad shift** — kinetic dial ran the antipodal flip only; the original H8c (actual Kuramoto perturbation) never ran.

## C. Borrowed numbers — validity conditions (state them or mark UNKNOWN)

- **Kanter α_c(q) ≈ 0.138·q(q−1)/2** — derived for DENSE, homogeneous, zero-temperature Potts attractor networks with a basin criterion. Our measurements: sparse heavy-tailed graph, T = 1.15·Tc, kinetic 60-sweep readout, m>0.7 knee criterion. The α(4) exact match (0.85 vs 0.83) may be partly coincidental given four mismatched conditions — treat as suggestive, not confirmatory.
- **0.138 (Hopfield/AGS)** — same caveats; dense, zero-T.
- **"~0.14 stored intents per neighbor" (now in CHARTER.md)** — measured at q=2 only; α is strongly q-dependent (0.85 at q=4, 2.5 at q=8). See tension T1.
- **Extropic 10,000× energy claim** — vendor marketing figure, generative-workload-specific, externally unverified.
- **Morone–Makse CI** — derived for optimal network fragmentation (static percolation), not kinetic field injection; the lab falsified its placement advantage twice.

## D. Known tensions to resolve in writing

- **T1**: CHARTER.md's covenant-budget line quotes ~0.14·⟨k⟩ without its q=2 validity condition; the density run measured α(q) rising to 2.5. The budget rule needs either the condition attached or a q-aware restatement.
- **T2**: Old finding 6's slogan "the topology owns the ordering; the substrate sets the scale" is now sharpened by H8a/H9: in equilibrium the scale is q-invariant; kinetically the scale is set by state-space granularity. The two formulations should be reconciled in one sentence or the old one bounded.
- **T3**: Finding 5's "learned intent percolates 4× cheaper" carries an unstated cost: valley depth. The master curve predicts learned couplings steer dearly — unmeasured. The claim should travel with that condition until the redirect experiment runs.
