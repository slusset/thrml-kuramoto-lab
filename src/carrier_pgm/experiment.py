"""Two-phase clamped-Gibbs protocol mirroring the oracle's perturbation.

Phase A  lock: carriers clamped to the OLD target (+1). The lattice relaxes
         into a coherent state aligned with it.
Phase B  the purpose moves: clamp values flip to the NEW target (-1), chain
         continues from where phase A left off. Does the flip percolate?

Scores per recorded sample (free nodes only):
  homeostatic  = mean s_i s_j over free-free edges   (locked? target-blind)
  teleological = mean s_i * target                   (locked ON target?)
  GAP          = homeostatic - teleological
"""

import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from thrml import SamplingSchedule, sample_states

from .model import ModelBundle, RingSpec, build

OLD_TARGET = +1
NEW_TARGET = -1


@dataclass
class Result:
    spec: RingSpec
    hom_a: np.ndarray  # per-sample homeostatic score, phase A
    tel_a: np.ndarray  # per-sample teleological score vs OLD target, phase A
    hom_b: np.ndarray
    tel_b: np.ndarray  # vs NEW target -- the number that matters

    def summary(self, window: int = 50) -> tuple[float, float, float]:
        """Post-perturbation steady state, like the oracle's summarize()."""
        hom = float(self.hom_b[-window:].mean())
        tel = float(self.tel_b[-window:].mean())
        return hom, tel, hom - tel


def _spins(bits: jax.Array) -> np.ndarray:
    """bool {False,True} -> spin {-1,+1}, as numpy (n_samples, n_free)."""
    return np.asarray(bits, dtype=int) * 2 - 1


def _scores(bits, bundle: ModelBundle, target: int):
    s = _spins(bits)
    e = bundle.free_edge_pairs
    if len(e):
        hom = (s[:, e[:, 0]] * s[:, e[:, 1]]).mean(axis=1)  # never sees the target
    else:  # h so high no free-free edge remains: vacuously coherent
        hom = np.ones(s.shape[0])
    tel = (s * target).mean(axis=1)
    return hom, tel


def _clamp_state(bundle: ModelBundle, target: int) -> list:
    if not bundle.clamped_blocks:
        return []
    n_c = len(bundle.carrier_idx)
    return [jnp.full((n_c,), target == +1, dtype=bool)]


def run_experiment(
    spec: RingSpec,
    n_lock: int = 300,
    n_watch: int = 600,
    steps_per_sample: int = 1,
    key: jax.Array | None = None,
) -> Result:
    """Run both phases; every Gibbs sweep is recorded so the series are dynamics."""
    bundle = build(spec)

    if len(bundle.free_idx) == 0:  # h == 1: everyone carries the invariant;
        ones = np.ones(n_watch)    # fidelity is definitional, nothing to sample
        return Result(
            spec=spec, hom_a=np.ones(n_lock), tel_a=np.ones(n_lock),
            hom_b=ones, tel_b=ones,
        )

    key = jax.random.key(spec.seed) if key is None else key
    k_a, k_b = jax.random.split(key, 2)

    # observe each free block (to carry state into phase B) + flat readout
    observed = bundle.free_blocks + [bundle.readout_block]

    # --- Phase A: locked onto the old target (the oracle's premise: the ---
    # --- swarm ran to coherence on the original purpose before perturb) ---
    state0 = [
        jnp.full((len(b.nodes),), OLD_TARGET == +1, dtype=bool)
        for b in bundle.free_blocks
    ]
    sched_a = SamplingSchedule(
        n_warmup=0, n_samples=n_lock, steps_per_sample=steps_per_sample
    )
    out_a = sample_states(
        k_a, bundle.program, sched_a, state0,
        _clamp_state(bundle, OLD_TARGET), observed,
    )
    hom_a, tel_a = _scores(out_a[-1], bundle, OLD_TARGET)

    # --- Phase B: the purpose moves; continue from phase A's final state ---
    state_b = [blk[-1] for blk in out_a[:-1]]
    sched_b = SamplingSchedule(
        n_warmup=0, n_samples=n_watch, steps_per_sample=steps_per_sample
    )
    out_b = sample_states(
        k_b, bundle.program, sched_b, state_b,
        _clamp_state(bundle, NEW_TARGET), observed,
    )
    hom_b, tel_b = _scores(out_b[-1], bundle, NEW_TARGET)

    return Result(spec=spec, hom_a=hom_a, tel_a=tel_a, hom_b=hom_b, tel_b=tel_b)


def sweep(
    hs, base: RingSpec = RingSpec(), verbose: bool = True, **run_kwargs
) -> list[tuple[float, float, float, float]]:
    """h-sweep printing the oracle's table. Returns (h, hom, tel, gap) rows."""
    rows = []
    if verbose:
        print(f"{'h':>5} {'homeostatic':>12} {'teleological':>13} {'GAP':>8}   regime")
        print("-" * 60)
    for h in hs:
        spec = dataclasses.replace(base, h=float(h))
        hom, tel, gap = run_experiment(spec, **run_kwargs).summary()
        rows.append((float(h), hom, tel, gap))
        if verbose:
            regime = (
                "FROZEN: locked, off-target" if gap > 0.5
                else "EDGE: self-corrects" if tel > 0.6
                else "transitional"
            )
            print(f"{h:>5.2f} {hom:>12.3f} {tel:>13.3f} {gap:>8.3f}   {regime}")
    return rows


def critical_h(rows, cross: float = 0.6) -> float | None:
    """First h where teleological fidelity crosses `cross` (oracle's scan)."""
    prev = None
    for h, _, tel, _ in rows:
        if prev is not None and prev < cross <= tel:
            return h
        prev = tel
    return None
