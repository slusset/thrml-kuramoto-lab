"""The adversarial reading of the pinning result: infiltration.

The swarm is locked on the true target. At t=0, attackers CAPTURE the
top-k hubs -- their state is clamped to the anti-target. Defenders hold
their own clamped nodes loyal to the true target. Two boundary
conditions now fight over the same free lattice.

The question the forward experiment implies in reverse: if 2 clamped
hubs are enough to steer the network, are 2 captured hubs enough to
steal it -- and what defense (fraction, placement) restores it?

Scores are over free nodes only, vs the TRUE target. tel -> +1 means the
defense holds; tel -> -1 means the network followed the captured hubs.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram

from .model import RingSpec, greedy_coloring
from .topology import build_edges


def run_infiltration(
    spec: RingSpec,
    n_captured: int,
    defender_h: float = 0.0,
    defender_placement: str = "random",  # "random" | "hub" (next-best hubs)
    n_watch: int = 600,
    steps_per_sample: int = 1,
) -> dict:
    """Returns post-watch scores. Attackers always take the TOP hubs."""
    rng = np.random.default_rng(spec.seed)
    idx_edges = build_edges(spec)

    degree = np.zeros(spec.n, dtype=int)
    adj: dict[int, set] = {i: set() for i in range(spec.n)}
    for a, b in idx_edges:
        degree[a] += 1
        degree[b] += 1
        adj[a].add(b)
        adj[b].add(a)

    by_degree = np.argsort(-degree)
    captured_idx = np.sort(by_degree[:n_captured])
    captured_set = set(captured_idx.tolist())

    n_def = int(round(defender_h * spec.n))
    remaining = [i for i in range(spec.n) if i not in captured_set]
    if defender_placement == "hub":  # next-best hubs after the captured ones
        loyal_idx = np.sort([i for i in by_degree if i not in captured_set][:n_def])
    else:
        loyal_idx = np.sort(rng.permutation(remaining)[:n_def])
    loyal_set = set(loyal_idx.tolist())

    free_idx = np.array(
        [i for i in range(spec.n) if i not in captured_set and i not in loyal_set]
    )

    nodes = [SpinNode() for _ in range(spec.n)]
    edges = [(nodes[a], nodes[b]) for a, b in idx_edges]
    ebm = IsingEBM(
        nodes, edges,
        jnp.zeros((spec.n,)),
        jnp.ones((len(edges),)) * spec.j,
        jnp.array(spec.beta),
    )

    groups = greedy_coloring(adj, free_idx.tolist())
    free_blocks = [Block([nodes[i] for i in g]) for g in groups]
    clamped_blocks, clamp_state = [], []
    if len(loyal_idx):
        clamped_blocks.append(Block([nodes[i] for i in loyal_idx]))
        clamp_state.append(jnp.ones((len(loyal_idx),), dtype=bool))    # true target
    if len(captured_idx):
        clamped_blocks.append(Block([nodes[i] for i in captured_idx]))
        clamp_state.append(jnp.zeros((len(captured_idx),), dtype=bool))  # anti-target

    program = IsingSamplingProgram(ebm, free_blocks, clamped_blocks)

    # t=0: the swarm is locked on the true target; the capture just happened
    state0 = [jnp.ones((len(b.nodes),), dtype=bool) for b in free_blocks]
    readout = Block([nodes[i] for i in free_idx])
    sched = SamplingSchedule(
        n_warmup=0, n_samples=n_watch, steps_per_sample=steps_per_sample
    )
    out = sample_states(
        jax.random.key(spec.seed), program, sched, state0, clamp_state, [readout]
    )

    s = np.asarray(out[0], dtype=int) * 2 - 1
    tel = (s * (+1)).mean(axis=1)  # vs TRUE target
    pos = {int(i): p for p, i in enumerate(free_idx)}
    pairs = np.array(
        [(pos[a], pos[b]) for a, b in idx_edges if a in pos and b in pos], dtype=int
    ).reshape(-1, 2)
    hom = (
        (s[:, pairs[:, 0]] * s[:, pairs[:, 1]]).mean(axis=1)
        if len(pairs) else np.ones(s.shape[0])
    )

    w = min(50, n_watch)
    return {
        "tel": float(tel[-w:].mean()),
        "hom": float(hom[-w:].mean()),
        "tel_series": tel,
        "hom_series": hom,
        "captured_idx": captured_idx,
        "loyal_idx": loyal_idx,
        "captured_degrees": degree[captured_idx].tolist(),
    }


def contest_table(
    base: RingSpec,
    captured_ks=(1, 2, 3, 5, 8),
    defenses=((0.0, "random"), (0.05, "random"), (0.10, "random"), (0.05, "hub")),
    **kw,
) -> None:
    """Rows: number of captured hubs. Columns: defense. Cell: tel vs true target."""
    hdr = "".join(
        f"{'none' if h == 0 else f'{h:.0%} {p}':>14}" for h, p in defenses
    )
    print(f"{'captured':>9}{hdr}")
    print("-" * (9 + 14 * len(defenses)))
    for k in captured_ks:
        cells = []
        for h, p in defenses:
            r = run_infiltration(base, k, defender_h=h, defender_placement=p, **kw)
            t = r["tel"]
            mark = "HELD" if t > 0.6 else "LOST" if t < -0.6 else "torn"
            cells.append(f"{t:+.2f} {mark:<4}".rjust(14))
        print(f"{k:>9}" + "".join(cells))
