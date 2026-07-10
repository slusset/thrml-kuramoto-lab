"""Map the carrier-fraction Kuramoto experiment onto a clamped Ising PGM.

Correspondence with homeostasis_telos/main.py (the oracle):

  Kuramoto oscillator on a ring        ->  SpinNode on the same ring
  delay-bounded neighborhood (horizon) ->  ferromagnetic edges within +/- horizon
  mutual coherence pull K*sin(psi-th)  ->  coupling weight J on each edge
  carrier (reconstructs the target)    ->  CLAMPED node, pinned to the target spin
  intent diffusion via neighbors       ->  Gibbs relaxation of the free spins
  the target moves (novel perturb.)    ->  flip the clamp values mid-run
  holographic depth h                  ->  fraction of nodes clamped

The target-blind/target-relative distinction carries over: the local
correlation score NEVER references the target, while target overlap measures
alignment with the moved reference. Below an operational, horizon-dependent
pinning threshold the free lattice remains metastably in the old state. Above
it, the clamped nodes can nucleate a reversal within the observation window.
"""

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
from thrml import Block, SpinNode
from thrml.models import IsingEBM, IsingSamplingProgram

from .topology import build_edges


@dataclass(frozen=True)
class RingSpec:
    """Knobs of the experiment; `h` is the fraction of externally pinned nodes."""

    n: int = 120          # oracle N
    horizon: int = 6      # oracle delay-bounded neighborhood (ring/small_world)
    j: float = 0.15       # coupling per edge (Kuramoto K analog; sets metastability depth)
    beta: float = 1.0     # inverse temperature (noise analog, inverted)
    h: float = 0.05       # carrier fraction in [0, 1]
    seed: int = 7         # oracle seed
    topology: str = "ring"   # "ring" | "small_world" | "scale_free"
    p_rewire: float = 0.1    # small_world only
    m: int = 6               # scale_free only: edges per new node
    placement: str = "random"  # "random" | "hub" | "ci" -- where the carriers sit


@dataclass
class ModelBundle:
    """Everything sample_states needs, plus index bookkeeping for scoring."""

    spec: RingSpec
    nodes: list
    ebm: IsingEBM
    program: IsingSamplingProgram
    free_blocks: list          # colored Blocks of free nodes (Gibbs-parallel)
    clamped_blocks: list       # [] when h == 0
    readout_block: Block       # all free nodes in ring order, for observation
    free_idx: np.ndarray       # ring indices of free nodes, ring order
    carrier_idx: np.ndarray    # ring indices of carriers, ring order
    free_edge_pairs: np.ndarray  # (m, 2) positions into free_idx for free-free edges


def greedy_coloring(adj: dict[int, set], subset: list[int]) -> list[list[int]]:
    """Color `subset` so no two neighbors share a color; returns color groups."""
    if not subset:  # h == 1: every node is a carrier, nothing to color
        return []
    color: dict[int, int] = {}
    for i in subset:
        used = {color[j] for j in adj[i] if j in color}
        c = 0
        while c in used:
            c += 1
        color[i] = c
    groups: list[list[int]] = [[] for _ in range(max(color.values()) + 1)]
    for i in subset:
        groups[color[i]].append(i)
    return groups


def build(spec: RingSpec) -> ModelBundle:
    rng = np.random.default_rng(spec.seed)
    nodes = [SpinNode() for _ in range(spec.n)]

    idx_edges = build_edges(spec)
    edges = [(nodes[a], nodes[b]) for a, b in idx_edges]

    # carriers: random placement is the oracle's rule; "hub" places them on
    # the highest-degree nodes (pinning control); "ci" uses collective
    # influence CI_2 (Morone & Makse: degree weighted by degrees at the
    # boundary of the radius-2 ball -- static top-k version)
    n_carriers = int(round(spec.h * spec.n))
    if spec.placement in ("hub", "ci"):
        degree = np.zeros(spec.n, dtype=int)
        adj0: dict[int, set] = {i: set() for i in range(spec.n)}
        for a, b in idx_edges:
            degree[a] += 1
            degree[b] += 1
            adj0[a].add(b)
            adj0[b].add(a)
        if spec.placement == "hub":
            score = degree.astype(float)
        else:
            score = np.zeros(spec.n)
            for i in range(spec.n):
                ball1 = adj0[i]
                ball2 = set().union(*(adj0[j] for j in ball1)) - ball1 - {i}
                score[i] = (degree[i] - 1) * sum(degree[j] - 1 for j in ball2)
        carrier_idx = np.sort(np.argsort(-score)[:n_carriers])
    else:
        carrier_idx = np.sort(rng.permutation(spec.n)[:n_carriers])
    carrier_set = set(carrier_idx.tolist())
    free_idx = np.array([i for i in range(spec.n) if i not in carrier_set])

    adj: dict[int, set] = {i: set() for i in range(spec.n)}
    for a, b in idx_edges:
        adj[a].add(b)
        adj[b].add(a)

    # free nodes, graph-colored into Gibbs-parallel blocks
    groups = greedy_coloring(adj, free_idx.tolist())
    free_blocks = [Block([nodes[i] for i in g]) for g in groups]
    clamped_blocks = [Block([nodes[i] for i in carrier_idx])] if n_carriers else []

    ebm = IsingEBM(
        nodes,
        edges,
        jnp.zeros((spec.n,)),
        jnp.ones((len(edges),)) * spec.j,
        jnp.array(spec.beta),
    )
    program = IsingSamplingProgram(ebm, free_blocks, clamped_blocks)

    # free-free edges, re-indexed into positions within free_idx (for the
    # target-blind homeostatic score)
    pos = {int(i): p for p, i in enumerate(free_idx)}
    free_edge_pairs = np.array(
        [(pos[a], pos[b]) for a, b in idx_edges if a in pos and b in pos],
        dtype=int,
    ).reshape(-1, 2)

    return ModelBundle(
        spec=spec,
        nodes=nodes,
        ebm=ebm,
        program=program,
        free_blocks=free_blocks,
        clamped_blocks=clamped_blocks,
        readout_block=Block([nodes[i] for i in free_idx]),
        free_idx=free_idx,
        carrier_idx=carrier_idx,
        free_edge_pairs=free_edge_pairs,
    )
