"""Connectivity graphs for the substrate-invariance experiment.

The claim under test: the *constant* h_c moves between substrates, but the
*shape* (sharp threshold, sparse sufficiency) tracks graph structure the
same way in both. So the topology is the independent variable:

  ring        the oracle's graph -- neighbors within +/- horizon
  small_world ring with each edge rewired with prob p (Watts-Strogatz-ish);
              shortcuts should drop h_c -- percolation loves shortcuts
  scale_free  preferential attachment (Barabasi-Albert); hubs should drop
              h_c further, and make carrier PLACEMENT matter (clamp a hub
              vs clamp a leaf -- pinning-control territory)

All builders return a list of unordered index pairs, each pair once, no
self-loops, no duplicates -- what model.build() expects.
"""

import numpy as np


def ring(n: int, horizon: int = 6, seed: int | None = None) -> list[tuple[int, int]]:
    """The oracle's graph: each node coupled to neighbors within +/- horizon."""
    return [(i, (i + d) % n) for i in range(n) for d in range(1, horizon + 1)]


def small_world(
    n: int, horizon: int = 6, p_rewire: float = 0.1, seed: int = 7
) -> list[tuple[int, int]]:
    """Watts-Strogatz-style: the ring, with each edge rewired with prob p.

    Keeps the edge count identical to ring(n, horizon), so any threshold
    shift is attributable to shortcuts, not coupling budget.
    """
    rng = np.random.default_rng(seed)
    edges = set(frozenset(e) for e in ring(n, horizon))
    out: set[frozenset] = set()
    for e in edges:
        a, b = tuple(e)
        if rng.random() < p_rewire:
            for _ in range(100):  # find a fresh endpoint
                c = int(rng.integers(n))
                new = frozenset((a, c))
                if c != a and new not in out and new not in edges:
                    out.add(new)
                    break
            else:
                out.add(e)  # dense corner case: keep the original
        else:
            out.add(e)
    return [tuple(sorted(e)) for e in sorted(out, key=lambda s: sorted(s))]


def scale_free(n: int, m: int = 6, seed: int = 7) -> list[tuple[int, int]]:
    """Barabasi-Albert preferential attachment: each new node brings m edges.

    m=6 roughly matches the ring's mean degree (2*horizon = 12) so the
    coupling budget stays comparable. Expect hubs; expect h_c to depend on
    WHERE the carriers sit, not just how many there are.
    """
    rng = np.random.default_rng(seed)
    edges: set[frozenset] = set()
    targets = list(range(m))  # seed clique-ish core
    repeated: list[int] = list(range(m))
    for v in range(m, n):
        chosen: set[int] = set()
        while len(chosen) < m:
            pick = int(repeated[rng.integers(len(repeated))]) if repeated else int(
                rng.integers(v)
            )
            if pick != v:
                chosen.add(pick)
        for t in chosen:
            edges.add(frozenset((v, t)))
            repeated.extend([v, t])  # degree-proportional urn
    return [tuple(sorted(e)) for e in sorted(edges, key=lambda s: sorted(s))]


BUILDERS = {
    "ring": lambda spec: ring(spec.n, spec.horizon),
    "small_world": lambda spec: small_world(
        spec.n, spec.horizon, spec.p_rewire, spec.seed
    ),
    "scale_free": lambda spec: scale_free(spec.n, spec.m, spec.seed),
}


def build_edges(spec) -> list[tuple[int, int]]:
    try:
        return BUILDERS[spec.topology](spec)
    except KeyError:
        raise ValueError(
            f"unknown topology {spec.topology!r}; choose from {sorted(BUILDERS)}"
        ) from None
