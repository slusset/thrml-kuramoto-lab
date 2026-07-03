"""The substrate-invariance test: do thresholds TRACK across substrates?

For each (topology, placement) condition, the SAME graph instance and the
SAME carrier sets (matched RNG) are run through two entirely different
physics:

  Kuramoto  the oracle's phase-oscillator dynamics
            (homeostasis_telos/substrate_invariance.py, verified against
            main.py on the ring)
  Gibbs     block-Gibbs relaxation of the clamped Ising EBM (this repo)

If h_c moves in the same direction, with the same ordering of conditions,
under both dynamics, the topology -- not the substrate -- owns the
threshold. That is the substrate-invariance claim, tested.

Thresholds are kinetic (fixed horizons: oracle 700 post-perturbation
steps, PGM 300 sweeps) and single-instance (seed 7) by design: matched
instances, not ensemble averages. Criteria: oracle tel > 0.6 (its native
rule); PGM tel > 0.6 x hom (relative, see run_master.py).

Usage:
    uv run python run_substrate.py --conditions ring:random scale_free:hub
    uv run python run_substrate.py            # all four, prints the table
"""

import argparse
import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "homeostasis_telos"))

from carrier_pgm import RingSpec, run_experiment
from carrier_pgm.topology import build_edges
from substrate_invariance import summarize_graph  # noqa: E402  (oracle repo)

N = 120
SEED = 7
H_GRID = [0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24, 0.30]
CONDITIONS = [
    ("ring", "random"),
    ("small_world", "random"),
    ("scale_free", "random"),
    ("scale_free", "hub"),
]


def spec_for(topology, placement, h=0.0):
    return RingSpec(n=N, topology=topology, placement=placement, h=h, seed=SEED)


def hc_kuramoto(topology, placement):
    edges = build_edges(spec_for(topology, placement))
    for h in H_GRID:
        _, tel = summarize_graph(h, N, edges, placement=placement, seed=SEED)
        if tel > 0.6:
            return h
    return None


def hc_gibbs(topology, placement):
    for h in H_GRID:
        hom, tel, _ = run_experiment(
            spec_for(topology, placement, h), n_lock=100, n_watch=300
        ).summary()
        if hom < 0.3:
            return 0.0
        if tel > 0.6 * hom:
            return h
    return None


def fmt(hc):
    return f">{H_GRID[-1]:.2f}" if hc is None else f"{hc:.2f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--conditions", nargs="*", default=None,
                   help="topology:placement pairs; default all four")
    args = p.parse_args()
    conds = (
        [tuple(c.split(":")) for c in args.conditions]
        if args.conditions else CONDITIONS
    )

    print(f"{'condition':<24}{'h_c Kuramoto':>14}{'h_c Gibbs':>12}")
    print("-" * 50)
    for topo, place in conds:
        hk = hc_kuramoto(topo, place)
        hg = hc_gibbs(topo, place)
        print(f"{topo + ' / ' + place:<24}{fmt(hk):>14}{fmt(hg):>12}")
    print("\nread: same ordering across rows in both columns = the topology")
    print("owns the threshold; the substrate only sets the scale.")


if __name__ == "__main__":
    main()
