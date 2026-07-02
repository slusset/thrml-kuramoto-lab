"""Infiltration contest: captured hubs vs loyal defenders.

Usage:
    uv run python run_contest.py
    uv run python run_contest.py --topology ring --j 0.15
"""

import argparse

from carrier_pgm.infiltration import contest_table
from carrier_pgm.model import RingSpec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--j", type=float, default=0.15)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--topology", choices=["ring", "small_world", "scale_free"],
                   default="scale_free")
    p.add_argument("--m", type=int, default=6)
    p.add_argument("--p-rewire", type=float, default=0.1)
    args = p.parse_args()

    base = RingSpec(
        n=args.n, horizon=args.horizon, j=args.j, beta=args.beta, seed=args.seed,
        topology=args.topology, m=args.m, p_rewire=args.p_rewire,
    )
    print(f"topology: {args.topology} -- attackers take the top-k hubs")
    print("cells: free-lattice fidelity to the TRUE target\n")
    contest_table(base)


if __name__ == "__main__":
    main()
