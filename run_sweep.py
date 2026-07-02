"""CLI mirror of homeostasis_telos/main.py, on the THRML side.

Usage:
    uv run python run_sweep.py                    # oracle-style h table + scan
    uv run python run_sweep.py --beta 1.2 --j 0.4 # explore metastability depth
"""

import argparse

import numpy as np

from carrier_pgm import RingSpec, critical_h, sweep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--j", type=float, default=0.15)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--topology", choices=["ring", "small_world", "scale_free"],
                   default="ring")
    p.add_argument("--p-rewire", type=float, default=0.1, help="small_world only")
    p.add_argument("--m", type=int, default=6, help="scale_free: edges per node")
    p.add_argument("--placement", choices=["random", "hub"], default="random")
    p.add_argument("--scan-max", type=float, default=0.5)
    p.add_argument("--scan-points", type=int, default=26)
    args = p.parse_args()

    base = RingSpec(
        n=args.n, horizon=args.horizon, j=args.j, beta=args.beta, seed=args.seed,
        topology=args.topology, p_rewire=args.p_rewire, m=args.m,
        placement=args.placement,
    )
    print(f"topology: {args.topology}, placement: {args.placement}\n")

    sweep([0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 1.0], base=base)

    print("\nCritical-threshold scan (where does intent survive perturbation?):")
    rows = sweep(
        np.linspace(0, args.scan_max, args.scan_points), base=base, verbose=False
    )
    hc = critical_h(rows)
    if hc is None:
        print("  -> no crossing found in scan range (deepen beta/j or widen range)")
    else:
        print(f"  -> teleological fidelity crosses 0.6 near h = {hc:.3f}")


if __name__ == "__main__":
    main()
