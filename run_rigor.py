"""Rigor checks before trusting any threshold number.

  seeds   does h_c wobble across carrier placements / RNG streams?
  equil   is the threshold stable when Gibbs steps double? (mixing check)
  size    does the transition SHARPEN as N grows? (real transition vs
          sampler artifact -- finite-size scaling, coarse version)

Usage:
    uv run python run_rigor.py seeds
    uv run python run_rigor.py equil
    uv run python run_rigor.py size
"""

import argparse
import dataclasses

from carrier_pgm import RingSpec, run_experiment

FAST = dict(n_lock=150, n_watch=300)  # rigor runs favor breadth over depth
BASE = RingSpec()  # ring, beta=1.0, j=0.15


def cell(spec, **kw):
    _, tel, _ = run_experiment(spec, **{**FAST, **kw}).summary()
    return tel


def check_seeds() -> None:
    seeds = [1, 2, 3, 4, 7]
    hs = [0.01, 0.02, 0.03, 0.05]
    print("ring: post-flip teleological fidelity  (rows=seed, cols=h)")
    print(f"{'seed':>5}" + "".join(f"{h:>8.2f}" for h in hs))
    for s in seeds:
        row = [cell(dataclasses.replace(BASE, seed=s, h=h)) for h in hs]
        print(f"{s:>5}" + "".join(f"{t:>8.2f}" for t in row))
    print("\nread: the h where each row turns positive is that seed's h_c.")


def check_equil() -> None:
    hs = [0.01, 0.02, 0.03]
    print("ring, seed 7: fidelity at 1x vs 2x vs 4x Gibbs steps per sample")
    print(f"{'h':>6}{'1x':>8}{'2x':>8}{'4x':>8}")
    for h in hs:
        spec = dataclasses.replace(BASE, h=h)
        row = [cell(spec, steps_per_sample=k) for k in (1, 2, 4)]
        print(f"{h:>6.2f}" + "".join(f"{t:>8.2f}" for t in row))
    print("\nread: columns should agree; drift with more steps = not mixed.")


def check_size() -> None:
    ns = [60, 120, 240]
    ks = [0, 1, 2, 3, 4, 6]  # carriers by COUNT so placement density is h=k/N
    print("ring: fidelity vs carrier count k, per system size N")
    print(f"{'N':>5}" + "".join(f"{f'k={k}':>8}" for k in ks))
    for n in ns:
        row = [cell(dataclasses.replace(BASE, n=n, h=k / n)) for k in ks]
        print(f"{n:>5}" + "".join(f"{t:>8.2f}" for t in row))
    print("\nread: in h=k/N units a sharpening transition means the jump")
    print("happens at a k that grows slower than N (h_c shrinking) with a")
    print("steeper rise. A flat/blurry jump at all N = sampler artifact.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("check", choices=["seeds", "equil", "size"])
    args = p.parse_args()
    {"seeds": check_seeds, "equil": check_equil, "size": check_size}[args.check]()
