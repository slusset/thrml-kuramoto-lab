"""Master curve: kinetic threshold h_c as a function of beta*J.

Since beta multiplies the whole energy in THRML's IsingEBM, the Gibbs
dynamics depend only on the product beta*J -- the data collapse the
statistical-mechanics reading predicts is exact by construction. What's
left to measure is the law itself: how the threshold rises as the
metastable trap (beta*J) deepens, at a FIXED watch window (the threshold
is kinetic; quote it with its horizon).

h_c here = smallest h on the grid where median post-flip fidelity
exceeds 0.6 x median coherence -- i.e. the lattice points at the target
as strongly as it points anywhere. (An absolute cut conflates "flipped"
with "ordered": shallow traps cap fidelity at their own weak coherence.)
If coherence itself is < 0.3 the lattice is disordered -- no trap, no
threshold, h_c recorded as 0.

Usage:
    uv run python run_master.py --bj 0.15              # one point
    uv run python run_master.py --bj 0.10 0.15 0.20    # several
    uv run python run_master.py --plot                 # table + PNG from CSV
"""

import argparse
import csv
import dataclasses
import pathlib

from carrier_pgm import RingSpec, run_experiment

CSV = pathlib.Path(__file__).parent / "data" / "master_curve.csv"
SEEDS = (1, 3, 7)
H_GRID = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
N_WATCH = 300  # the kinetic horizon; part of the definition of h_c


def median(xs):
    s = sorted(xs)
    return s[len(s) // 2]


def find_hc(bj: float, topology: str = "ring", placement: str = "random"):
    base = RingSpec(j=bj, beta=1.0, topology=topology, placement=placement)
    for h in H_GRID:
        res = [
            run_experiment(
                dataclasses.replace(base, h=h, seed=s), n_lock=100, n_watch=N_WATCH
            ).summary()
            for s in SEEDS
        ]
        hom = median([r[0] for r in res])
        tel = median([r[1] for r in res])
        print(f"  bj={bj:.3f} h={h:.2f} hom={hom:+.2f} tel={tel:+.2f}")
        if hom < 0.3:
            return 0.0  # disordered: no trap to escape
        if tel > 0.6 * hom:
            return h
    return None  # frozen beyond grid


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bj", type=float, nargs="*", default=[])
    p.add_argument("--topology", default="ring")
    p.add_argument("--placement", default="random")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    CSV.parent.mkdir(exist_ok=True)
    if args.bj:
        new = CSV.exists()
        with open(CSV, "a", newline="") as f:
            w = csv.writer(f)
            if not new:
                w.writerow(["bj", "topology", "placement", "n_watch", "h_c"])
            for bj in args.bj:
                hc = find_hc(bj, args.topology, args.placement)
                w.writerow([bj, args.topology, args.placement, N_WATCH,
                            "" if hc is None else hc])
                print(f"bj={bj:.3f}  ->  h_c = {'>%.2f (frozen)' % H_GRID[-1] if hc is None else f'{hc:.2f}'}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = list(csv.DictReader(open(CSV)))
        pts = sorted(
            (float(r["bj"]), float(r["h_c"]))
            for r in rows if r["h_c"] != "" and r["topology"] == "ring"
        )
        frozen = sorted(float(r["bj"]) for r in rows if r["h_c"] == "")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if pts:
            ax.plot([b for b, _ in pts], [h for _, h in pts], "o-", c="tab:red")
        for b in frozen:
            ax.axvline(b, ls=":", c="gray")
            ax.text(b, ax.get_ylim()[1] * 0.9, " frozen", rotation=90, fontsize=8)
        ax.set(xlabel="β·J (trap depth)",
               ylabel=f"kinetic h_c (median of seeds, {N_WATCH}-sweep horizon)",
               title="Master curve: carriers required vs trap depth (ring)")
        out = CSV.parent / "master_curve.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"table:\n{open(CSV).read()}\nplot -> {out}")


if __name__ == "__main__":
    main()
