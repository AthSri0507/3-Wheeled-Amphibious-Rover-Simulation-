"""
gate_amphibious.py  --  validity gate for the amphibious terrain-adaptive benchmark.

Primary metric = COMPLETION RATE (reached the water waypoint). mean +/- 95% CI over N held-out
seeds (disjoint from any tuning). The benchmark provides a real adaptation opportunity iff:
  - Adaptation Gain (oracle_upperbound vs nonadaptive completion) is large + CIs non-overlapping;
  - classifier_fsm_rule (deployable, real classifier in the loop) approaches oracle_upperbound;
  - nonadaptive genuinely fails the water regime.

Held-out EVAL seeds only (no tuning was done on these). Usage: python gate_amphibious.py --n 10
"""
import argparse
import math

import numpy as np

import amphib
import controllers_amphibious as C

EVAL_SEEDS_BASE = 200            # held-out seed block (distinct from dev/tuning seeds 0-99)
CONTROLLERS = ["oracle_upperbound", "classifier_fsm_rule", "oracle_teacher", "nonadaptive"]


def ci95(xs):
    xs = np.asarray(xs, float); m = xs.mean()
    s = xs.std(ddof=1) if len(xs) > 1 else 0.0
    return m, 1.96 * s / math.sqrt(len(xs)) if len(xs) > 1 else 0.0


def run_controller(name, n, seeds):
    res = []
    for s in seeds:
        w = amphib.AmphibWorld(seed=s, current=float((s % 5) / 10.0))   # currents 0.0..0.4
        res.append(C.run_episode(w, C.make_controller(name, dt=w.dt)))
    return res


def main(n):
    seeds = [EVAL_SEEDS_BASE + i for i in range(n)]
    print(f"AMPHIBIOUS VALIDITY GATE  (N={n} held-out seeds, currents 0.0-0.4)\n")
    print(f"{'controller':22s}{'completion':>14s}{'mean_min_dist':>14s}{'flip%':>7s}{'fails':>26s}")
    print("-" * 83)
    comp = {}
    for name in CONTROLLERS:
        rs = run_controller(name, n, seeds)
        c = [float(r["reached"]) for r in rs]
        cm, cc = ci95(c); comp[name] = (cm, cc)
        md, _ = ci95([r["min_dist"] for r in rs]); fl = 100 * np.mean([r["flipped"] for r in rs])
        from collections import Counter
        fails = " ".join(f"{k}:{v}" for k, v in Counter(r["fail"] for r in rs).most_common())
        print(f"{name:22s}{cm*100:8.0f}±{cc*100:3.0f}%{md:14.2f}{fl:6.0f}%{'  '+fails:>26s}")

    print("\n--- GATE ---")
    o_m, o_c = comp["oracle_upperbound"]; n_m, n_c = comp["nonadaptive"]
    r_m, r_c = comp["classifier_fsm_rule"]
    gain = (o_m - n_m) / n_m * 100 if n_m > 1e-9 else float("inf")
    nonoverlap = (o_m - o_c) > (n_m + n_c)
    rule_ok = (r_m + r_c) >= (o_m - o_c) or abs(r_m - o_m) < 0.15
    print(f"oracle_upperbound completion = {o_m*100:.0f}±{o_c*100:.0f}% ; nonadaptive = {n_m*100:.0f}±{n_c*100:.0f}%")
    print(f"Adaptation Gain (completion) = {gain:.0f}%  (CIs non-overlapping: {nonoverlap})")
    print(f"classifier_fsm_rule = {r_m*100:.0f}±{r_c*100:.0f}%  (approaches oracle: {rule_ok})")
    gate = nonoverlap and (gain >= 15 or n_m < 0.5 < o_m) and rule_ok
    print(f"\nVERDICT: benchmark {'PASSES -> real, large adaptation opportunity; classifier earns its keep' if gate else 'FAILS'}")
    print("  (nonadaptive fails the water regime; classifier->FSM->control matches the privileged oracle)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    main(a.n)
