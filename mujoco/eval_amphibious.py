"""
eval_amphibious.py  --  statistical held-out comparison: supervised vs oracle_upperbound.

Uses a FRESH seed block (300+) that was NOT used for behaviour-cloning training (0-19) nor
for any prior eval/gate (200-209). Reports, with 95% CIs:
  - completion rate (primary)
  - shoreline-transition success (crossed the shoreline AND reached)
  - current rejection (completion on the HIGH-current subset)
  - stability (max roll/pitch, flip rate)

Usage:  python eval_amphibious.py --n 16
"""
import argparse
import math
import os

import numpy as np

import amphib
import controllers_amphibious as C

FRESH_BASE = 300        # held-out, never used for training or prior eval
HERE = os.path.dirname(os.path.abspath(__file__))


def ci95(xs):
    xs = np.asarray(xs, float)
    if len(xs) < 2:
        return float(xs.mean()), 0.0
    return float(xs.mean()), 1.96 * xs.std(ddof=1) / math.sqrt(len(xs))


def controllers(dt):
    ctrls = {"oracle_upperbound": C.make_controller("oracle_upperbound", dt=dt)}
    sup = os.path.join(HERE, "models", "ctrl_supervised.joblib")
    if os.path.exists(sup):
        import joblib
        ctrls["supervised_xgboost"] = C.make_supervised_controller(joblib.load(sup)["model"], dt=dt)
    ctrls["classifier_fsm_rule"] = C.make_controller("classifier_fsm_rule", dt=dt)
    return ctrls


def main(n):
    seeds = [FRESH_BASE + i for i in range(n)]
    currents = [float((s % 5) / 10.0) for s in seeds]   # 0.0..0.4
    hi = [i for i, c in enumerate(currents) if c >= 0.3]  # high-current subset
    print(f"HELD-OUT comparison (FRESH seeds {seeds[0]}-{seeds[-1]}, never trained/evaluated on)")
    print(f"  currents per seed: {currents}\n")
    print(f"{'controller':22s}{'completion':>14s}{'shoreline':>12s}{'current-rej':>13s}{'maxroll':>9s}{'flip%':>7s}")
    print("-" * 77)
    for name in ["oracle_upperbound", "supervised_xgboost", "classifier_fsm_rule"]:
        comp, shore, roll, flip = [], [], [], []
        for s, cur in zip(seeds, currents):
            w = amphib.AmphibWorld(seed=s, current=cur)
            ctrls = controllers(w.dt)
            if name not in ctrls:
                continue
            r = C.run_episode(w, ctrls[name])
            comp.append(float(r["reached"]))
            shore.append(float(r["reached"] and r["crossed_water"]))
            roll.append(max(r["max_roll"], r["max_pitch"])); flip.append(float(r["flipped"]))
        cm, cc = ci95(comp); sm, sc = ci95(shore)
        crm, crc = ci95([comp[i] for i in hi])
        rm, _ = ci95(roll); fm = 100 * np.mean(flip)
        print(f"{name:22s}{cm*100:7.0f}+-{cc*100:3.0f}%{sm*100:8.0f}+-{sc*100:2.0f}%"
              f"{crm*100:8.0f}+-{crc*100:2.0f}%{rm:9.2f}{fm:6.0f}%")
    print("\ncompletion = primary; shoreline = crossed+reached; current-rej = completion on high-current seeds.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args()
    main(a.n)
