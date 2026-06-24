"""
compare_controllers.py  --  final controller comparison on the FROZEN amphibious benchmark.

Task columns (held-out eval seeds, completion = primary): completion %, mean min-distance,
stuck-fraction, max roll/pitch, water-traversal. The deployable supervised controller is the
headline; rule and oracle are baselines/ceiling. (PPO/SAC omitted: the early-exit triggered --
the deployable rule already matches the oracle, so there is no headroom for RL.)
"""
import os
from collections import Counter

import numpy as np

import amphib
import controllers_amphibious as C

EVAL_SEEDS = [200 + i for i in range(10)]
HERE = os.path.dirname(os.path.abspath(__file__))


def run(make, seeds):
    rs = []
    for s in seeds:
        w = amphib.AmphibWorld(seed=s, current=float((s % 5) / 10.0))
        rs.append(C.run_episode(w, make(w.dt)))
    return rs


def main():
    makers = {
        "nonadaptive":         lambda dt: C.make_controller("nonadaptive", dt=dt),
        "classifier_fsm_rule": lambda dt: C.make_controller("classifier_fsm_rule", dt=dt),
        "oracle_upperbound":   lambda dt: C.make_controller("oracle_upperbound", dt=dt),
    }
    sup = os.path.join(HERE, "models", "ctrl_supervised.joblib")
    if os.path.exists(sup):
        import joblib
        m = joblib.load(sup)["model"]
        makers["supervised (DEPLOYED)"] = lambda dt, m=m: C.make_supervised_controller(m, dt=dt)

    order = ["nonadaptive", "classifier_fsm_rule", "supervised (DEPLOYED)", "oracle_upperbound"]
    print(f"FROZEN amphibious benchmark -- {len(EVAL_SEEDS)} held-out seeds\n")
    print(f"{'controller':24s}{'completion':>11s}{'min_dist':>9s}{'stuck%':>8s}{'maxroll':>9s}  water-traversal")
    print("-" * 78)
    for name in order:
        if name not in makers:
            continue
        rs = run(makers[name], EVAL_SEEDS)
        comp = 100 * np.mean([r["reached"] for r in rs])
        md = np.mean([r["min_dist"] for r in rs]); sf = 100 * np.mean([r["stuck_frac"] for r in rs])
        mr = np.mean([max(r["max_roll"], r["max_pitch"]) for r in rs])
        wt = 100 * np.mean([r["crossed_water"] for r in rs])
        print(f"{name:24s}{comp:9.0f}%{md:9.2f}{sf:7.0f}%{mr:9.2f}  crossed {wt:.0f}%")
    print("\n(completion is the primary metric; supervised = the deployed learned controller.)")


if __name__ == "__main__":
    main()
