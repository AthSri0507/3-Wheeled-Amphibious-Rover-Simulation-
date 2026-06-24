"""
eval_policy.py  --  per-terrain evaluation of three controllers.

Compares, per terrain (hard / sand / gravel / water):
  1. hand-tuned   : constant full-forward, terrain-AGNOSTIC  (action [1,0,0])
  2. rule-based   : terrain-aware throttle (baseline_rule)   -> isolates "terrain awareness helps"
  3. ppo          : the trained policy                       -> isolates "RL helps"

Reports mobility (distance) and **Cost of Transport CoT = energy/(mass·distance)** — the
publishable metric — plus mean episode reward. Physical metrics are normalization-free.

Usage:  python eval_policy.py --episodes 20
"""
import argparse
import contextlib
import io
import os

import numpy as np
from stable_baselines3 import PPO

import rover_env
import terrain as terr
from baseline_rule import RuleController

HERE = os.path.dirname(os.path.abspath(__file__))


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


METRICS = ["distance", "CoT", "total_reward", "energy",
           "progress_reward", "efficiency_penalty", "stability_penalty", "action_penalty"]


def run(env, action_fn, n):
    acc = {k: [] for k in METRICS}
    act = {"drive": [], "absteer": [], "absarm": []}
    for _ in range(n):
        obs, _ = env.reset()
        ea = []
        while True:
            a = np.clip(np.asarray(action_fn(obs, env), np.float32), -1, 1)  # as applied
            ea.append(a)
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
        ea = np.array(ea)
        act["drive"].append(ea[:, 0].mean())
        act["absteer"].append(np.abs(ea[:, 1]).mean())
        act["absarm"].append(np.abs(ea[:, 2]).mean())
        c = info["episode_components"]
        for k in METRICS:
            acc[k].append(c[k])
    out = {k: float(np.mean(acc[k])) for k in METRICS}
    out.update({k: float(np.mean(v)) for k, v in act.items()})
    return out


def main(episodes, model_path):
    import rover_sim
    rover_sim.QUIET = True
    env = rover_env.RoverEnv(seed=7)
    ppo = PPO.load(model_path, device="cpu")
    rule = RuleController()

    controllers = {
        "hand-tuned": lambda obs, e: np.array([1.0, 0.0, 0.0], np.float32),
        "rule-based": lambda obs, e: rule.act(e.terrain_spec.cls),
        "ppo":        lambda obs, e: ppo.predict(obs, deterministic=True)[0],
        # Exp A: PPO drive only, steer/arm forced to 0 (tests whether saturation is the cause)
        "ppo_nosa":   lambda obs, e: np.array([ppo.predict(obs, deterministic=True)[0][0], 0.0, 0.0], np.float32),
        # Exp B.1: PPO drive+steer kept, arm_trim forced to 0 (isolates the residual water arm saturation)
        "ppo_noarm":  lambda obs, e: np.array([*ppo.predict(obs, deterministic=True)[0][:2], 0.0], np.float32),
    }

    print(f"per-terrain eval ({episodes} episodes each)\n")
    allrows = {}
    for cls in terr.CLASSES:
        env.set_allowed_classes([cls])
        with _quiet():
            allrows[cls] = {name: run(env, fn, episodes) for name, fn in controllers.items()}

    # ---- table 1: mobility / efficiency ----
    hdr = f"{'terrain':8s}{'controller':12s}{'distance':>10s}{'CoT':>10s}{'reward':>10s}"
    print(hdr); print("-" * len(hdr))
    for cls in terr.CLASSES:
        for name, r in allrows[cls].items():
            print(f"{cls:8s}{name:12s}{r['distance']:10.3f}{r['CoT']:10.2f}{r['total_reward']:10.1f}")
        best = min(allrows[cls].items(), key=lambda kv: kv[1]['CoT'])[0]
        print(f"   -> lowest CoT: {best}")

    # ---- table 2: reward-component breakdown (PPO vs rule) ----
    print("\nreward-component breakdown (episode sums):")
    h2 = f"{'terrain':8s}{'controller':12s}{'progress':>10s}{'effic.':>10s}{'stabil.':>10s}{'action':>10s}{'energy':>9s}"
    print(h2); print("-" * len(h2))
    for cls in terr.CLASSES:
        for name in ("rule-based", "ppo"):
            r = allrows[cls][name]
            print(f"{cls:8s}{name:12s}{r['progress_reward']:10.1f}{r['efficiency_penalty']:10.1f}"
                  f"{r['stability_penalty']:10.1f}{r['action_penalty']:10.1f}{r['energy']:9.2f}")

    # ---- table 3: action diagnostics (is PPO oscillating steer/arm?) ----
    print("\naction diagnostics (mean drive, mean |steer|, mean |arm_trim|):")
    h3 = f"{'terrain':8s}{'controller':12s}{'drive':>10s}{'|steer|':>10s}{'|arm|':>10s}"
    print(h3); print("-" * len(h3))
    for cls in terr.CLASSES:
        for name in ("rule-based", "ppo"):
            r = allrows[cls][name]
            print(f"{cls:8s}{name:12s}{r['drive']:10.3f}{r['absteer']:10.3f}{r['absarm']:10.3f}")
    env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--model", default=os.path.join(HERE, "models", "ppo_rover.zip"))
    a = ap.parse_args()
    main(a.episodes, a.model)
