"""
eval_nav.py  --  navigation task evaluation (reach a laterally-offset goal).

Compares: straight baselines (hand-tuned, rule -- can't steer, should ~fail),
a hand-coded pivot-and-go proportional-nav controller (the competent reference),
and the trained PPO nav policy. Reports reach rate, mean final goal distance,
mean steps, and CoT.

Usage:  python eval_nav.py --episodes 40
"""
import argparse
import contextlib
import io
import os

import numpy as np
from stable_baselines3 import PPO

import rover_sim; rover_sim.QUIET = True
import rover_env
from baseline_rule import RuleController

HERE = os.path.dirname(os.path.abspath(__file__))


def prop_nav(obs, env, K=3.0):
    g = env._goal_relative()                      # [forward_comp, lateral_comp]
    steer = float(np.clip(-g[1] * K, -1, 1))      # turn toward goal
    drive = float(np.clip(g[0], 0.0, 1.0))        # drive only when facing it (pivot otherwise)
    return np.array([drive, steer, 0.0], np.float32)


def run(env, action_fn, n):
    reach, gdist, steps, cot = [], [], [], []
    for _ in range(n):
        obs, _ = env.reset()
        while True:
            obs, r, term, trunc, info = env.step(action_fn(obs, env))
            if term or trunc:
                break
        nv, c = info["nav"], info["episode_components"]
        reach.append(nv["reached"]); gdist.append(nv["goal_dist"]); steps.append(nv["steps"])
        cot.append(c["CoT"])
    return dict(reach=float(np.mean(reach)), gdist=float(np.mean(gdist)),
                steps=float(np.mean(steps)), cot=float(np.mean(cot)))


def main(episodes, model_path):
    env = rover_env.RoverEnv(seed=29, task="nav")
    ppo = PPO.load(model_path, device="cpu")
    rule = RuleController()
    controllers = {
        "hand-tuned (straight)": lambda o, e: np.array([1.0, 0.0, 0.0], np.float32),
        "rule (straight)":       lambda o, e: rule.act(e.terrain_spec.cls),
        "prop-nav (hand-coded)": prop_nav,
        "ppo (learned)":         lambda o, e: ppo.predict(o, deterministic=True)[0],
    }
    print(f"navigation eval ({episodes} episodes each)\n")
    hdr = f"{'controller':24s}{'reach%':>8s}{'final_dist':>11s}{'steps':>8s}{'CoT':>8s}"
    print(hdr); print("-" * len(hdr))
    for name, fn in controllers.items():
        with contextlib.redirect_stdout(io.StringIO()):
            r = run(env, fn, episodes)
        print(f"{name:24s}{r['reach']*100:7.0f}%{r['gdist']:11.2f}{r['steps']:8.0f}{r['cot']:8.2f}")
    env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--model", default=os.path.join(HERE, "models", "ppo_nav.zip"))
    a = ap.parse_args()
    main(a.episodes, a.model)
