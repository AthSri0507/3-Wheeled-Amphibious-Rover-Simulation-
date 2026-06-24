"""
eval_transitions.py  --  the headline evaluation: terrain crossings, not pure terrain.

Runs the four transition scenarios (hard->sand, sand->gravel, shoreline->water,
water->shoreline) for the three controllers (hand-tuned, rule-based, PPO) and reports
completion rate, net forward distance, CoT, time-stuck fraction, and reward — where
constant-forward baselines should struggle and terrain-adaptive RL can earn its keep.

Usage:  python eval_transitions.py --episodes 20
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
SCENARIOS = ["hard_sand", "sand_gravel", "shoreline_water", "water_shoreline"]


def run(env, action_fn, scenario, n):
    comp, cot, rew, stuck, netf = [], [], [], [], []
    for _ in range(n):
        obs, _ = env.reset(options={"scenario": scenario})
        while True:
            obs, r, term, trunc, info = env.step(action_fn(obs, env))
            if term or trunc:
                break
        s, c = info["scenario"], info["episode_components"]
        comp.append(s["completed"]); stuck.append(s["stuck_frac"]); netf.append(s["net_fwd"])
        cot.append(c["CoT"]); rew.append(c["total_reward"])
    return dict(completion=float(np.mean(comp)), CoT=float(np.mean(cot)),
                reward=float(np.mean(rew)), stuck=float(np.mean(stuck)),
                net_fwd=float(np.mean(netf)))


def main(episodes, model_path):
    env = rover_env.RoverEnv(seed=23)
    ppo = PPO.load(model_path, device="cpu")
    rule = RuleController()
    controllers = {
        "hand-tuned": lambda o, e: np.array([1.0, 0.0, 0.0], np.float32),
        "rule-based": lambda o, e: rule.act(e.terrain_spec.cls),
        "ppo":        lambda o, e: ppo.predict(o, deterministic=True)[0],
    }

    print(f"transition evaluation ({episodes} episodes each)\n")
    hdr = (f"{'scenario':16s}{'controller':12s}{'complete':>9s}{'net_fwd':>9s}"
           f"{'CoT':>8s}{'stuck':>7s}{'reward':>9s}")
    print(hdr); print("-" * len(hdr))
    for scn in SCENARIOS:
        rows = {}
        for name, fn in controllers.items():
            with contextlib.redirect_stdout(io.StringIO()):
                rows[name] = run(env, fn, scn, episodes)
        for name, r in rows.items():
            print(f"{scn:16s}{name:12s}{r['completion']*100:7.0f}% {r['net_fwd']:9.3f}"
                  f"{r['CoT']:8.2f}{r['stuck']*100:6.0f}%{r['reward']:9.0f}")
        print()
    env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--model", default=os.path.join(HERE, "models", "ppo_rover.zip"))
    a = ap.parse_args()
    main(a.episodes, a.model)
