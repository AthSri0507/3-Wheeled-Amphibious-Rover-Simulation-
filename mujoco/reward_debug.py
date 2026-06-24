"""
reward_debug.py  --  reward sanity gate before PPO.

Runs the random policy and the simple rule-based controller for >=100 episodes each
(on ONE throwaway env whose per-mode normalizer is warmed up first), and reports mean
episode reward, distance, energy, CoT, and the per-component breakdown. Asserts:
  - rule-based mean reward > random (clear margin)
  - no NaNs / no exploding components
  - per-mode normalization populated for BOTH land and water (components O(1))
The env / normalizer here are DISCARDED; PPO builds fresh envs (no stats persisted).

Usage:  python reward_debug.py --episodes 100
"""
import argparse
import contextlib
import io
import os

import numpy as np

import rover_env
from baseline_rule import RuleController


@contextlib.contextmanager
def _quiet():
    """Silence the RoverController's transition prints during stepping."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield

COMP_KEYS = ["progress_reward", "efficiency_penalty", "stability_penalty",
             "action_penalty", "total_reward", "distance", "energy", "CoT"]


def run(env, policy, n, rule=None):
    recs = []
    for _ in range(n):
        obs, _ = env.reset()
        bad = False
        while True:
            a = env.action_space.sample() if policy == "random" else rule.act(env.terrain_spec.cls)
            obs, r, term, trunc, info = env.step(a)
            if not np.isfinite(r) or not np.all(np.isfinite(obs)):
                bad = True
            if term or trunc:
                break
        c = dict(info["episode_components"]); c["_bad"] = bad
        recs.append(c)
    return recs


def summarize(recs):
    out = {k: float(np.mean([r[k] for r in recs])) for k in COMP_KEYS}
    out["_bad"] = any(r["_bad"] for r in recs)
    return out


def main(episodes):
    env = rover_env.RoverEnv(seed=0)
    rule = RuleController()

    print(f"warming per-mode normalizer (~populating >= {rover_env.WARMUP_N}/mode)...")
    with _quiet():
        run(env, "random", 25)                  # warm up the running stats
    pop = env.norm.populated()
    print(f"  normalizer samples: land={pop[rover_env.LAND]}  water={pop[rover_env.WATER]}")

    print(f"\nevaluating {episodes} episodes each (normalizer warmed)...")
    with _quiet():
        rnd = summarize(run(env, "random", episodes))
        rul = summarize(run(env, "rule", episodes, rule))

    hdr = f"{'metric':18s}{'random':>12s}{'rule':>12s}"
    print("\n" + hdr); print("-" * len(hdr))
    for k in COMP_KEYS:
        print(f"{k:18s}{rnd[k]:12.3f}{rul[k]:12.3f}")

    pop = env.norm.populated()
    both_warm = pop[rover_env.LAND] >= rover_env.WARMUP_N and pop[rover_env.WATER] >= rover_env.WARMUP_N

    print("\n--- GATE ---")
    checks = {
        "no NaNs / exploding": (not rnd["_bad"]) and (not rul["_bad"]),
        "rule total > random total": rul["total_reward"] > rnd["total_reward"],
        "rule distance > random distance": rul["distance"] > rnd["distance"],
        "per-mode normalizer warmed (land & water)": both_warm,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    passed = all(checks.values())
    print(f"\nGATE {'PASSED' if passed else 'FAILED'}")
    return passed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    a = ap.parse_args()
    ok = main(a.episodes)
    raise SystemExit(0 if ok else 1)
