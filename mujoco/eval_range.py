"""
eval_range.py  --  max-range locomotion-adaptation benchmark.

Drives a mixed packed/loose course (loose hidden under the "sand" label) under an energy
budget. Compares fixed-throttle controllers (the fair non-adaptive bound), an ORACLE that
adapts using the TRUE difficulty (the achievable ceiling), and the trained PPO. Reports
reach%, forward distance, energy, CoT, mean slip -- IN-DISTRIBUTION and OUT-OF-DISTRIBUTION
(loose harder than ever trained on) to verify learned adaptation vs memorization.

Usage:  python eval_range.py --episodes 30
"""
import argparse
import contextlib
import io
import os

import numpy as np

import rover_sim; rover_sim.QUIET = True
import rover_env

HERE = os.path.dirname(os.path.abspath(__file__))


def fixed(t):
    return lambda o, e: np.array([t, 0.0, 0.0], np.float32)


def oracle(o, e):
    # cheats: reads the true zone difficulty -> the calibrated optimum (packed 1.0 / loose 0.6)
    diff = e._range_diff_at(e._range_netfwd())
    return np.array([0.6 if diff > 0.5 else 1.0, 0.0, 0.0], np.float32)


def run(env, fn, n, ood):
    reach, fwd, energy, cot, slip = [], [], [], [], []
    for _ in range(n):
        obs, _ = env.reset(options={"ood": ood})
        sl = []
        while True:
            obs, r, term, trunc, info = env.step(fn(obs, env))
            sl.append(abs(float(obs[13])))     # slip channel (acc3,gyr3,rp2,vxy2,wheels3,slip@13)
            if term or trunc:
                break
        rg, c = info["range"], info["episode_components"]
        reach.append(rg["reached"]); fwd.append(rg["net_fwd"]); energy.append(rg["energy"])
        cot.append(c["CoT"]); slip.append(np.mean(sl))
    return dict(reach=float(np.mean(reach)), fwd=float(np.mean(fwd)), energy=float(np.mean(energy)),
               cot=float(np.mean(cot)), slip=float(np.mean(slip)))


def main(episodes, model_path, budget=None, steps=None):
    if budget is not None:
        rover_env.RANGE_BUDGET = budget
    if steps is not None:
        rover_env.RANGE_STEPS = steps
    print(f"[budget={rover_env.RANGE_BUDGET}J steps={rover_env.RANGE_STEPS} goal={rover_env.RANGE_GOAL}m]")
    env = rover_env.RoverEnv(seed=11, task="range")
    controllers = {
        "fixed 1.0 (full)":      fixed(1.0),
        "fixed 0.8":             fixed(0.8),
        "fixed 0.6 (fair rule)": fixed(0.6),
        "oracle adaptive":       oracle,
    }
    if model_path and os.path.exists(model_path):
        is_sac = "sac" in os.path.basename(model_path).lower()
        if is_sac:
            from stable_baselines3 import SAC
            net = SAC.load(model_path, device="cpu"); label = "sac (learned)"
        else:
            from stable_baselines3 import PPO
            net = PPO.load(model_path, device="cpu"); label = "ppo (learned)"
        controllers[label] = lambda o, e: net.predict(o, deterministic=True)[0]

    for ood in (False, True):
        tag = "OUT-OF-DISTRIBUTION (loose harder than trained)" if ood else "IN-DISTRIBUTION"
        print(f"\n=== {tag} -- {episodes} eps ===")
        hdr = f"{'controller':24s}{'reach%':>8s}{'fwd(m)':>8s}{'energy':>8s}{'CoT':>8s}{'slip':>7s}"
        print(hdr); print("-" * len(hdr))
        for name, fn in controllers.items():
            with contextlib.redirect_stdout(io.StringIO()):
                r = run(env, fn, episodes, ood)
            print(f"{name:26s}{r['reach']*100:7.0f}%{r['fwd']:8.2f}{r['energy']:8.1f}{r['cot']:8.2f}{r['slip']:7.2f}")
    env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--model", default=os.path.join(HERE, "models", "ppo_range.zip"))
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    main(a.episodes, a.model, a.budget, a.steps)
