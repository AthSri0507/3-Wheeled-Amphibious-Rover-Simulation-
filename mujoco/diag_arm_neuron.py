"""
diag_arm_neuron.py  --  is the arm_trim output a dead neuron?

Reports per-terrain SIGNED mean arm, arm std (variance), and the correlation between the
terrain-probability inputs and the arm output. If arm is ~constant (std~0) and uncorrelated
with terrain, the neuron is effectively dead (stuck at a rail regardless of input).
"""
import contextlib, io, os
import numpy as np
from stable_baselines3 import PPO
import rover_sim; rover_sim.QUIET = True
import rover_env
import terrain as terr

HERE = os.path.dirname(os.path.abspath(__file__))
ARM_IDX = 2
PROB0 = 16   # obs[16:20] = terrain prob [hard, sand, gravel, water]
EPS_PER = 8


def main():
    env = rover_env.RoverEnv(seed=31)
    ppo = PPO.load(os.path.join(HERE, "models", "ppo_rover.zip"), device="cpu")

    print(f"{'terrain':8s}{'mean_arm':>10s}{'std_arm':>10s}{'mean|steer|':>12s}{'mean_drive':>11s}")
    print("-" * 51)
    all_probs, all_arm = [], []
    for cls in terr.CLASSES:
        env.set_allowed_classes([cls])
        arms, steers, drives = [], [], []
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(EPS_PER):
                obs, _ = env.reset()
                while True:
                    a = ppo.predict(obs, deterministic=True)[0]
                    arms.append(float(a[ARM_IDX])); steers.append(abs(float(a[1]))); drives.append(float(a[0]))
                    all_probs.append(obs[PROB0:PROB0 + 4].copy()); all_arm.append(float(a[ARM_IDX]))
                    obs, r, term, trunc, info = env.step(a)
                    if term or trunc:
                        break
        print(f"{cls:8s}{np.mean(arms):10.3f}{np.std(arms):10.3f}{np.mean(steers):12.3f}{np.mean(drives):11.3f}")

    all_probs = np.array(all_probs); all_arm = np.array(all_arm)
    print("\ncorrelation(arm output, terrain probability):")
    for i, c in enumerate(terr.CLASSES):
        p = all_probs[:, i]
        r = np.corrcoef(p, all_arm)[0, 1] if p.std() > 1e-9 and all_arm.std() > 1e-9 else float("nan")
        print(f"  P({c:7s}) vs arm:  r = {r:+.3f}")
    print(f"\noverall arm: mean={all_arm.mean():+.3f}  std={all_arm.std():.3f}  "
          f"min={all_arm.min():+.2f}  max={all_arm.max():+.2f}")


if __name__ == "__main__":
    main()
