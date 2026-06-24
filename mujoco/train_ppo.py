"""
train_ppo.py  --  PPO V1 (stock single MlpPolicy) for terrain-adaptive control.

CPU only (device="cpu"); SubprocVecEnv is the throughput lever (physics is CPU-bound).
Includes the terrain curriculum (hard -> +sand -> +gravel -> +water) via a callback and
per-component reward logging to TensorBoard. Each env builds a FRESH per-mode reward
normalizer (no debug stats persisted); it warms up on these real rollouts.

Label-source annealing (privileged -> classifier terrain probs) is a follow-up increment;
V1 trains on privileged terrain to first prove the reward + control loop learns.

Usage:
  python train_ppo.py --steps 200000 --envs 8           # smoke / validation
  python train_ppo.py --steps 3000000 --envs 12         # full run
"""
import argparse
import math
import os

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

import rover_env

ARM_IDX = 2

HERE = os.path.dirname(os.path.abspath(__file__))

# terrain curriculum stages: (fraction of total steps, allowed classes, water spawn prob).
# Water enters EARLIER (0.40) and the final stage OVERSAMPLES water (0.65) to test whether
# the water deficit is an exposure/curriculum issue rather than an architecture limit.
CURRICULUM = [
    (0.00, ["hard"], 0.0),
    (0.15, ["hard", "sand"], 0.0),
    (0.30, ["hard", "sand", "gravel"], 0.0),
    (0.40, ["hard", "sand", "gravel", "water"], 0.50),
    (0.60, ["hard", "sand", "gravel", "water"], 0.65),
]


def make_env(rank, seed, task=None):
    def _f():
        import rover_sim
        rover_sim.QUIET = True                 # silence transition prints in workers
        if task:
            return rover_env.RoverEnv(allowed_classes=["hard", "sand", "gravel"],
                                      seed=seed + rank, task=task)
        return rover_env.RoverEnv(allowed_classes=CURRICULUM[0][1], seed=seed + rank)
    return _f


def _stage(frac):
    return max(i for i, c in enumerate(CURRICULUM) if frac >= c[0])


class CurriculumCallback(BaseCallback):
    def __init__(self, total_steps):
        super().__init__()
        self.total = total_steps
        self.stage = -1

    def _on_step(self):
        stage = _stage(self.num_timesteps / self.total)
        if stage != self.stage:
            self.stage = stage
            _, classes, wprob = CURRICULUM[stage]
            self.training_env.env_method("set_allowed_classes", classes)
            self.training_env.env_method("set_water_prob", wprob)
            if self.verbose:
                print(f"[curriculum] step {self.num_timesteps}: terrains -> {classes} water_prob={wprob}")
        return True


class ArmStdFloor(BaseCallback):
    """Per-dimension exploration floor: keep arm's policy log_std >= log(floor) so PPO can't
    stop exploring the arm dimension (the action-rate penalty was collapsing sigma_arm->0)."""
    def __init__(self, floor):
        super().__init__(); self.lo = math.log(floor)

    def _on_step(self):
        with torch.no_grad():
            self.model.policy.log_std.data[ARM_IDX].clamp_(min=self.lo)
        return True


class ComponentLogCallback(BaseCallback):
    """Log per-episode reward components (from env info) to TensorBoard."""
    def _on_step(self):
        for info in self.locals.get("infos", []):
            c = info.get("episode_components")
            if c:
                for k in ("progress_reward", "efficiency_penalty", "stability_penalty",
                          "action_penalty", "total_reward", "distance", "energy", "CoT"):
                    self.logger.record_mean(f"reward/{k}", float(c[k]))
        return True


def main(steps, n_envs, seed, sde=False, arm_explore=0.0, task=None):
    venv = SubprocVecEnv([make_env(i, seed, task) for i in range(n_envs)])
    venv = VecMonitor(venv)

    pk = dict(net_arch=[128, 128])
    extra = {}
    if sde:  # gSDE + tanh-bounded actions: removes the unbounded-Gaussian clip-saturation pathology
        extra = dict(use_sde=True, sde_sample_freq=4)
        pk["squash_output"] = True
    model = PPO(
        "MlpPolicy", venv, device="cpu", verbose=1, seed=seed,
        n_steps=1024, batch_size=2048, gae_lambda=0.95, gamma=0.99,
        learning_rate=3e-4, ent_coef=0.01, clip_range=0.2, n_epochs=10,
        policy_kwargs=pk, tensorboard_log=os.path.join(HERE, "tb"), **extra,
    )
    cbs = [ComponentLogCallback()]
    if not task:                       # terrain curriculum only for the base driving task
        cbs.insert(0, CurriculumCallback(steps))
    if arm_explore > 0:
        cbs.append(ArmStdFloor(arm_explore))
        print(f"[arm-explore] flooring arm log_std at log({arm_explore})")
    model.learn(total_timesteps=steps, callback=cbs, progress_bar=False)

    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    out = os.path.join(HERE, "models", f"ppo_{task}.zip" if task else "ppo_rover.zip")
    model.save(out)
    print(f"\nsaved {out}")
    venv.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sde", action="store_true", help="gSDE + tanh-bounded actions")
    ap.add_argument("--arm-explore", type=float, default=0.0,
                    help="floor arm policy std at this value (e.g. 0.35) to revive the dead arm dim")
    ap.add_argument("--task", default=None, help="adaptation task mode (e.g. nav)")
    a = ap.parse_args()
    main(a.steps, a.envs, a.seed, sde=a.sde, arm_explore=a.arm_explore, task=a.task)
