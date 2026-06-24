"""
train_sac.py  --  SAC on the SAME range task as PPO (clean algorithm-only comparison).

SAC is off-policy (replay buffer -> sample-efficient) with entropy-regularized exploration
and a learned temperature -- the standard choice for continuous-control locomotion, and a
direct test of whether PPO's collapse on this task was the ALGORITHM (A/B: per-step-noise
exploration, local optima) or the TASK (D: near-flat reward / weak advantage signal).

Same env, same reward (RoverEnv task="range"), same observation/action space as train_ppo.

Usage:  python train_sac.py --steps 700000
"""
import argparse
import os

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

import rover_env

HERE = os.path.dirname(os.path.abspath(__file__))


def make_env(seed):
    import rover_sim
    rover_sim.QUIET = True
    return rover_env.RoverEnv(allowed_classes=["hard", "sand", "gravel"], seed=seed, task="range")


class ComponentLogCallback(BaseCallback):
    def _on_step(self):
        for info in self.locals.get("infos", []):
            c = info.get("episode_components")
            if c:
                for k in ("progress_reward", "efficiency_penalty", "total_reward",
                          "distance", "energy", "CoT"):
                    self.logger.record_mean(f"reward/{k}", float(c[k]))
            r = info.get("range")
            if r:
                self.logger.record_mean("range/reached", float(r["reached"]))
                self.logger.record_mean("range/net_fwd", float(r["net_fwd"]))
        return True


def main(steps, seed, init=None):
    torch.set_num_threads(8)
    env = make_env(seed)
    if init and os.path.exists(init):
        print(f"continuing from {init}")
        model = SAC.load(init, env=env, device="cpu")
        model.learning_starts = 2_000   # quickly refill buffer (weights preserved)
    else:
        model = SAC(
            "MlpPolicy", env, device="cpu", verbose=1, seed=seed,
            learning_rate=3e-4, buffer_size=300_000, learning_starts=5_000,
            batch_size=256, tau=0.005, gamma=0.99, train_freq=2, gradient_steps=1,  # ~2x CPU throughput
            ent_coef="auto", policy_kwargs=dict(net_arch=[128, 128]),  # match PPO; faster CPU updates
            tensorboard_log=os.path.join(HERE, "tb"),
        )
    model.learn(total_timesteps=steps, callback=ComponentLogCallback(), progress_bar=False,
                log_interval=10, reset_num_timesteps=not bool(init))
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    out = os.path.join(HERE, "models", "sac_range.zip")
    model.save(out)
    print(f"\nsaved {out}")
    env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=700_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="continue from this checkpoint")
    a = ap.parse_args()
    main(a.steps, a.seed, a.init)
