"""
rollout_log.py  --  one replay dataset format for sim AND the future real robot.

A rollout is a list of per-step records with a fixed schema; both the simulator and the
real rover log to it, so real data drops straight into classifier retraining, PPO
fine-tuning, and offline RL without reformatting.

Schema (per step):
  timestamp, observation, terrain_probs, action, reward_components, power, pose, terrain_truth
"""
import numpy as np

SCHEMA = ["timestamp", "observation", "terrain_probs", "action",
          "reward_components", "power", "pose", "terrain_truth"]


class RolloutLogger:
    def __init__(self):
        self.rows = []

    def add(self, timestamp, observation, terrain_probs, action,
            reward_components, power, pose, terrain_truth):
        self.rows.append(dict(
            timestamp=float(timestamp),
            observation=np.asarray(observation, np.float32),
            terrain_probs=np.asarray(terrain_probs, np.float32),
            action=np.asarray(action, np.float32),
            reward_components=dict(reward_components),
            power=float(power),
            pose=np.asarray(pose, np.float32),
            terrain_truth=str(terrain_truth),
        ))

    def __len__(self):
        return len(self.rows)

    def save(self, path):
        np.savez_compressed(
            path,
            timestamp=np.array([r["timestamp"] for r in self.rows], np.float32),
            observation=np.stack([r["observation"] for r in self.rows]),
            terrain_probs=np.stack([r["terrain_probs"] for r in self.rows]),
            action=np.stack([r["action"] for r in self.rows]),
            power=np.array([r["power"] for r in self.rows], np.float32),
            pose=np.stack([r["pose"] for r in self.rows]),
            terrain_truth=np.array([r["terrain_truth"] for r in self.rows]),
            reward_components=np.array([r["reward_components"] for r in self.rows], dtype=object),
            schema=np.array(SCHEMA),
        )

    @staticmethod
    def load(path):
        return np.load(path, allow_pickle=True)
