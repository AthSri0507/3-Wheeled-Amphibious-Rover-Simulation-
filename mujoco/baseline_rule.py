"""
baseline_rule.py  --  deliberately simple terrain-aware rule controller.

Only job: show that directed locomotion beats random. Constant forward throttle scaled
per terrain, zero steering, nominal arm trim. NOT optimized (and not meant to be).
"""
import numpy as np

# forward throttle per terrain (lower on loose/rough ground to limit slip)
THROTTLE = {"hard": 1.0, "sand": 0.8, "gravel": 0.7, "water": 1.0}


class RuleController:
    def act(self, terrain_cls):
        return np.array([THROTTLE.get(terrain_cls, 0.8), 0.0, 0.0], dtype=np.float32)
