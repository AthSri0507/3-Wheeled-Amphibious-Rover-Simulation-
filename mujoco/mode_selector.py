"""
mode_selector.py  --  LAYER 1: terrain classifier + FSM mode selector (FROZEN, deployable).

Owns ALL terrain identification and the land/water decision. Drives RoverController via the
existing request_water()/request_land()/auto() override -- rover_sim.py is NOT modified. This
stack is identical for every Layer-2 controller in the comparison.

mode_source:
  "classifier" -> rolling window -> frozen RF -> water-prob -> FSM (hysteresis) -> request_*  (DEPLOYABLE)
  "oracle"     -> privileged submersion auto-transition (ctrl.auto())                         (upper bound only)
  "forced_land"/"forced_water" -> non-adaptive baselines
"""
import os
from collections import deque

import numpy as np
import joblib

import features as F

HERE = os.path.dirname(os.path.abspath(__file__))
CLF_PATH = os.path.join(HERE, "models", "terrain_clf.joblib")

# FSM hysteresis on water probability. The classifier is very confident in CLEAR water (~0.85)
# but only ~0.25-0.30 at the transitional waterline (rover half in water, churning); dry land is
# ~0.05. We deploy on the early/ambiguous signal (>HI) so the rover commits to water config and
# can propel through the lip, and retract only when clearly back on land (<LO).
WATER_HI = 0.22
WATER_LO = 0.12
CLASSIFY_EVERY = 25  # run the classifier at ~10 Hz (dt=0.004 -> every 25 steps); FSM needs no more


class ModeSelector:
    def __init__(self, mode_source="classifier", dt=0.004):
        self.mode_source = mode_source
        self.dt = dt
        if mode_source == "classifier":
            blob = joblib.load(CLF_PATH)
            self.pipe = blob["pipe"]; self.win_sec = blob["win_sec"]; self.fs = blob["fs"]
            self.classes = list(blob["classes"]); self.feat_names = blob["feature_names"]
            self.water_idx = self.classes.index("water")
            self.win_env = max(8, round(self.win_sec / dt))     # frames at sim rate
            self.win_train = max(8, round(self.win_sec * self.fs))
            self.buf = deque(maxlen=self.win_env)
        self.reset()

    def reset(self):
        self._water = False           # FSM latch
        self._k = 0
        self.probs = np.array([1.0, 0.0, 0.0, 0.0])   # land-biased default until the buffer fills
        if self.mode_source == "classifier":
            self.buf.clear()

    def _classify(self):
        """Resample the sim-rate window to the classifier's training rate, then predict_proba."""
        win = np.array(self.buf)                            # (win_env, RAW_COLS)
        idx = np.linspace(0, len(win) - 1, self.win_train)
        res = np.empty((self.win_train, win.shape[1]))
        for c in range(win.shape[1]):
            res[:, c] = np.interp(idx, np.arange(len(win)), win[:, c])
        x = F.features_vector(res, fs=self.fs).reshape(1, -1)
        return self.pipe.predict_proba(x)[0]               # in self.classes order

    def update(self, world):
        """Run Layer 1 for this step: set the rover's mode goal. Returns terrain probs (4-vec)."""
        c = world.ctrl
        if self.mode_source == "oracle":
            c.auto()                                        # privileged submersion decision
            return None
        if self.mode_source == "forced_land":
            c.request_land(); return None
        if self.mode_source == "forced_water":
            c.request_water(); return None
        # --- classifier + FSM ---
        self.buf.append(world.raw_frame())
        self._k += 1
        if len(self.buf) >= self.win_env and self._k % CLASSIFY_EVERY == 0:
            self.probs = self._classify()
        wp = float(self.probs[self.water_idx])
        if wp > WATER_HI:
            self._water = True
        elif wp < WATER_LO:
            self._water = False
        c.request_water() if self._water else c.request_land()
        return self.probs
