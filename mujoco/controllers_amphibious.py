"""
controllers_amphibious.py  --  LAYER 2 within-mode controllers + episode runner.

Every controller obeys the same I/O contract: inputs = {terrain probs, mode, IMU, slip, velocity,
distance_to_waypoint, heading_error, cross_track_error}; outputs ONLY {drive, steer, arm} setpoints.
Mode switching is owned entirely by Layer 1 (mode_selector). The classifier->FSM stack is identical
for every controller; only the Layer-2 policy differs.

Controllers assembled here:
  oracle_upperbound  = privileged mode (submersion) + well-tuned control   (ceiling; NOT deployable)
  oracle_teacher     = classifier-FSM mode + well-tuned control            (BC teacher; student-obs only)
  classifier_fsm_rule= classifier-FSM mode + simple rule control           (deployable baseline)
  nonadaptive        = forced-land mode + simple rule control              (must fail the water regime)
"""
import numpy as np

import amphib
import rover_sim
from mode_selector import ModeSelector

HEAD_DEADBAND = 0.05   # rad; don't fight tiny heading errors (avoids jitter)
# Steering sign per mode (land/water). Differential-wheel and swivel-rudder yaw have opposite
# signs, and the FLIP (reverse drive) inverts both again -- so the correct signs are config-
# dependent and pinned here (calibrated empirically for the current FLIP setting).
LSIGN = 1.0    # land  (calibrated for FLIP=True)
WSIGN = 1.0    # water (calibrated for FLIP=True)

# Student/supervised observation vector (deployable inputs only; same info the rule sees).
OBS_KEYS = ["p_hard", "p_sand", "p_gravel", "p_water", "deploy", "ax", "ay", "az",
            "gx", "gy", "gz", "roll", "pitch", "vx", "vy", "slip", "dist", "heading_err", "cross_track"]


def obs_to_vec(obs):
    p = obs.get("probs", np.array([1.0, 0.0, 0.0, 0.0]))
    return np.array([p[0], p[1], p[2], p[3], obs["deploy"],
                     obs["acc"][0], obs["acc"][1], obs["acc"][2],
                     obs["gyro"][0], obs["gyro"][1], obs["gyro"][2],
                     obs["roll"], obs["pitch"], obs["vxy"][0], obs["vxy"][1],
                     obs["slip"], obs["dist"], obs["heading_err"], obs["cross_track"]], np.float32)


# ---------- Layer-2 policies: obs(dict) -> [drive, steer, arm] ----------
def waypoint_control(obs, k_head=1.2, k_cross=0.0):
    """Waypoint seeker. steer turns toward the goal; the swivel rudder in WATER yaws with the
    OPPOSITE sign to land differential steering, so the steer sign is flipped in water. Full
    drive is kept (the rover is slow and must hold momentum through the waterline lip)."""
    he = float(obs["heading_err"])
    if abs(he) < HEAD_DEADBAND:
        he = 0.0
    sign = WSIGN if obs["mode"] == rover_sim.WATER else LSIGN
    steer = sign * (k_head * he + k_cross * obs["cross_track"])
    steer = float(np.clip(steer, -1.0, 1.0))
    return np.array([1.0, steer, 0.0], np.float32)


def rule_layer2(obs):
    return waypoint_control(obs, k_head=1.5, k_cross=0.0)      # simple baseline


def tuned_layer2(obs):
    return waypoint_control(obs, k_head=2.0, k_cross=0.6)      # better: actively cancels cross-track


# ---------- assembled controllers (Layer 1 + Layer 2) ----------
class AmphibController:
    def __init__(self, mode_source, layer2, dt=0.004, name="", clf_path=None):
        self.ms = ModeSelector(mode_source, dt=dt, clf_path=clf_path)
        self.layer2 = layer2
        self.name = name

    def reset(self):
        self.ms.reset()

    def act(self, world, obs):
        probs = self.ms.update(world)          # Layer 1 sets the mode goal
        if probs is not None:
            obs["probs"] = np.asarray(probs)   # expose classifier probs to Layer 2
        return self.layer2(obs)                # Layer 2 emits setpoints


class SupervisedLayer2:
    """Layer-2 wrapper around a trained model: obs -> {drive, steer, arm} setpoints."""
    def __init__(self, model):
        self.model = model

    def __call__(self, obs):
        x = obs_to_vec(obs).reshape(1, -1)
        a = np.asarray(self.model.predict(x)).reshape(-1)[:3]
        return np.clip(a, [-1, -1, -1], [1, 1, 1]).astype(np.float32)


def make_supervised_controller(model, dt=0.004):
    return AmphibController("classifier", SupervisedLayer2(model), dt, "supervised")


def collect_demos(seeds, current_fn=lambda s: (s % 3) * 0.05):
    """Run oracle_teacher over randomized ROUND-TRIP courses (land->A->water->B->U-turn->back to A)
    so the student learns the FULL behaviour -- turning toward arbitrary waypoints, crossing both
    ways, the U-turn, and driving back OUT of the water -- and log (obs_vec, action) pairs for BC.
    Low current so the teacher reliably completes the hard water U-turn return (clean demos)."""
    X, Y = [], []
    for s in seeds:
        w = amphib.AmphibWorld(seed=s, current=float(current_fn(s)))
        ctrl = make_controller("oracle_teacher", dt=w.dt)
        cps = amphib.roundtrip_checkpoints(w.rng)
        obs = w.reset(checkpoints=cps); ctrl.reset()
        while True:
            probs = ctrl.ms.update(w)
            if probs is not None:
                obs["probs"] = np.asarray(probs)
            a = ctrl.layer2(obs)
            X.append(obs_to_vec(obs)); Y.append(np.asarray(a, np.float32))
            obs = w.step(a)
            _, all_done = w.advance_checkpoint()
            obs = w.observe()
            if all_done or w.t >= amphib.MAX_STEPS:
                break
            if abs(obs["roll"]) > amphib.FLIP_LIMIT or abs(obs["pitch"]) > amphib.FLIP_LIMIT:
                break
    return np.array(X, np.float32), np.array(Y, np.float32)


def make_controller(name, dt=0.004, clf_path=None):
    if name == "oracle_upperbound":
        return AmphibController("oracle", tuned_layer2, dt, name)
    if name == "oracle_teacher":
        return AmphibController("classifier", tuned_layer2, dt, name, clf_path=clf_path)
    if name == "classifier_fsm_rule":
        return AmphibController("classifier", rule_layer2, dt, name, clf_path=clf_path)
    if name == "nonadaptive":
        return AmphibController("forced_land", rule_layer2, dt, name)
    raise ValueError(name)


# ---------- episode runner + metrics ----------
def run_episode(world, controller, current=None, land_class=None, checkpoints=None):
    obs = world.reset(current=current, land_class=land_class, checkpoints=checkpoints)
    controller.reset()
    min_dist = obs["dist"]; stuck = 0; flipped = False; crossed_water = False
    max_roll = max_pitch = 0.0
    while True:
        a = controller.act(world, obs)
        obs = world.step(a)
        min_dist = min(min_dist, obs["dist"])
        if obs["speed"] < amphib.STUCK_SPEED:
            stuck += 1
        max_roll = max(max_roll, abs(obs["roll"])); max_pitch = max(max_pitch, abs(obs["pitch"]))
        if obs["pos"][1] < amphib.SHORE_Y:
            crossed_water = True
        if abs(obs["roll"]) > amphib.FLIP_LIMIT or abs(obs["pitch"]) > amphib.FLIP_LIMIT:
            flipped = True; reached = False; break
        _, all_done = world.advance_checkpoint()
        if all_done:
            reached = True; break
        obs = world.observe()                # refresh nav after a possible checkpoint switch
        if world.t >= amphib.MAX_STEPS:
            reached = False; break
    fail = ("flip" if flipped else "none" if reached else
            "no_water" if not crossed_water else "stuck" if stuck > 0.5 * world.t else "current_drift")
    return dict(reached=bool(reached), steps=world.t, min_dist=float(min_dist),
                crossed_water=bool(crossed_water), flipped=bool(flipped),
                stuck_frac=stuck / max(world.t, 1), max_roll=float(max_roll),
                max_pitch=float(max_pitch), energy=float(world.energy), fail=fail)
