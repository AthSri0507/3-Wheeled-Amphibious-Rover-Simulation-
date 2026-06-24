"""
rover_env.py  --  Gymnasium env for terrain-adaptive PPO control.

Wraps the MuJoCo rover, reusing rover_sim.RoverController for buoyancy + the automatic
land<->water transition, and terrain.TerrainManager for per-episode terrain. The policy
issues a HYBRID action (setpoints + normalized effort); the gross land/water config is
still owned by the auto-transition.

The priority of this file is a mathematically correct, NaN-safe, per-mode-normalized
EFFICIENCY reward with per-component logging (see RewardCfg + _reward). CoT is computed
for logging only and is never part of the reward.

Per-mode reward normalization uses running statistics that live on the env instance and
are NEVER persisted: debug runs use throwaway instances; PPO builds fresh envs whose
stats warm up on real rollouts.
"""
import math
from dataclasses import dataclass

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

import rover_sim
import terrain as terr

# ---- reward constants (named, per the spec) ----
EPS = 1e-3                      # m, distance floor in energy/(dist+EPS)
MAX_EFFICIENCY_PENALTY = 60.0   # cap on energy-per-metre so rare spikes can't dominate
SPEED_GATE = 0.02               # m/s; below this the efficiency penalty is gated to 0
WARMUP_N = 3000                 # per-mode samples with sigma=1 before std-normalization
NORM_CLIP = 10.0                # clip normalized components
FLOOR_FRAC = 0.20               # relative std floor: sigma >= FLOOR_FRAC*|mean| so a near-constant
                                # (saturated) signal can't blow up under scale-only normalization
W_PROG, W_EFF, W_STAB, W_ACT = 1.0, 1.00, 0.10, 0.05  # W_EFF raised 0.3->1.0 to chase lower CoT
W_ACT_MAG = 0.15                # action-MAGNITUDE penalty (steer^2+arm^2): discourages SITTING at the
                                # clip boundary (the action-rate term only penalizes CHANGE). Global.
ALIVE_BONUS = 0.02
SLIP_PEN_THRESH = 1.0           # |slip| above this adds to the stability penalty

# ---- episode / action mapping ----
MAX_STEPS = 1000                # 4 s at dt=0.004
ARM_TRIM_RANGE = 0.15           # rad of arm-tilt trim authority
FLIP_LIMIT = 1.3                # rad roll/pitch -> terminate
OBS_NOISE = 0.01

LAND, WATER = rover_sim.LAND, rover_sim.WATER

# --- navigation task: reach a laterally-offset goal on land (requires steering) ---
NAV_SUCCESS_R = 0.25           # m, reach radius
NAV_GOAL_BONUS = 8.0           # reward for reaching the goal
NAV_PROG_SCALE = 60.0          # scale on per-step goal-distance reduction (raw ~mm -> O(1))
NAV_STEPS = 2600               # nav episodes get more time to maneuver (rover is slow ~0.1 m/s)

# --- max-range task: drive forward across a mixed packed/loose course under an energy
# budget. The difficulty (loose vs packed) is HIDDEN under the same "sand" label, sampled
# per-episode and randomized in position so the policy must SENSE loose soil (emergent slip)
# and throttle down -- over-throttling loose depletes the budget early -> the episode ends
# sooner -> less total forward progress (return). No efficiency PENALTY: the budget is the
# mechanism. Calibrated from bench_throttle_sweep (packed opt=1.0, loose opt=0.6). ---
RANGE_GOAL = 1.5               # m forward to "complete"
# Calibrated (bench + eval_range sweep): packed-dominant course where, IN-DIST, fixed-0.8 ties the
# adaptive oracle (single throttle suffices) but OOD (harder loose) only the slip-sensing adaptive
# policy stays robust. PPO's job is to learn that adaptation -> match fixed in-dist, beat it OOD.
RANGE_BUDGET = 16.0           # J energy budget
RANGE_STEPS = 5800            # time limit
# PACKED-dominant course: long firm stretches (reward speed) punctuated by short loose
# patches (reward backing off). A fixed-low throttle is too slow over the packed majority
# (times out); a fixed-high throttle depletes on the loose patches. Only adaptation threads both.
RANGE_PACKED_LEN = (0.40, 0.60)
RANGE_LOOSE_LEN = (0.15, 0.28)
RANGE_GOAL_BONUS = 15.0
RANGE_LOOSE = (0.80, 1.00)   # in-distribution loose difficulty range
RANGE_LOOSE_OOD = (1.20, 1.45)  # out-of-distribution loose (harder than ever trained on)
RANGE_PACKED = (0.00, 0.20)
RANGE_PROG_SCALE = 20.0      # scale on RAW forward progress (full 1.5 m course -> ~30 reward)
# Raw per-step energy penalty: makes the efficiency tradeoff LOCAL so PPO doesn't need to
# credit-assign battery depletion thousands of steps ahead (gamma=0.99 can't). Calibrated
# (calib_energy_w) so the per-step reward gradient prefers full throttle on packed and ~0.6
# on loose, while crossing loose stays net-positive (so the policy keeps moving forward).
RANGE_ENERGY_W = 0.5

# --- transition-evaluation scenarios (used only via reset(options={"scenario": ...})) ---
# spawn=(x, y, z, yaw); a/b = terrain classes (b!=None -> temporal land blend a->b mid-episode);
# goal: "fwd_dist" (travel >= thr after the blend), "reach_water", or "reach_land".
SCENARIOS = {
    "hard_sand":       dict(spawn=(0.0,  0.30, 0.13, 0.0),     a="hard",  b="sand",   steps=1400, goal="fwd_dist", thr=0.30),
    "sand_gravel":     dict(spawn=(0.0,  0.30, 0.13, 0.0),     a="sand",  b="gravel", steps=1400, goal="fwd_dist", thr=0.25),
    "shoreline_water": dict(spawn=(0.0, -0.40, 0.13, 0.0),     a="hard",  b=None,     steps=2800, goal="reach_water"),
    "water_shoreline": dict(spawn=(0.0, -1.00, 0.10, np.pi),   a="water", b=None,     steps=3200, goal="reach_land"),
}


class RunningMeanStd:
    """Welford running mean/variance for a scalar stream."""
    __slots__ = ("n", "mean", "M2")

    def __init__(self):
        self.n = 0; self.mean = 0.0; self.M2 = 0.0

    def update(self, x):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.M2 += d * (x - self.mean)

    @property
    def std(self):
        return math.sqrt(self.M2 / self.n) if self.n > 1 else 1.0


class ModeNorm:
    """Per-mode (land/water) running std for each reward component; scale-only normalize."""
    COMPONENTS = ("prog", "eff", "stab", "act", "act_mag")

    def __init__(self):
        self.stats = {m: {c: RunningMeanStd() for c in self.COMPONENTS}
                      for m in (LAND, WATER)}

    def normalize(self, mode, comp, value, nonneg=False):
        rms = self.stats[mode][comp]
        rms.update(float(value))
        # relative std floor keeps a near-constant (e.g. saturated) signal from exploding
        sigma = max(rms.std, FLOOR_FRAC * abs(rms.mean)) if rms.n >= WARMUP_N else 1.0
        z = value / (sigma + 1e-8)
        return float(np.clip(z, 0.0 if nonneg else -NORM_CLIP, NORM_CLIP))

    def populated(self):
        return {m: min(self.stats[m][c].n for c in self.COMPONENTS) for m in (LAND, WATER)}


class RoverEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, allowed_classes=None, seed=None, max_steps=MAX_STEPS, task=None):
        super().__init__()
        rover_sim.ensure_terrain()
        self.m = mujoco.MjModel.from_xml_path(rover_sim.XML)
        self.d = mujoco.MjData(self.m)
        self.ctrl = rover_sim.RoverController(self.m, self.d)
        self.tm = terr.TerrainManager(self.m)
        self.rng = np.random.default_rng(seed)
        self.task = task                        # None | "nav" (adaptation-mandatory task modes)
        self.goal_xy = np.zeros(2)
        self.allowed_classes = list(allowed_classes or terr.CLASSES)
        self.water_prob = 0.30                  # P(spawn directly in water) when water is allowed
        self.max_steps = max_steps
        self._default_max_steps = max_steps
        self.scenario = None                    # set via reset(options={"scenario": name}) for eval
        self._range_ood = False                 # set via reset(options={"ood": True}) for OOD eval
        self.norm = ModeNorm()                 # persists across episodes; NOT persisted to disk

        self.rover_bid = self.m.body("rover").id
        self._sidx = {n: (int(self.m.sensor_adr[self.m.sensor(n).id]),
                          int(self.m.sensor_dim[self.m.sensor(n).id]))
                      for n in ("imu_acc", "imu_gyro", "imu_quat",
                                "fl_wheel_vel", "fr_wheel_vel", "rear_wheel_vel")}
        self.dt = float(self.m.opt.timestep)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)  # drive, steer, arm
        obs0 = self._reset_internal()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs0.shape, dtype=np.float32)

    def set_allowed_classes(self, classes):
        """Curriculum hook: restrict which terrain classes reset() may sample."""
        self.allowed_classes = list(classes)

    def set_water_prob(self, p):
        """Curriculum hook: probability of spawning directly in water (water oversampling)."""
        self.water_prob = float(p)

    # ---------- terrain / spawn ----------
    def _reset_internal(self):
        if self.scenario:
            sc = SCENARIOS[self.scenario]
            self.max_steps = sc["steps"]
            self.terrain_spec = terr.sample_spec(self.rng, allowed=[sc["a"]])
            self.tm.apply(self.terrain_spec, self.rng)
            x, y, z, yaw = sc["spawn"]
            mujoco.mj_resetData(self.m, self.d)
            self.d.qpos[0:3] = [x, y, z]
            self.d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        elif self.task == "nav":
            self.max_steps = NAV_STEPS
            cls = str(self.rng.choice([c for c in self.allowed_classes if c != "water"] or ["hard"]))
            self.terrain_spec = terr.sample_spec(self.rng, allowed=[cls])
            self.tm.apply(self.terrain_spec, self.rng)
            mujoco.mj_resetData(self.m, self.d)
            self.d.qpos[0:3] = [0.0, 0.40, 0.13]
            self.d.qpos[3:7] = [1, 0, 0, 0]
            # goal offset laterally (|x|>=0.3 forces a turn), close enough to reach on land
            gx = float(self.rng.uniform(0.3, 0.7)) * (1 if self.rng.random() < 0.5 else -1)
            gy = float(self.rng.uniform(-0.3, 0.2))
            self.goal_xy = np.array([gx, gy])
        elif self.task == "range":
            self.max_steps = RANGE_STEPS
            self._range_budget = RANGE_BUDGET
            self._range_zones = self._make_zones(self._range_ood)
            self._range_reached = False
            self.terrain_spec = terr.difficulty_spec("sand", 0.0)   # label "sand"; diff set per step
            self.tm.apply(self.terrain_spec, self.rng)
            mujoco.mj_resetData(self.m, self.d)
            self.d.qpos[0:3] = [0.0, 0.40, 0.13]
            self.d.qpos[3:7] = [1, 0, 0, 0]
        else:
            self.max_steps = self._default_max_steps
            water_first = self.rng.random() < self.water_prob and "water" in self.allowed_classes
            cls = "water" if water_first else str(self.rng.choice(self.allowed_classes))
            tprob = 0.3 if cls != "water" else 0.0
            self.terrain_spec = terr.sample_spec(self.rng, allowed=[cls], transition_prob=tprob)
            self.tm.apply(self.terrain_spec, self.rng)
            mujoco.mj_resetData(self.m, self.d)
            self.d.qpos[0:3] = [0.0, -1.9, 0.10] if self.terrain_spec.is_water else [0.0, 0.15, 0.13]
            self.d.qpos[3:7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.m, self.d)

        self.ctrl.__init__(self.m, self.d)
        if self.terrain_spec.is_water:
            self.ctrl.deploy = 1.0
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.prev_xy = self.d.xpos[self.rover_bid][0:2].copy()
        self.t = 0
        self.ep = dict(progress=0.0, efficiency=0.0, stability=0.0, action=0.0,
                       total=0.0, distance=0.0, energy=0.0)
        self._label = self._soft_label()
        # scenario tracking
        self._scn_start = self.d.xpos[self.rover_bid][0:2].copy()
        fwd0 = -self.d.xmat[self.rover_bid].reshape(3, 3)[:, 1][0:2]
        self._scn_fwd = fwd0 / (np.linalg.norm(fwd0) + 1e-9)
        self._scn_reached = False
        self._scn_stuck = 0
        self._prev_goal_dist = float(np.linalg.norm(self.goal_xy - self.prev_xy))
        self._nav_reached = False
        return self._obs()

    def _make_zones(self, ood=False):
        """Randomized alternating packed/loose zones along the forward path. Positions and
        which-zone-is-loose are randomized per episode so the policy must SENSE loose soil
        (slip) rather than memorize where it is."""
        loose_rng = RANGE_LOOSE_OOD if ood else RANGE_LOOSE
        zones, dist, k = [], 0.0, 0
        start_packed = True   # always begin on firm ground (rover spawns on packed)
        while dist < RANGE_GOAL + 0.6:
            is_loose = ((k % 2 == 1) == start_packed)
            length = float(self.rng.uniform(*(RANGE_LOOSE_LEN if is_loose else RANGE_PACKED_LEN)))
            diff = float(self.rng.uniform(*(loose_rng if is_loose else RANGE_PACKED)))
            zones.append((dist, dist + length, diff))
            dist += length; k += 1
        return zones

    def _range_netfwd(self):
        return float((self.d.xpos[self.rover_bid][0:2] - self._scn_start) @ self._scn_fwd)

    def _range_diff_at(self, net_fwd):
        for lo, hi, diff in self._range_zones:
            if lo <= net_fwd < hi:
                return diff
        return self._range_zones[-1][2]

    def _goal_relative(self):
        """Unit direction to goal in the rover's body frame: [forward_comp, lateral_comp]."""
        R = self.d.xmat[self.rover_bid].reshape(3, 3)
        fwd = -R[:, 1][0:2]; right = R[:, 0][0:2]
        to_goal = self.goal_xy - self.d.xpos[self.rover_bid][0:2]
        u = to_goal / (np.linalg.norm(to_goal) + 1e-9)
        return np.array([u @ (fwd / (np.linalg.norm(fwd) + 1e-9)),
                         u @ (right / (np.linalg.norm(right) + 1e-9))], dtype=np.float32)

    def _soft_label(self):
        v = self.terrain_spec.label_vector(0.0) * 0.9 + 0.1 / terr.N_CLASSES
        return (v / v.sum()).astype(np.float32)

    # ---------- sensors / obs ----------
    def _sens(self, k):
        a, n = self._sidx[k]
        return self.d.sensordata[a:a + n]

    def _roll_pitch(self):
        w, x, y, z = self._sens("imu_quat")
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1, min(1, 2 * (w * y - z * x))))
        return roll, pitch

    def _power(self):
        return float(np.sum(np.abs(self.d.actuator_force * self.d.actuator_velocity)))

    def _obs(self):
        acc = self._sens("imu_acc") + self.rng.standard_normal(3) * OBS_NOISE
        gyr = self._sens("imu_gyro") + self.rng.standard_normal(3) * OBS_NOISE
        roll, pitch = self._roll_pitch()
        vxy = self.d.qvel[0:2]
        wheels = np.array([self._sens("fl_wheel_vel")[0], self._sens("fr_wheel_vel")[0],
                           self._sens("rear_wheel_vel")[0]])
        speed = float(np.linalg.norm(vxy))
        slip = float(np.clip((wheels.mean() * 0.031 - speed) / max(speed, 0.05), -3, 3))
        obs = np.concatenate([
            acc, gyr, [roll, pitch], vxy, wheels, [slip],
            [self._power()], [self.ctrl.deploy], self._label, self.prev_action,
        ]).astype(np.float32)
        if self.task == "nav":     # append goal direction (body frame) -> obs dim 25
            obs = np.concatenate([obs, self._goal_relative()]).astype(np.float32)
        elif self.task == "range":  # append remaining energy + remaining distance -> obs dim 25
            e_left = max(0.0, 1.0 - self.ep["energy"] / self._range_budget)
            g_left = max(0.0, 1.0 - self._range_netfwd() / RANGE_GOAL)
            obs = np.concatenate([obs, [e_left, g_left]]).astype(np.float32)
        return obs

    # ---------- action -> actuators ----------
    def _apply_action(self, a):
        drive, steer, arm = float(a[0]), float(a[1]), float(a[2])
        c = self.ctrl
        c.update_transition()                      # sets deploy/mode + tilt/prop targets
        if c.mode == LAND:
            base = drive * rover_sim.DRIVE_SPEED
            turn = steer * rover_sim.PIVOT_SPEED
            c._set("front_left_wheel", np.clip(base - turn, -rover_sim.MAX_VEL, rover_sim.MAX_VEL))
            c._set("front_right_wheel", np.clip(base + turn, -rover_sim.MAX_VEL, rover_sim.MAX_VEL))
            c._set("rear_wheel", base)
        else:  # WATER
            c._set("front_left_wheel", 0.0)
            c._set("front_right_wheel", 0.0)
            c._set("rear_wheel", drive * rover_sim.DRIVE_SPEED)
            c.swv = steer * rover_sim.SWIVEL_MAX
            c._set("rear_swivel", c.swv)
        # arm trim (clamped by actuator ctrlrange)
        c._set("front_right_tilt", c.d.ctrl[c.aid["front_right_tilt"]] + arm * ARM_TRIM_RANGE)
        c._set("front_left_tilt", c.d.ctrl[c.aid["front_left_tilt"]] + arm * ARM_TRIM_RANGE)

    # ---------- reward ----------
    def _reward(self, action):
        d = self.d
        xy = d.xpos[self.rover_bid][0:2]
        dxy = xy - self.prev_xy
        dist = float(np.linalg.norm(dxy))
        # forward heading = rover body -Y axis, projected to xy
        fwd = -d.xmat[self.rover_bid].reshape(3, 3)[:, 1][0:2]
        nf = np.linalg.norm(fwd)
        fwd = fwd / nf if nf > 1e-9 else np.zeros(2)
        prog = float(dxy @ fwd)
        if self.task == "nav":     # progress = reduction in distance to the goal (requires turning)
            gd = float(np.linalg.norm(self.goal_xy - xy))
            prog = (self._prev_goal_dist - gd) * NAV_PROG_SCALE
            self._prev_goal_dist = gd
        speed = float(np.linalg.norm(d.qvel[0:2]))
        energy = self._power() * self.dt

        eff = energy / (dist + EPS)
        eff = min(eff, MAX_EFFICIENCY_PENALTY)
        if speed < SPEED_GATE:
            eff = 0.0

        roll, pitch = self._roll_pitch()
        wheels = np.array([self._sens("fl_wheel_vel")[0], self._sens("fr_wheel_vel")[0],
                           self._sens("rear_wheel_vel")[0]])
        slip = (wheels.mean() * 0.031 - speed) / max(speed, 0.05)
        stability = roll * roll + pitch * pitch + max(0.0, abs(slip) - SLIP_PEN_THRESH)
        action_rate = float(np.sum((action - self.prev_action) ** 2))      # jitter (covers steer & arm)
        # magnitude penalty on STEER only: steer's optimum is ~0 (go straight). Arm is NOT penalized by
        # magnitude because its optimum is a non-zero trim (water sweep: best at arm~=-0.5, beats baseline)
        # -- penalizing |arm| would push it to 0 and cap performance below the achievable optimum.
        action_mag = float(action[1] ** 2)

        mode = self.ctrl.mode
        prog_n = self.norm.normalize(mode, "prog", prog)
        eff_n = self.norm.normalize(mode, "eff", eff, nonneg=True)
        stab_n = self.norm.normalize(mode, "stab", stability, nonneg=True)
        act_n = self.norm.normalize(mode, "act", action_rate, nonneg=True)
        actmag_n = self.norm.normalize(mode, "act_mag", action_mag, nonneg=True)

        if self.task == "nav":
            # navigation REQUIRES big steering -> drop the steer-magnitude + efficiency penalties
            # (those were tuned for efficient straight driving and would punish the needed maneuvering)
            comp = dict(progress=W_PROG * prog_n, efficiency=0.0,
                        stability=-W_STAB * stab_n, action=-W_ACT * act_n)
        elif self.task == "range":
            # reward RAW forward distance (+ a dominant goal bonus in step()) -- NOT normalized.
            # Normalizing per-step progress rewards slow survival (each step ~O(1) summed over a
            # long episode), decoupling reward from real distance -> the policy creeps and never
            # reaches the goal. Raw distance: creeping/depleting => low distance => low reward;
            # the energy budget still forces backing off on loose soil to travel farther.
            comp = dict(progress=RANGE_PROG_SCALE * prog, efficiency=-RANGE_ENERGY_W * energy,
                        stability=0.0, action=0.0)
        else:
            comp = dict(progress=W_PROG * prog_n, efficiency=-W_EFF * eff_n,
                        stability=-W_STAB * stab_n,
                        action=-(W_ACT * act_n + W_ACT_MAG * actmag_n))
        # no alive bonus on the range task: it would reward slow survival over reaching the goal
        alive = 0.0 if self.task == "range" else ALIVE_BONUS
        total = comp["progress"] + comp["efficiency"] + comp["stability"] + comp["action"] + alive

        # accumulate for episode-level logging
        self.ep["progress"] += comp["progress"]; self.ep["efficiency"] += comp["efficiency"]
        self.ep["stability"] += comp["stability"]; self.ep["action"] += comp["action"]
        self.ep["total"] += total; self.ep["distance"] += dist; self.ep["energy"] += energy
        self.prev_xy = xy.copy()
        return float(total)

    # ---------- gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)               # sets self._np_random (required by env_checker)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.scenario = (options or {}).get("scenario")
        self._range_ood = bool((options or {}).get("ood", False))
        return self._reset_internal(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).clip(-1, 1)
        self._apply_action(action)
        # scenario land terrain blend a->b across the middle of the episode
        if self.scenario and SCENARIOS[self.scenario]["b"]:
            sc = SCENARIOS[self.scenario]
            t0, dur = 0.4 * self.max_steps, 0.3 * self.max_steps
            frac = float(np.clip((self.t - t0) / dur, 0.0, 1.0))
            self.tm.apply_scalars(terr.lerp_spec(self.terrain_spec, sc["b"], frac, self.rng))
        # max-range course: apply the hidden packed/loose difficulty for the current position
        if self.task == "range":
            diff = self._range_diff_at(self._range_netfwd())
            self.tm.apply_scalars(terr.difficulty_spec("sand", diff))
        self.ctrl.apply_water_forces()
        if self.ctrl.mode == WATER:     # disturbance whenever floating (returns 0 unless spec is water)
            self.d.xfrc_applied[self.rover_bid, 0:2] += self.tm.water_force(self.terrain_spec, self.rng)
        mujoco.mj_step(self.m, self.d)
        self.t += 1

        reward = self._reward(action)
        self.prev_action = action
        roll, pitch = self._roll_pitch()
        pos = self.d.xpos[self.rover_bid]
        terminated = bool(abs(roll) > FLIP_LIMIT or abs(pitch) > FLIP_LIMIT or
                          abs(pos[0]) > 3.7 or pos[1] > 3.7 or pos[1] < -3.9)
        if self.task == "nav" and float(np.linalg.norm(self.goal_xy - pos[0:2])) < NAV_SUCCESS_R:
            reward += NAV_GOAL_BONUS
            self._nav_reached = True
            terminated = True
        if self.task == "range":
            if self._range_netfwd() >= RANGE_GOAL:
                reward += RANGE_GOAL_BONUS
                self._range_reached = True
                terminated = True
            elif self.ep["energy"] >= self._range_budget:   # battery depleted before the goal
                terminated = True
        truncated = self.t >= self.max_steps
        obs = self._obs()
        info = {}

        if self.scenario:
            if float(np.linalg.norm(self.d.qvel[0:2])) < SPEED_GATE:
                self._scn_stuck += 1
            goal = SCENARIOS[self.scenario]["goal"]
            if goal == "reach_water" and self.ctrl.deploy > 0.6 and pos[1] < -0.9:
                self._scn_reached = True
            elif goal == "reach_land" and self.ctrl.deploy < 0.3 and pos[1] > -0.55:
                self._scn_reached = True

        if (terminated or truncated) and self.scenario:
            sc = SCENARIOS[self.scenario]
            net_fwd = float((pos[0:2] - self._scn_start) @ self._scn_fwd)
            completed = (net_fwd >= sc["thr"]) if sc["goal"] == "fwd_dist" else self._scn_reached
            info["scenario"] = dict(name=self.scenario, completed=bool(completed),
                                    net_fwd=net_fwd, stuck_frac=self._scn_stuck / max(self.t, 1))

        if (terminated or truncated) and self.task == "nav":
            info["nav"] = dict(reached=bool(self._nav_reached), steps=self.t,
                               goal_dist=float(np.linalg.norm(self.goal_xy - pos[0:2])))

        if (terminated or truncated) and self.task == "range":
            info["range"] = dict(reached=bool(self._range_reached), steps=self.t,
                                 net_fwd=self._range_netfwd(), energy=self.ep["energy"],
                                 ood=self._range_ood)

        if terminated or truncated:
            mass = float(np.sum(self.m.body_mass))
            cot = self.ep["energy"] / (mass * self.ep["distance"] + 1e-6)
            info["episode_components"] = dict(
                progress_reward=self.ep["progress"], efficiency_penalty=self.ep["efficiency"],
                stability_penalty=self.ep["stability"], action_penalty=self.ep["action"],
                total_reward=self.ep["total"], distance=self.ep["distance"],
                energy=self.ep["energy"], CoT=cot, terrain=self.terrain_spec.cls)
        if not np.all(np.isfinite(obs)) or not math.isfinite(reward):
            terminated = True
            reward = 0.0 if not math.isfinite(reward) else reward
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return obs, reward, terminated, truncated, info
