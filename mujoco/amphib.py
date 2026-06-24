"""
amphib.py  --  Amphibious terrain-adaptive control benchmark (harness + Layer-1 + Layer-2).

Self-contained: drives the MuJoCo rover directly via RoverController (rover_sim.py UNMODIFIED),
so mode switching uses the existing request_water()/request_land() override. The classifier->FSM
mode selector is Layer 1 (frozen, identical for every controller); Layer 2 emits only the
{drive, steer, arm} setpoints. PID-style setpoints are handled by the existing position/velocity
actuators.

Course (full amphibious traversal):
  spawn on land (+Y) -> drive -Y across land -> cross the shoreline (y < SHORE) into the water
  -> traverse the basin under a lateral CURRENT -> reach a floating WAYPOINT.

A controller that does not switch to water config CANNOT propel in the water (wheels in the water,
no propeller) -> fails the regime. The classifier-FSM controller deploys correctly -> traverses.
"""
import os
import numpy as np
import mujoco

import rover_sim
import terrain as terr

HERE = os.path.dirname(os.path.abspath(__file__))
rover_sim.QUIET = True

# ---- course geometry (world frame; forward = -Y) ----
SPAWN = (0.0, 0.7, 0.13)          # short land stretch before the shoreline
WAYPOINT = (0.0, -1.7)            # in the water basin (~1.1 m past the shore)
SHORE_Y = rover_sim.WATER_REGION[3]   # -0.55: land/water boundary
WP_RADIUS = 0.45                  # reach radius
MAX_STEPS = 22000                 # ~88 s (slow rover + deep-water round trip)


def make_checkpoints(rng):
    """A randomized 4-checkpoint course from the land spawn into the water, with lateral
    detours that force the rover to TURN (not a straight line). Last checkpoint is in the water."""
    sx = 1.0 if rng.random() < 0.5 else -1.0       # which way the first turn goes
    return [
        np.array([sx * rng.uniform(0.30, 0.45), 0.15]),    # land, turn one way
        np.array([-sx * rng.uniform(0.25, 0.40), -0.70]),  # near shore, turn back
        np.array([sx * rng.uniform(0.20, 0.35), -1.25]),   # in the water
        np.array([0.0, -1.70]),                            # final waypoint
    ]


def roundtrip_checkpoints(rng):
    """The round-trip demo course: spawn -> land checkpoint A (a turn, not straight) -> into the
    WATER -> water checkpoint B (another turn) -> U-turn -> back across the shoreline (water->land
    transition) -> back to land checkpoint A. Exercises BOTH transitions + a U-turn."""
    sx = 1.0 if rng.random() < 0.5 else -1.0
    A = np.array([sx * rng.uniform(0.40, 0.50), 0.15])             # land, offset from spawn (forces a turn)
    B = np.array([-sx * rng.uniform(0.30, 0.45), rng.uniform(-2.0, -1.7)])  # FAR into the water basin
    return [A, B, A.copy()]                                        # A -> B -> back to A
FLIP_LIMIT = 1.3                  # rad roll/pitch -> flip (terminate)
ARM_TRIM_RANGE = 0.15
STUCK_SPEED = 0.02

# ---- Layer-2 action mapping (mirrors the proven keyboard/env mapping) ----
DRIVE_SPEED = rover_sim.DRIVE_SPEED
PIVOT_SPEED = rover_sim.PIVOT_SPEED
SWIVEL_MAX = rover_sim.SWIVEL_MAX
MAX_VEL = rover_sim.MAX_VEL

# The model's propeller sits at the -Y body end and thrusts -Y, so driving -Y leads with the
# single propeller wheel (a "puller"). FLIP drives the rover +Y-body-first instead: the two
# wheels lead and the propeller TRAILS and pushes from behind (a realistic "pusher"). Done
# purely in this harness (rover_sim.py / rover.xml untouched): spawn yawed 180 deg, define the
# nose as +Y body, and reverse the wheel/propeller commands.
FLIP = True
SPAWN_QUAT = [0.0, 0.0, 0.0, 1.0] if FLIP else [1.0, 0.0, 0.0, 0.0]   # yaw 180 vs identity
FWD_SIGN = 1.0 if FLIP else -1.0      # body +Y is the nose when flipped
DRIVE_SIGN = -1.0 if FLIP else 1.0    # reverse wheels/propeller to move +Y-body-first


def _q2rp(q):
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return float(roll), float(pitch)


class AmphibWorld:
    """Owns the model/data/controller and the course; steps the sim under a controller."""

    def __init__(self, seed=0, current=0.5, land_class="sand"):
        rover_sim.ensure_terrain()
        self.m = mujoco.MjModel.from_xml_path(rover_sim.XML)
        self.d = mujoco.MjData(self.m)
        self.ctrl = rover_sim.RoverController(self.m, self.d)
        self.tm = terr.TerrainManager(self.m)
        self.rng = np.random.default_rng(seed)
        self.current = current               # lateral water-current magnitude (N)
        self.land_class = land_class
        self.rover_bid = self.m.body("rover").id
        self.dt = float(self.m.opt.timestep)
        self._sidx = {n: (int(self.m.sensor_adr[self.m.sensor(n).id]),
                          int(self.m.sensor_dim[self.m.sensor(n).id]))
                      for n in ("imu_acc", "imu_gyro", "imu_quat",
                                "fl_wheel_vel", "fr_wheel_vel", "rear_wheel_vel")}
        self.wp = np.array(WAYPOINT)

    def _sens(self, k):
        a, n = self._sidx[k]
        return self.d.sensordata[a:a + n]

    def reset(self, current=None, land_class=None, checkpoints=None):
        if current is not None:
            self.current = current
        if land_class is not None:
            self.land_class = land_class
        # checkpoints: a sequence the rover visits in order (default = single final waypoint).
        self.checkpoints = [np.asarray(c, float) for c in (checkpoints or [WAYPOINT])]
        self.cp_idx = 0
        self.wp = self.checkpoints[0]
        spec = terr.sample_spec(self.rng, allowed=[self.land_class])
        self.tm.apply(spec, self.rng)
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[0:3] = SPAWN
        self.d.qpos[3:7] = SPAWN_QUAT
        mujoco.mj_forward(self.m, self.d)
        self.ctrl.__init__(self.m, self.d)
        # random lateral current direction (held for the episode)
        self._cur_dir = 1.0 if self.rng.random() < 0.5 else -1.0
        self.t = 0
        self.start_xy = self.d.xpos[self.rover_bid][0:2].copy()
        self.energy = 0.0
        return self.observe()

    # ---- observation pieces (Layer-2 + classifier raw frame) ----
    def raw_frame(self):
        acc = self._sens("imu_acc"); gyr = self._sens("imu_gyro")
        roll, pitch = _q2rp(self._sens("imu_quat"))
        # present wheels in the FORWARD convention (negate under flip) so the classifier/slip see
        # the same signal distribution they were trained on (driving forward), not reversed wheels.
        wl = DRIVE_SIGN * float(self._sens("fl_wheel_vel")[0])
        wr = DRIVE_SIGN * float(self._sens("fr_wheel_vel")[0])
        wrear = DRIVE_SIGN * float(self._sens("rear_wheel_vel")[0])
        speed = float(np.linalg.norm(self.d.qvel[0:2]))
        return np.array([self.t * self.dt, acc[0], acc[1], acc[2], gyr[0], gyr[1], gyr[2],
                         roll, pitch, wl, wr, wrear, speed], dtype=np.float64)

    def nav_features(self):
        p = self.d.xpos[self.rover_bid][0:2]
        R = self.d.xmat[self.rover_bid].reshape(3, 3)
        fwd = FWD_SIGN * R[:, 1][0:2]; fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
        right = FWD_SIGN * R[:, 0][0:2]; right = right / (np.linalg.norm(right) + 1e-9)
        to_wp = self.wp - p
        dist = float(np.linalg.norm(to_wp))
        u = to_wp / (dist + 1e-9)
        heading_err = float(np.arctan2(u @ right, u @ fwd))   # signed angle to waypoint
        # cross-track: perpendicular distance from the start->waypoint line
        line = self.wp - self.start_xy; line = line / (np.linalg.norm(line) + 1e-9)
        cross = float(np.cross(line, p - self.start_xy))
        return dict(dist=dist, heading_err=heading_err, cross_track=cross)

    def observe(self):
        nav = self.nav_features()
        roll, pitch = _q2rp(self._sens("imu_quat"))
        vxy = self.d.qvel[0:2].copy()
        wheels = DRIVE_SIGN * np.array([self._sens("fl_wheel_vel")[0], self._sens("fr_wheel_vel")[0],
                                        self._sens("rear_wheel_vel")[0]])
        speed = float(np.linalg.norm(vxy))
        slip = float(np.clip((wheels.mean() * 0.031 - speed) / max(speed, 0.05), -3, 3))
        return dict(mode=self.ctrl.mode, deploy=self.ctrl.deploy, roll=roll, pitch=pitch,
                    vxy=vxy, speed=speed, slip=slip, acc=self._sens("imu_acc").copy(),
                    gyro=self._sens("imu_gyro").copy(), pos=self.d.xpos[self.rover_bid][0:2].copy(),
                    **nav)

    # ---- apply a Layer-2 [drive, steer, arm] action (mirrors the env mapping) ----
    def _apply(self, action):
        drive, steer, arm = float(action[0]), float(action[1]), float(action[2])
        c = self.ctrl
        c.update_transition()                     # respects manual mode set by the FSM
        if c.mode == rover_sim.LAND:
            base = DRIVE_SIGN * drive * DRIVE_SPEED; turn = steer * PIVOT_SPEED
            c._set("front_left_wheel", np.clip(base - turn, -MAX_VEL, MAX_VEL))
            c._set("front_right_wheel", np.clip(base + turn, -MAX_VEL, MAX_VEL))
            c._set("rear_wheel", base)
        else:
            c._set("front_left_wheel", 0.0); c._set("front_right_wheel", 0.0)
            c._set("rear_wheel", DRIVE_SIGN * drive * DRIVE_SPEED)   # reverse prop -> thrust from behind
            c.swv = steer * SWIVEL_MAX; c._set("rear_swivel", c.swv)
        c._set("front_right_tilt", c.d.ctrl[c.aid["front_right_tilt"]] + arm * ARM_TRIM_RANGE)
        c._set("front_left_tilt", c.d.ctrl[c.aid["front_left_tilt"]] + arm * ARM_TRIM_RANGE)

    def step(self, action):
        self._apply(action)
        self.ctrl.apply_water_forces()
        if self.ctrl.mode == rover_sim.WATER:     # lateral current while floating
            self.d.xfrc_applied[self.rover_bid, 0] += self._cur_dir * self.current
        mujoco.mj_step(self.m, self.d)
        self.t += 1
        self.energy += float(np.sum(np.abs(self.d.actuator_force * self.d.actuator_velocity))) * self.dt
        return self.observe()

    def advance_checkpoint(self):
        """If the current checkpoint is reached, advance to the next. Returns (reached_now, all_done)."""
        if np.linalg.norm(self.wp - self.d.xpos[self.rover_bid][0:2]) >= WP_RADIUS:
            return False, False
        if self.cp_idx >= len(self.checkpoints) - 1:
            return True, True                       # final checkpoint reached
        self.cp_idx += 1
        self.wp = self.checkpoints[self.cp_idx]
        self.start_xy = self.d.xpos[self.rover_bid][0:2].copy()   # reset cross-track baseline per leg
        return True, False
