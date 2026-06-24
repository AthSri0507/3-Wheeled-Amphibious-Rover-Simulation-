"""
rover_sim.py  --  Amphibious Triangular Rover, MuJoCo port
==========================================================
Port of the Webots rover (rover_terrain.wbt + rover_controller.py) to MuJoCo.

Joint / control conventions (identical to the Webots controller):
  front_right_tilt : axis Y   land = 0.0   water = -1.5708 (arm swings right)
  front_left_tilt  : axis Y   land = 0.0   water = +1.5708 (arm swings left)
  rear_swivel      : axis Z (yaw)  steers the rear pod left/right
  rear_prop_tilt   : axis Z   land = 0.0   water = +1.5708 (wheel -> propeller)
  *_wheel          : axis X   spins the drive wheels

MuJoCo has no free water surface, so buoyancy + hydro-drag + propeller thrust
are applied as external forces (data.xfrc_applied) inside RoverController.step().
A rectangular water region (WATER_REGION) at height WATER_Z defines where the
forces act, so the rover drives on land and floats when it enters the water.

Run:
  python rover_sim.py            # interactive viewer (keyboard)
  python rover_sim.py --demo     # scripted land+water demo
  python rover_sim.py --demo --headless   # demo with no window (prints telemetry)

The land<->water transition is AUTOMATIC: just drive off the beach (-Y) into the water
and the arms/propeller deploy as the hull starts to float (and retract when you drive
back out). T/G/F are optional manual overrides.

INTERACTIVE KEYS (latched -- a press stays in effect until you change it).
Driving uses the ARROW keys (W/A/S/D are reserved by the MuJoCo viewer):
  LAND :  Up/Down drive fwd/back   Left/Right pivot turn   Q/E arc turn   Space stop
  WATER:  Up/Down propel (rear prop only)  Left/Right steer (rear-pod swivel)  Space stop/centre
  ANY  :  T force-deploy water   G force-retract land   F back to automatic
"""

import os
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "rover.xml")


def ensure_terrain():
    """rover.xml needs beach.png (the heightfield) in the project root; build it if absent."""
    png = os.path.join(os.path.dirname(HERE), "beach.png")
    if not os.path.exists(png):
        sys.path.insert(0, HERE)
        import make_terrain
        make_terrain.main()

# ----------------------------------------------------------------------------
# Tuning (mirrors rover_controller.py)
# ----------------------------------------------------------------------------
WHEEL_RADIUS = 0.031
MAX_VEL      = 8.0

DRIVE_SPEED  = 4.0     # rad/s straight
PIVOT_SPEED  = 1.5     # rad/s pivot
ARC_OUTER    = 4.0
ARC_INNER    = 1.6

TILT_R_LAND  =  0.0
TILT_R_WATER = -1.5708
TILT_L_LAND  =  0.0
TILT_L_WATER =  1.5708

PROP_LAND    =  0.0
PROP_WATER   =  1.5708

SWIVEL_MAX   =  0.70    # max rear-pod steering angle (rad)
SWIVEL_RATE  =  2.0     # how fast A/D ramps the swivel toward the target (rad/s)
SWIVEL_STEP  =  0.04    # (land helpers; unused by water steering now)

LAND, WATER = "LAND", "WATER"

QUIET = False  # set True to silence transition prints (RL training)

# GLFW key codes for the arrow keys (W/A/S/D are reserved by the MuJoCo viewer)
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 262, 263, 264, 265
_ARROWS = (KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP)

# ----------------------------------------------------------------------------
# Water model
# ----------------------------------------------------------------------------
RHO   = 1000.0          # water density kg/m^3
GRAV  = 9.81
WATER_Z = 0.0           # water surface height (= top of the water geom in rover.xml)
WATER_REGION = (-4.0, 4.0, -4.0, -0.55)   # xmin, xmax, ymin, ymax (basin/beach side, -Y)

# --- amphibious auto-transition (smooth land<->water hand-off) ---
DEPLOY_RATE = 0.7       # how fast the amphibious config deploys/retracts (fraction per s)
SUB_DEPLOY  = 0.18      # chassis submerged-fraction at which it auto-deploys to water
SUB_RETRACT = 0.05      # ...and below which it auto-retracts to land (hysteresis)

# per-body effective displaced volume (m^3) and vertical half-extent (m).
# Volumes are intentionally generous (treats the chassis/pod as a hull) so the
# rover floats with its body near the surface instead of nearly fully submerged.
BUOY = {
    "rover":            dict(vol=2.0e-3, h=0.05),
    "front_right_arm":  dict(vol=0.15e-3, h=0.04),
    "front_left_arm":   dict(vol=0.15e-3, h=0.04),
    "rear_pod":         dict(vol=0.5e-3,  h=0.05),
    "front_right_wheel":dict(vol=0.05e-3, h=0.031),
    "front_left_wheel": dict(vol=0.05e-3, h=0.031),
    "rear_prop_wheel":  dict(vol=0.05e-3, h=0.031),
}
DRAG_LIN  = 6.0         # linear drag coefficient
DRAG_QUAD = 8.0         # quadratic drag coefficient
DRAG_ANG  = 0.4         # rotational drag (keeps the hull from rolling/yawing wildly)
THRUST_GAIN = 0.20      # N per rad/s of rear-wheel speed (propeller)
STEER_GAIN  = 0.08      # yaw torque per (swivel_rad * prop_speed): outboard rudder effect
COB_LIFT  = 0.05        # centre-of-buoyancy height above each body origin (m).
                        # CoB above CoM gives a righting moment so the hull stays upright.


def clamp(v, lo=-MAX_VEL, hi=MAX_VEL):
    return max(lo, min(hi, v))


class RoverController:
    """Holds the LAND/WATER state machine, writes actuator ctrl, applies water forces."""

    def __init__(self, model, data):
        self.m = model
        self.d = data

        self.aid = {n: model.actuator(n).id for n in (
            "front_left_wheel", "front_right_wheel", "rear_wheel",
            "front_right_tilt", "front_left_tilt", "rear_swivel", "rear_prop_tilt")}

        self.bid = {n: model.body(n).id for n in BUOY}
        self.rover_bid = model.body("rover").id
        self.pod_bid   = model.body("rear_pod").id
        self.rwheel_jadr = model.joint("rear_wheel").dofadr[0]

        self.mode = LAND
        self.cmd  = "stop"      # latched drive command
        self.steer = "none"     # latched water steering: none | left | right
        self.lv = 0.0
        self.rv = 0.0
        self.swv = 0.0

        # amphibious transition state
        self.deploy = 0.0       # 0 = land config, 1 = full water config (ramps smoothly)
        self.manual = None      # None = automatic; "water"/"land" = forced override
        self.auto_water = False # hysteresis latch for the automatic decision

        # start in land configuration
        self._set("front_right_tilt", TILT_R_LAND)
        self._set("front_left_tilt",  TILT_L_LAND)
        self._set("rear_prop_tilt",   PROP_LAND)
        self._set("rear_swivel", 0.0)

    # ----- actuator helpers -------------------------------------------------
    def _set(self, name, val):
        self.d.ctrl[self.aid[name]] = val

    def drive(self, l, r):
        self.lv = clamp(l)
        self.rv = clamp(r)

    def push_motors(self):
        if self.mode == LAND:
            self._set("front_left_wheel", self.lv)
            self._set("front_right_wheel", self.rv)
            # lock rear wheel during a pivot so it does not fight the turn
            self._set("rear_wheel", 0.0 if (self.lv * self.rv < 0) else (self.lv + self.rv) * 0.5)
        else:
            # WATER: only the submerged rear wheel (propeller) drives; the front
            # wheels are splayed up out of the water, so they stay idle.
            self._set("front_left_wheel", 0.0)
            self._set("front_right_wheel", 0.0)
            self._set("rear_wheel", (self.lv + self.rv) * 0.5)   # full prop thrust

    # ----- amphibious transition --------------------------------------------
    def request_water(self):
        self.manual = "water"
        QUIET or print("[manual] forcing WATER deploy")

    def request_land(self):
        self.manual = "land"
        QUIET or print("[manual] forcing LAND retract")

    def auto(self):
        self.manual = None
        QUIET or print("[manual] cleared -> automatic land/water transition")

    def chassis_submersion(self):
        """Fraction of the chassis below the water surface (0 on land, ->1 fully under)."""
        p = BUOY["rover"]
        com = self.d.xipos[self.rover_bid]
        xmin, xmax, ymin, ymax = WATER_REGION
        if not (xmin < com[0] < xmax and ymin < com[1] < ymax):
            return 0.0
        frac = (WATER_Z - (com[2] - p["h"])) / (2.0 * p["h"])
        return min(max(frac, 0.0), 1.0)

    def update_transition(self):
        """Smoothly deploy/retract the amphibious config and pick LAND vs WATER mode."""
        if   self.manual == "water": goal = 1.0
        elif self.manual == "land":  goal = 0.0
        else:                                       # automatic, with hysteresis
            sub = self.chassis_submersion()
            if   sub > SUB_DEPLOY:  self.auto_water = True
            elif sub < SUB_RETRACT: self.auto_water = False
            goal = 1.0 if self.auto_water else 0.0

        # ramp the deployment fraction toward the goal at a limited rate
        step = DEPLOY_RATE * self.m.opt.timestep
        self.deploy = min(max(self.deploy + max(-step, min(step, goal - self.deploy)), 0.0), 1.0)

        # drive the arm/prop joints from the (smoothly changing) deployment fraction
        self._set("front_right_tilt", TILT_R_LAND + self.deploy * (TILT_R_WATER - TILT_R_LAND))
        self._set("front_left_tilt",  TILT_L_LAND + self.deploy * (TILT_L_WATER - TILT_L_LAND))
        self._set("rear_prop_tilt",   PROP_LAND   + self.deploy * (PROP_WATER   - PROP_LAND))

        # propulsion mode flips at the half-way point
        new_mode = WATER if self.deploy > 0.5 else LAND
        if new_mode != self.mode:
            if new_mode == WATER:
                QUIET or print("[~> WATER]  afloat: arms spread, propeller deployed")
            else:
                QUIET or print("[~> LAND ]  beached: arms down, wheels rolling")
                self.swv = 0.0; self.steer = "none"; self._set("rear_swivel", 0.0)
        self.mode = new_mode

    # ----- input: latched command from a single key press -------------------
    def on_key(self, key):
        # arrow keys drive & steer (UP/DOWN = fwd/back; LEFT/RIGHT = turn or steer)
        if key in _ARROWS:
            if   key == KEY_UP:    self.cmd = "fwd"
            elif key == KEY_DOWN:  self.cmd = "back"
            elif key == KEY_LEFT:
                if self.mode == LAND: self.cmd = "left"
                else:                 self.steer = "left"   # rear pod swings, craft yaws left
            elif key == KEY_RIGHT:
                if self.mode == LAND: self.cmd = "right"
                else:                 self.steer = "right"
            return

        try:
            c = chr(key).upper()
        except ValueError:
            return
        # transition overrides (optional; normally the transition is automatic)
        if c == "T": self.request_water(); return
        if c == "G": self.request_land();  return
        if c == "F": self.auto();          return
        if c == " ": self.cmd = "stop"; self.steer = "none"; return
        # arc turns are land-only
        if self.mode == LAND:
            if   c == "Q": self.cmd = "arc_left"
            elif c == "E": self.cmd = "arc_right"

    # ----- apply the latched command to wheel velocities --------------------
    def apply_command(self):
        if self.mode == LAND:
            if   self.cmd == "fwd":       self.drive(DRIVE_SPEED, DRIVE_SPEED)
            elif self.cmd == "back":      self.drive(-DRIVE_SPEED, -DRIVE_SPEED)
            elif self.cmd == "left":      self.drive(-PIVOT_SPEED,  PIVOT_SPEED)
            elif self.cmd == "right":     self.drive( PIVOT_SPEED, -PIVOT_SPEED)
            elif self.cmd == "arc_left":  self.drive(ARC_INNER, ARC_OUTER)
            elif self.cmd == "arc_right": self.drive(ARC_OUTER, ARC_INNER)
            else:  # stop -> coast down
                self.lv *= 0.75; self.rv *= 0.75
                if abs(self.lv) < 0.04: self.lv = 0.0
                if abs(self.rv) < 0.04: self.rv = 0.0
        else:  # WATER
            if   self.cmd == "fwd":  self.drive(DRIVE_SPEED, DRIVE_SPEED)
            elif self.cmd == "back": self.drive(-DRIVE_SPEED, -DRIVE_SPEED)
            else:
                self.lv *= 0.85; self.rv *= 0.85
            # ramp the rear-pod swivel toward the latched steering target (or recentre)
            step = SWIVEL_RATE * self.m.opt.timestep
            if   self.steer == "left":  self.swv = clamp(self.swv - step, -SWIVEL_MAX, SWIVEL_MAX)
            elif self.steer == "right": self.swv = clamp(self.swv + step, -SWIVEL_MAX, SWIVEL_MAX)
            else:                       self.swv *= 0.92          # gently re-centre
            self._set("rear_swivel", self.swv)

    # ----- water forces -----------------------------------------------------
    def apply_water_forces(self):
        d, m = self.d, self.m
        d.xfrc_applied[:] = 0.0
        xmin, xmax, ymin, ymax = WATER_REGION
        vel6 = np.zeros(6)

        for name, p in BUOY.items():
            bid = self.bid[name]
            com = d.xipos[bid]               # body COM, world frame
            if not (xmin < com[0] < xmax and ymin < com[1] < ymax):
                continue
            bottom = com[2] - p["h"]
            frac = (WATER_Z - bottom) / (2.0 * p["h"])
            frac = min(max(frac, 0.0), 1.0)
            if frac <= 0.0:
                continue

            # buoyancy (up), applied at a centre-of-buoyancy above the body origin.
            # Applying an off-COM force = force at COM + torque (r x F); with the CoB
            # above the COM this yields a righting moment that keeps the hull upright.
            R = d.xmat[bid].reshape(3, 3)
            cob = d.xpos[bid] + R @ np.array([0.0, 0.0, COB_LIFT])
            r = cob - com
            F = np.array([0.0, 0.0, RHO * GRAV * p["vol"] * frac])
            d.xfrc_applied[bid, 0:3] += F
            d.xfrc_applied[bid, 3:6] += np.cross(r, F)

            # hydro drag opposing COM linear velocity
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, vel6, 0)
            v = vel6[3:6]
            speed = np.linalg.norm(v)
            d.xfrc_applied[bid, 0:3] += -(DRAG_LIN + DRAG_QUAD * speed) * frac * v

        # rotational drag on the chassis (keeps it from spinning wildly in water)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.rover_bid, vel6, 0)
        d.xfrc_applied[self.rover_bid, 3:6] += -DRAG_ANG * vel6[0:3]

        # propeller thrust: rear wheel spinning while deployed drives the craft.
        # Forward is the rover's -Y (pod local -Y in world) so that W moves the SAME
        # direction on land and in water (consistent through the transition).
        if self.mode == WATER:
            pod_com = d.xipos[self.pod_bid]
            if xmin < pod_com[0] < xmax and ymin < pod_com[1] < ymax and pod_com[2] - 0.05 < WATER_Z:
                fwd = -d.xmat[self.pod_bid].reshape(3, 3)[:, 1]   # pod local -Y in world
                wheel_spin = d.qvel[self.rwheel_jadr]
                thrust = THRUST_GAIN * wheel_spin
                d.xfrc_applied[self.pod_bid, 0:3] += thrust * fwd
                # rudder/outboard steering: swivelling the deployed prop yaws the hull
                d.xfrc_applied[self.rover_bid, 5] += STEER_GAIN * self.swv * wheel_spin

    # ----- one control update (call every sim step) -------------------------
    def step(self):
        self.update_transition()   # smooth land<->water deploy + pick mode
        self.apply_command()
        self.push_motors()
        self.apply_water_forces()

    # ----- telemetry --------------------------------------------------------
    def telemetry(self):
        d = self.d
        quat = d.xquat[self.rover_bid]
        w, x, y, z = quat
        roll  = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
        pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
        spd = (self.lv + self.rv) * 0.5 * WHEEL_RADIUS
        pos = d.xpos[self.rover_bid]
        return (f"[{self.mode}] pos=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:.2f}) "
                f"spd={spd:+.2f}m/s deploy={self.deploy*100:3.0f}% swivel={np.degrees(self.swv):+.0f}deg "
                f"roll={roll:+.0f} pitch={pitch:+.0f}")


# ============================================================================
# Entry points
# ============================================================================
def run_interactive():
    ensure_terrain()
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    ctrl = RoverController(m, d)

    print(__doc__.split("Run:")[0])
    print("  Interactive viewer running. Click the window, then use the keys above.\n")

    with mujoco.viewer.launch_passive(m, d, key_callback=ctrl.on_key) as viewer:
        last = time.time()
        tick = 0
        while viewer.is_running():
            step_start = time.time()
            ctrl.step()
            mujoco.mj_step(m, d)
            viewer.sync()

            tick += 1
            if time.time() - last > 1.0:
                last = time.time()
                print(ctrl.telemetry())

            dt = m.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


def run_demo(headless=False):
    ensure_terrain()
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    ctrl = RoverController(m, d)

    # scripted phases: (label, duration_s, action(ctrl)). The rover starts on land and
    # simply drives forward (-Y) into the water; the land<->water transition is automatic.
    phases = [
        ("LAND: drive down the beach toward the water", 13.0, lambda c: setattr(c, "cmd", "fwd")),
        ("WATER: keep propelling once afloat",           4.0, lambda c: setattr(c, "cmd", "fwd")),
        ("WATER: steer right",                           4.0, lambda c: (setattr(c, "cmd", "fwd"),
                                                                         setattr(c, "steer", "right"))),
        ("WATER: steer left",                            4.0, lambda c: (setattr(c, "cmd", "fwd"),
                                                                         setattr(c, "steer", "left"))),
        ("reverse back up the beach (auto-retracts to land)", 21.0, lambda c: (setattr(c, "cmd", "back"),
                                                                               setattr(c, "steer", "none"))),
        ("LAND: back ashore, stop",                      3.0, lambda c: setattr(c, "cmd", "stop")),
    ]

    def run_loop(viewer=None):
        for label, dur, action in phases:
            print(f"\n>>> {label}")
            action(ctrl)
            n = int(dur / m.opt.timestep)
            for i in range(n):
                t0 = time.time()
                ctrl.step()
                mujoco.mj_step(m, d)
                if viewer is not None:
                    viewer.sync()
                    dt = m.opt.timestep - (time.time() - t0)
                    if dt > 0:
                        time.sleep(dt)
                if i % int(0.5 / m.opt.timestep) == 0:
                    print("    " + ctrl.telemetry())
                if viewer is not None and not viewer.is_running():
                    return

    if headless:
        run_loop(None)
        print("\nDemo finished (headless).")
    else:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            run_loop(viewer)
            print("\nDemo finished. Close the window to exit.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)


if __name__ == "__main__":
    if "--demo" in sys.argv:
        run_demo(headless="--headless" in sys.argv)
    else:
        run_interactive()
