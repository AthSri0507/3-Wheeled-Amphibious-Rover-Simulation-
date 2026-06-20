"""
rover_controller.py  -  Amphibious Triangular Rover v4
========================================================
Joint conventions (from WBT):
  front_right_tilt_joint : axis Y, anchor right side
    land  = 0.0     (arm hangs straight down, wheel on ground)
    water = +1.5708 (arm swings outward to the RIGHT, 90 deg)

  front_left_tilt_joint  : axis Y, Solid has Z-rotation 3.1416 (mirrored)
    land  = 0.0
    water = -1.5708 (arm swings outward to the LEFT, 90 deg)

  rear_swivel_joint      : axis Z (yaw), steers pod left/right

  rear_prop_tilt_joint   : axis Z (yaw)
    land  = 0.0     (wheel hangs down, rolls along Y = forward)
    water = +1.5708 (wheel yaws 90 deg, face now points backward along -Y)
                    spinning it pushes water backward = forward propulsion
                    exactly like a boat outboard motor

  rear_wheel_joint       : axis X (spins the wheel)

CONTROLS:
  LAND:  W/S = forward/back   A/D = pivot turn   W+A/D = arc turn   T = water
  WATER: W/S = propulsor      A/D = swivel steer  Space = stop       G = land
"""

from controller import Robot, Keyboard

# ── Tuning ──────────────────────────────────────────────────────────────────
WHEEL_RADIUS  = 0.031
MAX_VEL       = 8.0

DRIVE_SPEED   = 2.0    # rad/s straight
PIVOT_SPEED   = 0.6    # rad/s pivot (slow = no tipping)
ARC_OUTER     = 2.0
ARC_INNER     = 0.8

# arm positions
TILT_R_LAND   =  0.0
TILT_R_WATER  = -1.5708   # right arm swings right (outward)
TILT_L_LAND   =  0.0
TILT_L_WATER  =  1.5708   # left  arm swings left  (outward)

# propeller tilt (Z-axis yaw)
PROP_LAND     =  0.0      # wheel face points DOWN  → rolls on ground
PROP_WATER    =  1.5708   # wheel face points BACK  → pushes water rearward

SWIVEL_MAX    =  0.50
SWIVEL_STEP   =  0.03

LAND  = 'LAND'
WATER = 'WATER'

# ── Init ────────────────────────────────────────────────────────────────────
robot    = Robot()
timestep = int(robot.getBasicTimeStep())

fl         = robot.getDevice('front_left_wheel_joint')
fr         = robot.getDevice('front_right_wheel_joint')
rw         = robot.getDevice('rear_wheel_joint')
fl_tilt    = robot.getDevice('front_left_tilt_joint')
fr_tilt    = robot.getDevice('front_right_tilt_joint')
swivel     = robot.getDevice('rear_swivel_joint')
prop_tilt  = robot.getDevice('rear_prop_tilt_joint')

# velocity mode for drive wheels
for m in (fl, fr, rw):
    m.setPosition(float('inf'))
    m.setVelocity(0.0)

# Set all joints to land position at startup
fr_tilt.setPosition(TILT_R_LAND)
fl_tilt.setPosition(TILT_L_LAND)
swivel.setPosition(0.0)
prop_tilt.setPosition(PROP_LAND)

# Sensors
fl_ws  = robot.getDevice('front_left_wheel_sensor');  fl_ws.enable(timestep)
fr_ws  = robot.getDevice('front_right_wheel_sensor'); fr_ws.enable(timestep)
fl_ts  = robot.getDevice('front_left_tilt_sensor');   fl_ts.enable(timestep)
rs_ts  = robot.getDevice('rear_swivel_sensor');       rs_ts.enable(timestep)
pt_ts  = robot.getDevice('rear_prop_tilt_sensor');    pt_ts.enable(timestep)
imu    = robot.getDevice('imu');                      imu.enable(timestep)

kb = Keyboard()
kb.enable(timestep)

# ── State ────────────────────────────────────────────────────────────────────
mode    = LAND
lv      = 0.0   # left wheel velocity
rv      = 0.0   # right wheel velocity
swv     = 0.0   # swivel angle
tick    = 0

print("=" * 60)
print("  Amphibious Rover v4")
print("  LAND : W/S=drive  A/D=pivot  W+A/D=arc  T=water mode")
print("  WATER: W/S=propulsor  A/D=steer  G=land mode  Spc=stop")
print("  >> Click 3D viewport before pressing keys! <<")
print("=" * 60)


def clamp(v, lo=-MAX_VEL, hi=MAX_VEL):
    return max(lo, min(hi, v))


def drive(l, r):
    global lv, rv
    lv = clamp(l)
    rv = clamp(r)


def push_motors():
    fl.setVelocity(lv)
    fr.setVelocity(rv)
    if mode == LAND:
        # lock rear during pivot so it doesn't resist the turn
        rw.setVelocity(0.0 if (lv * rv < 0) else (lv + rv) * 0.5)
    else:
        rw.setVelocity((lv + rv) * 0.5)   # full prop thrust in water


def enter_water():
    global mode, swv
    mode = WATER
    fr_tilt.setPosition(TILT_R_WATER)   # right arm swings right →
    fl_tilt.setPosition(TILT_L_WATER)   # left  arm swings left  ←
    prop_tilt.setPosition(PROP_WATER)   # rear wheel yaws 90° → propeller
    drive(0, 0)
    print("[MODE → WATER]  arms spread ←→  |  prop: ship-propeller mode")


def enter_land():
    global mode, swv
    mode = LAND
    fr_tilt.setPosition(TILT_R_LAND)
    fl_tilt.setPosition(TILT_L_LAND)
    prop_tilt.setPosition(PROP_LAND)    # rear wheel yaws back → rolling wheel
    swv = 0.0
    swivel.setPosition(0.0)
    drive(0, 0)
    print("[MODE → LAND ]  arms down  |  prop: rolling wheel")


# ── Main loop ────────────────────────────────────────────────────────────────
while robot.step(timestep) != -1:

    # --- read ALL keys held this step ---
    keys = set()
    k = kb.getKey()
    while k != -1:
        keys.add(k)
        k = kb.getKey()

    W   = ord('W') in keys
    S   = ord('S') in keys
    A   = ord('A') in keys
    D   = ord('D') in keys
    SPC = ord(' ') in keys
    T   = ord('T') in keys
    G   = ord('G') in keys

    # ─ LAND ──────────────────────────────────────────────────────────────────
    if mode == LAND:
        if T:
            enter_water()
        elif SPC:
            drive(0, 0)
        elif W and A:
            drive(ARC_INNER, ARC_OUTER)
        elif W and D:
            drive(ARC_OUTER, ARC_INNER)
        elif S and A:
            drive(-ARC_INNER, -ARC_OUTER)
        elif S and D:
            drive(-ARC_OUTER, -ARC_INNER)
        elif W:
            drive(DRIVE_SPEED, DRIVE_SPEED)
        elif S:
            drive(-DRIVE_SPEED, -DRIVE_SPEED)
        elif A:
            drive(-PIVOT_SPEED,  PIVOT_SPEED)
        elif D:
            drive( PIVOT_SPEED, -PIVOT_SPEED)
        else:
            lv *= 0.75
            rv *= 0.75
            if abs(lv) < 0.04: lv = 0.0
            if abs(rv) < 0.04: rv = 0.0

    # ─ WATER ─────────────────────────────────────────────────────────────────
    elif mode == WATER:
        if G:
            enter_land()
        elif SPC:
            drive(0, 0)
            swv *= 0.85
            swivel.setPosition(swv)
        else:
            if W:
                drive(DRIVE_SPEED, DRIVE_SPEED)
            elif S:
                drive(-DRIVE_SPEED, -DRIVE_SPEED)
            else:
                lv *= 0.85
                rv *= 0.85

            if A:
                swv = clamp(swv - SWIVEL_STEP, -SWIVEL_MAX, SWIVEL_MAX)
            elif D:
                swv = clamp(swv + SWIVEL_STEP, -SWIVEL_MAX, SWIVEL_MAX)
            else:
                swv *= 0.93
            swivel.setPosition(swv)

    push_motors()

    # ─ Telemetry every ~2 s ──────────────────────────────────────────────────
    tick += 1
    if tick >= int(2000 / timestep):
        tick = 0
        roll, pitch, yaw = imu.getRollPitchYaw()
        tilt  = fl_ts.getValue()
        sv    = rs_ts.getValue()
        pt    = pt_ts.getValue()
        spd   = ((lv + rv) * 0.5) * WHEEL_RADIUS
        kstr  = ''.join(sorted(chr(k) for k in keys if 32 <= k <= 126))
        pstr  = f"PROP({pt*57.3:.0f}deg)" if pt > 0.3 else f"ROLL({pt*57.3:.0f}deg)"
        print(
            f"[{mode}] spd={spd:+.2f}m/s | "
            f"arm={tilt*57.3:.0f}deg swivel={sv*57.3:.0f}deg {pstr} | "
            f"roll={roll*57.3:.0f}deg pitch={pitch*57.3:.0f}deg | "
            f"keys=[{kstr}]"
        )
