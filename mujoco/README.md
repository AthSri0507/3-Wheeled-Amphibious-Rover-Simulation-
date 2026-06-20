# Amphibious Triangular Rover — MuJoCo port

A MuJoCo re-implementation of the Webots rover (`../rover_terrain.wbt` +
`../rover_controller.py`). The rover is a 3-wheeled amphibious robot: two front
tilting arms and a rear swivel pod whose wheel yaws 90° to act as a boat
propeller. It drives on **LAND** and floats / propels on **WATER**.

## Files
- `rover.xml` — the MJCF model. Reuses the original binary STL meshes in place
  (via `meshdir=".."`); detailed meshes are visual-only, collision uses simple
  primitives (boxes/cylinders), mirroring the Webots `boundingObject`s.
- `rover_sim.py` — controller (ported 1:1 from the Webots script) + the custom
  water/buoyancy model + the automatic land↔water transition + two entry points.
- `make_terrain.py` — generates `../beach.png`, the heightfield used as the terrain
  (a flat beach that ramps down into a deep water basin). `rover_sim.py` builds it
  automatically on first run if it's missing.

## Install & run
```bash
pip install mujoco            # tested with mujoco 3.9.0, Python 3.10

# interactive viewer (drive it yourself):
python mujoco/rover_sim.py

# scripted land+water demo (opens a window):
python mujoco/rover_sim.py --demo

# same demo with no window, prints telemetry only:
python mujoco/rover_sim.py --demo --headless
```
Run from the project root (the folder that contains both `mujoco/` and the
`*.STL` files), so the mesh paths resolve.

## Controls (interactive)
Keys are **latched** — a press stays in effect until you change it (the passive
viewer only reports key presses, not holds). Click the 3D window first. Driving uses
the **arrow keys** (`W`/`A`/`S`/`D` are reserved by the MuJoCo viewer for rendering).

| Mode  | Keys |
|-------|------|
| LAND  | `↑`/`↓` drive fwd/back · `←`/`→` pivot turn · `Q`/`E` arc turn · `Space` stop |
| WATER | `↑`/`↓` propel fwd/back (rear propeller only) · `←`/`→` steer left/right (rear-pod swivel) · `Space` stop + re-centre |
| any   | `T` force-deploy water · `G` force-retract land · `F` back to automatic |

`↑` drives the **same direction** (−Y, toward the water) on land and in water, so you
can drive straight off the beach without the controls flipping.

### Smooth land↔water transition (automatic)
The terrain is a **beach** that ramps down into a water basin (toward −Y). Just drive
`W` off the beach: as the hull starts to float, the arms spread and the rear wheel
deploys into propeller mode **gradually** (watch `deploy=…%` in the telemetry), and the
front wheels hand off to the propeller only once afloat — so it never stalls at the
water's edge. Drive `S` back up the beach and it auto-retracts to rolling-wheel mode.
`T`/`G` are optional manual overrides; `F` returns to automatic.

In water the front wheels splay up out of the water, so only the **submerged rear
wheel** propels; **A/D** swing the rear pod (the propeller) to steer like an outboard
motor — *latched*, so one press keeps turning until you press the opposite key or
`Space` to centre.

## How the model maps from Webots
- Z-up, metres (Webots mesh `scale 0.001` → MuJoCo mesh asset scale).
- Each Webots `Solid` → `<body>`; `HingeJoint{axis,anchor}` → `<joint pos=anchor>`;
  `Physics{mass,com,inertiaMatrix}` → `<inertial>`.
- 7 actuators: 3 wheel **velocity** servos + 4 tilt/swivel/prop **position** servos,
  matching the modes the original controller used.

## Water model (no native water in MuJoCo)
MuJoCo has no free water surface, so `RoverController.apply_water_forces()` adds
external forces (`data.xfrc_applied`) every step, only for bodies inside the
`WATER_REGION` rectangle and below `WATER_Z`:
- **Buoyancy** `ρ·g·V·(submerged fraction)`, applied at a centre-of-buoyancy a few
  cm **above** the body COM so the hull self-rights (no listing).
- **Hydro drag** (linear + quadratic) opposing velocity, plus rotational drag.
- **Propeller thrust** along the rear pod's forward axis ∝ rear-wheel speed, plus a
  rudder-style yaw torque ∝ swivel angle for steering.

The automatic transition compares the **chassis submerged fraction** against
`SUB_DEPLOY`/`SUB_RETRACT` (with hysteresis) and ramps a `deploy` value 0→1 at
`DEPLOY_RATE`, which smoothly drives the arm/prop joints between their land and water
angles. `WATER_Z=0` is the shoreline; the beach geometry lives in `make_terrain.py`.

All gains are module-level constants at the top of `rover_sim.py`
(`WATER_Z`, `WATER_REGION`, `RHO`, `BUOY`, `DRAG_*`, `THRUST_GAIN`, `STEER_GAIN`,
`COB_LIFT`, `DEPLOY_RATE`, `SUB_DEPLOY`, `SUB_RETRACT`) — tune them to taste.

## Notes / known limitations
- Inertias for the arm/pod/prop bodies are approximated from their collision
  primitives (the Webots model only gave their mass); refine from CAD if needed.
- Wheel joints have `armature` (rotor inertia) added — without it the free-spinning
  wheels make the velocity servo numerically unstable in water.
- Rover collision geoms use `contype=2/conaffinity=1` (ground/rocks `1/2`) so the
  rover never self-collides; only rover-vs-terrain contacts are computed.
- The buoyancy/propeller model is a qualitative force approximation, not CFD.
