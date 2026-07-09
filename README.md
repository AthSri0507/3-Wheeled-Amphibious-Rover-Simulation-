# Amphibious Triangular Rover

A simulation of a three-wheeled amphibious rover. The rover drives on land on three
wheels, and when it enters water its two front arms swing out for stability while the
rear wheel rotates 90° to act as a propeller. The same vehicle is modelled in two
simulators: the original Webots world, and a MuJoCo port with a custom buoyancy model.

## The vehicle

- **Chassis** with an electronics box, a camera, and an IMU.
- **Two front arms**, each with a tilt joint and a driven wheel. On land the wheels
  hang down and roll; in water the arms splay outward.
- **Rear pod** on a swivel joint, carrying a wheel that tilts into a propeller for
  water propulsion. The swivel steers the craft like an outboard motor.

Seven actuated joints in total (three wheel drives, two arm tilts, the rear swivel,
and the propeller tilt), plus position sensors and an IMU.

## Tools used

- **MuJoCo 3.9** (Python bindings) — physics simulation and viewer.
- **Webots R2025a** — the original simulation world (`rover_terrain.wbt`).
- **Python 3.10** with **NumPy** — control logic and the buoyancy model.
- **Pillow** — generates the terrain heightfield (`beach.png`).
- **STL meshes** exported from CAD, shared by both simulators.

## Repository layout

```
.
├── rover_terrain.wbt        # Webots world
├── rover_controller.py      # Webots controller (keyboard driving)
├── *.STL                    # CAD meshes (chassis, arms, wheels, rear assembly)
├── beach.png                # terrain heightfield used by the MuJoCo scene
└── mujoco/
    ├── rover.xml            # MJCF model (geometry, joints, actuators, scene)
    ├── rover_sim.py         # controller + buoyancy model + viewer / demo
    ├── make_terrain.py      # regenerates beach.png
    └── README.md            # details of the MuJoCo model and water model
```

## Requirements

- Python 3.10 or newer
- Install the Python dependencies:

  ```bash
  pip install mujoco numpy pillow
  ```

- Optional: **Webots R2025a** if you want to run the original world.

## Usage

### MuJoCo

Run everything from the project root (the folder containing both `mujoco/` and the
`.STL` files) so the mesh and terrain paths resolve.

```bash
# interactive viewer — drive it yourself
python mujoco/rover_sim.py

# scripted land-to-water demo (opens a window)
python mujoco/rover_sim.py --demo

# same demo with no window, prints telemetry only
python mujoco/rover_sim.py --demo --headless
```

The terrain is a beach that ramps down into a water basin. Drive off the beach and the
rover deploys its arms and propeller automatically as it begins to float, then retracts
again when it climbs back onto land.

#### Controls

Driving uses the **arrow keys** (W/A/S/D are reserved by the MuJoCo viewer). A key press
stays in effect until you change it. Click the 3D window first.

| Mode  | Keys |
|-------|------|
| Land  | `↑`/`↓` forward/back · `←`/`→` pivot turn · `Q`/`E` arc turn · `Space` stop |
| Water | `↑`/`↓` propel · `←`/`→` steer · `Space` stop |
| Any   | `T` force water · `G` force land · `F` automatic |

To inspect the model on its own, the standalone MuJoCo `simulate` viewer can open
`mujoco/rover.xml` directly (drag-and-drop), but it shows the model only — the driving
and buoyancy run through `rover_sim.py`.

### Webots

Open `rover_terrain.wbt` in Webots and run the simulation; `rover_controller.py` reads
the keyboard for driving.

## Model and water notes

MuJoCo has no built-in water surface, so buoyancy, hydrodynamic drag, and propeller
thrust are applied as external forces each step for the parts of the rover below the
waterline. The detailed STL meshes are used for visuals while collision uses simple box
and cylinder primitives. See `mujoco/README.md` for the full description of the model,
the water model, and the tuning parameters.
