"""Measure per-step progress & energy rates on packed vs loose at several throttles, then
report the throttle that maximizes (PROG_SCALE*prog - W*energy) for a few W -- to pick an
energy weight where loose prefers a LOW throttle and packed prefers a HIGH throttle."""
import numpy as np
import mujoco
import rover_sim; rover_sim.QUIET = True
import terrain as terr
from rover_env import RANGE_PROG_SCALE

rover_sim.ensure_terrain()
m = mujoco.MjModel.from_xml_path(rover_sim.XML); tm = terr.TerrainManager(m)
WHEELS = ("front_left_wheel", "front_right_wheel", "rear_wheel")
N = 1500


def rates(diff, throttle):
    d = mujoco.MjData(m)
    tm.apply(terr.difficulty_spec("sand", diff), np.random.default_rng(0))
    ctrl = rover_sim.RoverController(m, d)
    d.qpos[0:3] = [0, 0.40, 0.13]; d.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d); ctrl.__init__(m, d)
    rover = m.body("rover").id; y0 = float(d.xpos[rover][1]); E = 0.0; dt = m.opt.timestep
    for _ in range(N):
        ctrl.update_transition()
        for w in WHEELS:
            d.ctrl[ctrl.aid[w]] = throttle * rover_sim.DRIVE_SPEED
        mujoco.mj_step(m, d)
        E += float(np.sum(np.abs(d.actuator_force * d.actuator_velocity))) * dt
    dist = y0 - float(d.xpos[rover][1])
    return dist / N, E / N      # per-step progress (m), per-step energy (J)


throttles = [0.6, 0.8, 1.0]
data = {}
for name, diff in (("PACKED", 0.0), ("LOOSE", 1.0)):
    print(f"\n{name}:  throttle  prog/step   energy/step")
    for t in throttles:
        p, e = rates(diff, t)
        data[(name, t)] = (p, e)
        print(f"          {t:5.1f}   {p*1e4:8.3f}e-4   {e*1e3:8.3f}e-3")

print("\nbest throttle under reward = PROG_SCALE*prog - W*energy:")
print(f"{'W':>5s} | {'PACKED':>8s} | {'LOOSE':>8s}")
for W in (0, 1, 2, 3, 5, 8, 12):
    best = {}
    for name in ("PACKED", "LOOSE"):
        scores = {t: RANGE_PROG_SCALE * data[(name, t)][0] - W * data[(name, t)][1] for t in throttles}
        best[name] = max(scores, key=scores.get)
    print(f"{W:>5d} | {best['PACKED']:>8.1f} | {best['LOOSE']:>8.1f}")
