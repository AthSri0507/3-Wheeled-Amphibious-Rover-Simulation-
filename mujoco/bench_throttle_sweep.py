"""
bench_throttle_sweep.py  --  STEP 0 benchmark-validation gate (NO PPO).

Sweeps a CONSTANT throttle on packed (d=0) vs loose (d=1) soil under BOTH an energy
budget and a time limit (so creeping can't trivially win), measuring whether the rover
reaches the distance goal. The benchmark is valid only if the best throttle DIFFERS
between packed and loose (no single fixed throttle is near-optimal on both) -- otherwise
adaptation isn't required and RL has no opening.

Tune BUDGET / MAX_STEPS / GOAL / difficulty contrast until a clear split appears.
"""
import argparse
import numpy as np
import mujoco

import rover_sim; rover_sim.QUIET = True
import terrain as terr

m = None; tm = None
WHEELS = ("front_left_wheel", "front_right_wheel", "rear_wheel")


def run(difficulty, throttle, budget, max_steps, goal):
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    spec = terr.difficulty_spec("sand", difficulty)
    tm.apply(spec, rng)
    ctrl = rover_sim.RoverController(m, d)
    d.qpos[0:3] = [0.0, 0.40, 0.13]; d.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    ctrl.__init__(m, d)
    rover = m.body("rover").id
    start_y = float(d.xpos[rover][1]); energy = 0.0; dt = m.opt.timestep
    for i in range(max_steps):
        ctrl.update_transition()
        for w in WHEELS:
            d.ctrl[ctrl.aid[w]] = throttle * rover_sim.DRIVE_SPEED
        mujoco.mj_step(m, d)
        energy += float(np.sum(np.abs(d.actuator_force * d.actuator_velocity))) * dt
        dist = start_y - float(d.xpos[rover][1])     # forward = -Y
        if dist >= goal:
            return dict(reached=True, dist=dist, energy=energy, steps=i + 1)
        if energy >= budget:
            return dict(reached=False, dist=dist, energy=energy, steps=i + 1)
    return dict(reached=False, dist=dist, energy=energy, steps=max_steps)


def main(budget, max_steps, goal):
    global m, tm
    rover_sim.ensure_terrain()
    m = mujoco.MjModel.from_xml_path(rover_sim.XML); tm = terr.TerrainManager(m)
    throttles = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print(f"budget={budget}J  time={max_steps} steps ({max_steps*float(m.opt.timestep):.1f}s)  goal={goal}m\n")
    print(f"{'throttle':>9s} | {'PACKED reach/dist/steps':>26s} | {'LOOSE reach/dist/steps':>26s}")
    print("-" * 70)
    best = {0.0: (-1, None), 1.0: (-1, None)}
    for t in throttles:
        cells = {}
        for dff in (0.0, 1.0):
            r = run(dff, t, budget, max_steps, goal)
            cells[dff] = r
            # reached always beats not-reached; among reached fewer steps is better;
            # among failures farther is better.
            score = (1e6 - r["steps"]) if r["reached"] else (r["dist"] * 100)
            if score > best[dff][0]:
                best[dff] = (score, t)
        def fmt(r):
            return f"{'OK ' if r['reached'] else 'no '}{r['dist']:.2f}m/{r['steps']:>4d}"
        print(f"{t:>9.1f} | {fmt(cells[0.0]):>26s} | {fmt(cells[1.0]):>26s}")
    print("\n--- GATE ---")
    print(f"best throttle PACKED = {best[0.0][1]} ;  LOOSE = {best[1.0][1]}")
    split = best[0.0][1] != best[1.0][1]
    print(f"adaptation required (best throttles differ): {'YES -> gate PASSES' if split else 'NO -> redesign'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=8.0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--goal", type=float, default=1.0)
    a = ap.parse_args()
    main(a.budget, a.steps, a.goal)
