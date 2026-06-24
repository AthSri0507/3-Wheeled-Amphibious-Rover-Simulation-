"""
demo_amphibious.py  --  the deployed ML pipeline running LIVE in MuJoCo, navigating CHECKPOINTS.

  Sensors -> Terrain Classifier -> FSM (mode) -> Controller -> PID -> Actuators

The rover visits 3-4 visible checkpoints (it must TURN between them, not go straight), crosses the
shoreline, deploys to water, and reaches the final waypoint in the basin. Checkpoints are drawn as
spheres: RED = pending, YELLOW = current target, GREEN = reached. The propeller now TRAILS (two
wheels lead). RoverController is UNMODIFIED (rover_sim.py stays the separate manual demo).

  python demo_amphibious.py             # live viewer (supervised controller if trained, else rule)
  python demo_amphibious.py --rule      # force the classifier-FSM rule controller
  python demo_amphibious.py --headless  # no window, prints telemetry
"""
import argparse
import os
import time

import numpy as np
import mujoco
import mujoco.viewer

import amphib
import controllers_amphibious as C

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "ctrl_supervised.joblib")


def build(dt, force_rule):
    if not force_rule and os.path.exists(MODEL):
        import joblib
        blob = joblib.load(MODEL)
        print(f"controller: SUPERVISED ({blob.get('family','?')}) -> PID  [the deployed ML controller]")
        return C.make_supervised_controller(blob["model"], dt=dt)
    print("controller: classifier-FSM RULE -> PID")
    return C.make_controller("classifier_fsm_rule", dt=dt)


GREEN = [0.1, 0.9, 0.2, 1.0]; YELLOW = [1.0, 0.85, 0.1, 1.0]; RED = [0.9, 0.2, 0.2, 1.0]


def draw_markers(viewer, world):
    """Draw checkpoints as coloured spheres (red pending / yellow current / green reached).
    Deduplicate repeated positions (e.g. the round-trip visits A twice) keeping the active colour."""
    best = {}
    for i, cp in enumerate(world.checkpoints):
        key = (round(float(cp[0]), 2), round(float(cp[1]), 2))
        prio = 2 if i == world.cp_idx else (1 if i < world.cp_idx else 0)   # current > reached > pending
        color = YELLOW if i == world.cp_idx else (GREEN if i < world.cp_idx else RED)
        if key not in best or prio > best[key][0]:
            best[key] = (prio, cp, color)
    scn = viewer.user_scn; scn.ngeom = 0
    for _, cp, color in best.values():
        if scn.ngeom >= scn.maxgeom:
            break
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.08, 0, 0]), np.array([cp[0], cp[1], 0.14]),
                            np.eye(3).flatten(), np.asarray(color, np.float32))
        scn.ngeom += 1


def telem(w, ctrl, obs):
    p = getattr(ctrl.ms, "probs", np.array([1, 0, 0, 0.0]))
    cls = ["hard", "sand", "gravel", "water"][int(np.argmax(p))]
    return (f"[{obs['mode']:5s}] cp={w.cp_idx+1}/{len(w.checkpoints)} terrain={cls:6s} "
            f"p_water={p[3]:.2f} deploy={obs['deploy']*100:3.0f}% pos=({obs['pos'][0]:+.2f},{obs['pos'][1]:+.2f}) "
            f"dist_cp={obs['dist']:.2f}m")


def main(force_rule, headless, seed, current, oneway):
    w = amphib.AmphibWorld(seed=seed, current=current)
    ctrl = build(w.dt, force_rule)
    cps = amphib.make_checkpoints(w.rng) if oneway else amphib.roundtrip_checkpoints(w.rng)
    obs = w.reset(checkpoints=cps); ctrl.reset()
    kind = "ONE-WAY into water" if oneway else "ROUND TRIP (land->A->water->B->U-turn->back to A)"
    print(f"course: {kind}\ncheckpoints:", [list(np.round(c, 2)) for c in cps], f"(current={current})\n")

    def loop(viewer=None):
        last = time.time()
        while w.t < amphib.MAX_STEPS:
            a = ctrl.act(w, obs); w.step(a)
            rn, done = w.advance_checkpoint()
            obs.update(w.observe())
            if rn and not done:
                print(f"  >>> checkpoint {w.cp_idx} reached, heading to next")
            if viewer is not None:
                draw_markers(viewer, w); viewer.sync()
                if not viewer.is_running():
                    return
            if time.time() - last > 0.5:
                last = time.time(); print("  " + telem(w, ctrl, obs))
            if done:
                print("\n>>> ALL CHECKPOINTS REACHED."); return
        print("\n>>> episode ended (did not finish).")

    if headless:
        loop(None)
    else:
        with mujoco.viewer.launch_passive(w.m, w.d) as viewer:
            loop(viewer)
            while viewer.is_running():
                draw_markers(viewer, w); viewer.sync(); time.sleep(0.05)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", action="store_true", help="force the classifier-FSM rule controller instead of supervised")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--oneway", action="store_true", help="one-way into water instead of the round trip")
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--current", type=float, default=0.05)   # low: the water U-turn return is hard against current
    a = ap.parse_args()
    main(a.rule, a.headless, a.seed, a.current, a.oneway)
