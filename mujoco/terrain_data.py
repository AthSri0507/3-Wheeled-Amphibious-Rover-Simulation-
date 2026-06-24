"""
terrain_data.py  --  generate the labeled terrain dataset for the classifier.

Drives the rover (reusing rover_sim.RoverController) over randomized terrains and
logs the IMU + wheel-encoder streams (the columns in features.RAW_COLS) at ~100 Hz
with realistic sensor noise. Four run types, with >=~30% touching a terrain boundary:

  pure-land   (hard/sand/gravel)  : constant terrain, drive forward on the plateau
  land-trans  (clsA <-> clsB)     : friction/roughness blend mid-run (temporal)
  water       (water)             : spawn in the basin, propel; planar disturbances on
  shoreline   (land -> water)     : drive down the real beach; label blends by `deploy`

Soft per-step labels (4-vector over hard/sand/gravel/water). Splits must be BY RUN.

Usage:  python terrain_data.py --runs 240 --out ../terrain_dataset.npz
"""
import argparse
import os
import sys

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rover_sim
import terrain as terr
import features as F

FS_LOG = 100.0                      # Hz logged
RUN_SECONDS_LAND = 3.0
RUN_SECONDS_CROSS = 7.0

# sensor noise (per-sample) and per-run bias scales -- kept below the terrain vibration
# signature (accel terrain std ~0.07-0.15 m/s^2) so it adds realism without swamping it.
ACC_NOISE, ACC_BIAS = 0.030, 0.020  # m/s^2
GYR_NOISE, GYR_BIAS = 0.004, 0.002  # rad/s
SPD_NOISE = 0.008                   # m/s
WHEEL_QUANT = 0.05                  # rad/s encoder quantization


def _sensor(m):
    idx = {}
    for name in ("imu_acc", "imu_gyro", "imu_quat",
                 "fl_wheel_vel", "fr_wheel_vel", "rear_wheel_vel"):
        sid = m.sensor(name).id
        idx[name] = (int(m.sensor_adr[sid]), int(m.sensor_dim[sid]))
    return idx


def _roll_pitch(q):
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return roll, pitch


def _read(m, d, sidx, bias, rng):
    g = lambda k: d.sensordata[sidx[k][0]: sidx[k][0] + sidx[k][1]]
    acc = g("imu_acc") + bias["acc"] + rng.standard_normal(3) * ACC_NOISE
    gyr = g("imu_gyro") + bias["gyr"] + rng.standard_normal(3) * GYR_NOISE
    roll, pitch = _roll_pitch(g("imu_quat"))
    wl, wr, wre = (float(g("fl_wheel_vel")[0]), float(g("fr_wheel_vel")[0]),
                   float(g("rear_wheel_vel")[0]))
    quant = lambda v: round(v / WHEEL_QUANT) * WHEEL_QUANT
    speed = float(np.linalg.norm(d.qvel[0:2])) + rng.standard_normal() * SPD_NOISE
    return [float(d.time), *acc, *gyr, float(roll), float(pitch),
            quant(wl), quant(wr), quant(wre), speed]


def _spawn(d, m, water):
    mujoco.mj_resetData(m, d)
    if water:
        d.qpos[0:3] = [0.0, -1.9, 0.10]
    else:
        d.qpos[0:3] = [0.0, 0.15, 0.13]
    d.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)


def run_episode(m, d, tm, ctrl, spec, run_type, rng):
    """Simulate one episode; return (raw[T,C], labels[T,4])."""
    water = spec.is_water or run_type == "shoreline"
    _spawn(d, m, water)
    tm.apply(spec, rng)
    # randomized per-run sensor bias
    bias = {"acc": rng.standard_normal(3) * ACC_BIAS, "gyr": rng.standard_normal(3) * GYR_BIAS}

    ctrl.__init__(m, d)                       # reset controller state on this data
    ctrl.cmd = "fwd"
    dur = RUN_SECONDS_CROSS if run_type == "shoreline" else RUN_SECONDS_LAND
    if run_type == "water":
        dur = 4.0

    # temporal land transition schedule
    blend_t0 = rng.uniform(0.8, 1.4)
    blend_dur = 1.0

    raw, labels = [], []
    next_log = 0.0
    dt = m.opt.timestep
    nsteps = int(dur / dt)
    for i in range(nsteps):
        # occasional gentle turn for motion variety (land only)
        if not water and rng.random() < 0.002:
            ctrl.cmd = rng.choice(["fwd", "fwd", "fwd", "left", "right"])

        # land transition: interpolate scalar terrain props through the blend window
        blend = 0.0
        if run_type == "land_trans" and spec.cls2 is not None:
            blend = float(np.clip((d.time - blend_t0) / blend_dur, 0.0, 1.0))
            tm.apply_scalars(terr.lerp_spec(spec, spec.cls2, blend, rng))

        ctrl.step()
        # water disturbance force on the rover (currents/wind/waves)
        if water:
            fxy = tm.water_force(spec, rng)
            d.xfrc_applied[ctrl.rover_bid, 0:2] += fxy
        mujoco.mj_step(m, d)

        if d.time >= next_log:
            next_log += 1.0 / FS_LOG
            raw.append(_read(m, d, SIDX, bias, rng))
            # label
            if run_type == "shoreline":
                w = float(np.clip(ctrl.deploy, 0.0, 1.0))   # land->water by deploy
                lab = (1 - w) * spec.label_vector(0.0) + w * _ONEHOT_WATER
            elif run_type == "land_trans":
                lab = spec.label_vector(blend)
            else:
                lab = spec.label_vector(0.0)
            labels.append(lab)
    return np.array(raw, dtype=np.float32), np.array(labels, dtype=np.float32)


_ONEHOT_WATER = np.eye(terr.N_CLASSES, dtype=np.float32)[terr.CLASS_IDX["water"]]
SIDX = None


def generate(n_runs, out_path, seed=0):
    global SIDX
    rover_sim.ensure_terrain()
    m = mujoco.MjModel.from_xml_path(rover_sim.XML)
    d = mujoco.MjData(m)
    SIDX = _sensor(m)
    tm = terr.TerrainManager(m)
    ctrl = rover_sim.RoverController(m, d)
    rng = np.random.default_rng(seed)

    # run-type plan: ensure >= ~30% boundary (land_trans + shoreline)
    types = (["pure_hard"] * 3 + ["pure_sand"] * 3 + ["pure_gravel"] * 3 +
             ["water"] * 3 + ["land_trans"] * 3 + ["shoreline"] * 2)  # 17 per cycle, ~29% boundary

    raws, labs, run_ids, meta = [], [], [], []
    for r in range(n_runs):
        rtype = types[r % len(types)]
        if rtype.startswith("pure_"):
            cls = rtype.split("_")[1]
            spec = terr.sample_spec(rng, allowed=[cls])
            run_type = "water" if cls == "water" else "pure"
        elif rtype == "water":
            spec = terr.sample_spec(rng, allowed=["water"]); run_type = "water"
        elif rtype == "land_trans":
            spec = terr.sample_spec(rng, allowed=["hard", "sand", "gravel"], transition_prob=1.0)
            run_type = "land_trans"
        else:  # shoreline
            spec = terr.sample_spec(rng, allowed=["hard", "sand", "gravel"])
            run_type = "shoreline"

        raw, lab = run_episode(m, d, tm, ctrl, spec, run_type, rng)
        if len(raw) < 30:
            continue
        raws.append(raw); labs.append(lab)
        run_ids.append(np.full(len(raw), r, dtype=np.int32))
        meta.append((r, spec.cls, spec.cls2 or "", run_type,
                     float(spec.mass_scale), float(spec.battery_scale)))
        if (r + 1) % 20 == 0:
            print(f"  {r+1}/{n_runs} runs ({rtype})")

    raw_all = np.concatenate(raws); lab_all = np.concatenate(labs)
    rid_all = np.concatenate(run_ids)
    meta = np.array(meta, dtype=object)
    np.savez_compressed(out_path, raw=raw_all, labels=lab_all, run_id=rid_all,
                        meta=meta, raw_cols=np.array(F.RAW_COLS), fs=FS_LOG,
                        classes=np.array(terr.CLASSES))
    print(f"wrote {out_path}: {len(raws)} runs, {len(raw_all)} samples, "
          f"{raw_all.shape[1]} cols @ {FS_LOG:.0f} Hz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=240)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "terrain_dataset.npz"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    generate(a.runs, a.out, a.seed)
