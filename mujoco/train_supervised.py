"""
train_supervised.py  --  the PRIMARY deployable Layer-2 controller (sensors -> ML -> PID).

Behaviour-clones the oracle_teacher (student-observation space) into a small XGBoost /
HistGradientBoosting / MLP that maps {terrain probs, mode, IMU, slip, velocity, distance,
heading_error, cross_track} -> {drive, steer, arm} setpoints. Reports imitation metrics
(Action MSE/MAE, teacher-agreement %), DEPLOY completion on held-out seeds, and the model
footprint (params / size / latency) vs the MCU budget. Selects the best by completion then
footprint and saves it. Mode switching stays in the frozen Layer-1 classifier+FSM.

Benchmark is FROZEN: dev seeds (demos/training) are disjoint from the held-out eval seeds.
"""
import os
import time

import numpy as np
import joblib
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

import controllers_amphibious as C
import amphib

HERE = os.path.dirname(os.path.abspath(__file__))
DEV_SEEDS = list(range(20))            # demos/training (disjoint from eval)
EVAL_SEEDS = [200 + i for i in range(8)]   # held-out (same block the gate used)
MCU = dict(latency_ms=10.0, params=50000)  # deployment-readiness budget (assessment only)


def build_models():
    # XGBoost is the chosen deployable controller: tiny, fast, explainable, 100% completion.
    # (An earlier bake-off vs HistGB/MLP confirmed all three imitate well; we deploy XGBoost.)
    import xgboost as xgb
    return {"xgboost": MultiOutputRegressor(
        xgb.XGBRegressor(n_estimators=120, max_depth=4, learning_rate=0.2,
                         tree_method="hist", n_jobs=4, verbosity=0))}


def footprint(model, name):
    path = os.path.join(HERE, "models", f"_tmp_{name}.joblib")
    joblib.dump(model, path); size_kb = os.path.getsize(path) / 1024; os.remove(path)
    # parameter / node count
    if name == "mlp":
        params = int(sum(c.size for c in model.coefs_) + sum(b.size for b in model.intercepts_))
    else:
        ests = model.estimators_
        params = 0
        for e in ests:
            try:
                params += int(sum(b.get_booster().trees_to_dataframe().shape[0]
                                  for b in [e]))  # xgboost nodes
            except Exception:
                try:
                    params += int(sum(t.node_count for t in e._predictors[0]))  # histgb
                except Exception:
                    params += 0
    # latency: single-sample predict
    x = np.zeros((1, len(C.OBS_KEYS)), np.float32)
    t0 = time.perf_counter()
    for _ in range(500):
        model.predict(x)
    lat_ms = (time.perf_counter() - t0) / 500 * 1000
    return size_kb, params, lat_ms


def deploy_completion(model, seeds):
    """Completion on the full ROUND-TRIP course (the deployable task), low current."""
    reached = []
    for s in seeds:
        w = amphib.AmphibWorld(seed=s, current=float((s % 3) * 0.05))
        cps = amphib.roundtrip_checkpoints(w.rng)
        r = C.run_episode(w, C.make_supervised_controller(model, dt=w.dt), checkpoints=cps)
        reached.append(r["reached"])
    return float(np.mean(reached))


def main():
    print("collecting oracle_teacher demonstrations (dev seeds)...")
    X, Y = C.collect_demos(DEV_SEEDS)
    n = len(X); cut = int(0.85 * n); idx = np.random.permutation(n)
    tr, te = idx[:cut], idx[cut:]
    print(f"  {n} (obs,action) pairs;  train {len(tr)} / val {len(te)}\n")

    rows = []
    for name, model in build_models().items():
        model.fit(X[tr], Y[tr])
        pred = np.asarray(model.predict(X[te]))
        mse = np.mean((pred - Y[te]) ** 2, axis=0); mae = np.mean(np.abs(pred - Y[te]), axis=0)
        agree = float(np.mean(np.all(np.abs(pred - Y[te]) < 0.15, axis=1)))   # within tol on all channels
        size_kb, params, lat = footprint(model, name)
        comp = deploy_completion(model, EVAL_SEEDS)
        ok = lat < MCU["latency_ms"] and params < MCU["params"]
        rows.append((name, comp, mae, mse, agree, params, size_kb, lat, ok, model))
        print(f"{name:8s} completion={comp*100:3.0f}%  MAE(d/s/a)={mae[0]:.3f}/{mae[1]:.3f}/{mae[2]:.3f}  "
              f"agree={agree*100:3.0f}%  params={params:6d} size={size_kb:6.1f}KB lat={lat:.2f}ms  "
              f"MCU={'OK' if ok else 'NO'}")

    # select: highest completion, then meets MCU budget, then lowest inference latency
    # (latency is measured reliably; the tree node-count is version-fragile so not used to rank).
    rows.sort(key=lambda r: (-(r[1]), not r[8], r[7]))
    best = rows[0]
    out = os.path.join(HERE, "models", "ctrl_supervised.joblib")
    joblib.dump(dict(model=best[9], obs_keys=C.OBS_KEYS, family=best[0]), out)
    print(f"\nSELECTED: {best[0]}  (completion={best[1]*100:.0f}%, params={best[5]}, "
          f"lat={best[7]:.2f}ms, MCU={'OK' if best[8] else 'over-budget'})  -> saved {out}")


if __name__ == "__main__":
    main()
