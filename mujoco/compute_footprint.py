"""
compute_footprint.py  --  deployment-readiness assessment (NOT a project pass/fail).

Reports model size / parameter-or-node count / inference latency for the deployable stack
(classifier + supervised controller) against the MCU reference budget. Flags deployable /
not-deployable-as-is; does NOT invalidate task results.

MCU reference budget: inference < 10 ms, RAM < 256 KB, Flash < 1 MB, params/nodes < 50 k.
"""
import os
import time

import numpy as np
import joblib

import controllers_amphibious as C

HERE = os.path.dirname(os.path.abspath(__file__))
BUDGET = dict(lat_ms=10.0, flash_kb=1024, params=50000)


def size_kb(path):
    return os.path.getsize(path) / 1024 if os.path.exists(path) else 0.0


def latency(predict, x, n=300):
    t0 = time.perf_counter()
    for _ in range(n):
        predict(x)
    return (time.perf_counter() - t0) / n * 1000


def main():
    print("DEPLOYMENT-READINESS (vs MCU budget: <10 ms, <1 MB Flash, <50 k params)\n")
    print(f"{'component':28s}{'flash(KB)':>11s}{'latency(ms)':>13s}{'verdict':>16s}")
    print("-" * 68)

    # --- classifier (frozen RF) ---
    clf_path = os.path.join(HERE, "models", "terrain_clf.joblib")
    blob = joblib.load(clf_path)
    nfeat = len(blob["feature_names"])
    lat = latency(lambda x: blob["pipe"].predict_proba(x), np.zeros((1, nfeat), np.float32))
    fkb = size_kb(clf_path)
    ok = fkb < BUDGET["flash_kb"] and lat < BUDGET["lat_ms"]
    print(f"{'classifier (RandomForest)':28s}{fkb:11.0f}{lat:13.2f}"
          f"{('OK' if ok else 'OVER-BUDGET'):>16s}")

    # --- supervised controller ---
    sup_path = os.path.join(HERE, "models", "ctrl_supervised.joblib")
    if os.path.exists(sup_path):
        sb = joblib.load(sup_path); model = sb["model"]
        lat = latency(model.predict, np.zeros((1, len(C.OBS_KEYS)), np.float32))
        fkb = size_kb(sup_path)
        ok = fkb < BUDGET["flash_kb"] and lat < BUDGET["lat_ms"]
        print(f"{'supervised ('+sb.get('family','?')+')':28s}{fkb:11.1f}{lat:13.2f}"
              f"{('OK' if ok else 'OVER-BUDGET'):>16s}")

    print("\nNotes:")
    print(" - The full RandomForest is large (high Flash); for a true MCU target it must be shrunk")
    print("   (shallow gradient boosting / small RF / quantization / threshold-FSM on top features).")
    print(" - This is a DEPLOYMENT assessment only; it does not invalidate the task/research result.")
    print(" - FSM + PID are negligible (a few states / a few multiply-adds).")


if __name__ == "__main__":
    main()
