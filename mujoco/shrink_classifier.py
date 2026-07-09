"""
shrink_classifier.py  --  fit the terrain classifier on a microcontroller.

Builds two families of small classifiers and evaluates them under ONE identical protocol:
  A) directly-trained compact models (small RF / shallow XGB / tiny MLP) on the soft labels
  B) knowledge-distilled students (fit to the 25 MB RF teacher's soft outputs)
each with FULL (37) or LITE (33, no-FFT) features.

For every candidate it reports: diagnostics (accuracy, water recall) [NOT deciding], footprint
(size / params / latency vs the MCU budget), and the PRIMARY metric -- held-out MISSION completion
(the amphibious water goal, with that classifier driving the FSM). The winner is the smallest-
footprint candidate whose mission completion matches the full-RF baseline AND meets the MCU budget.
Deploys it to models/terrain_clf.joblib (backs up the full RF as terrain_clf_full.joblib).

  python shrink_classifier.py --mission-seeds 3
"""
import argparse
import os
import time

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import xgboost as xgb

import features as F
from train_classifier import build_windows, FREEZE_WIN

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
BUDGET = dict(flash_kb=1024, params=50000, lat_ms=10.0)
CLASSES = ["hard", "sand", "gravel", "water"]
WATER = 3


class SoftProbaRegressor:
    """Wrap a fitted regressor-pipeline (predicts a 4-vector) as a classifier with predict_proba.
    Used for knowledge distillation: the regressor is fit to the teacher's soft probabilities."""
    def __init__(self, reg_pipe): self.reg = reg_pipe
    def predict_proba(self, X):
        p = np.clip(np.asarray(self.reg.predict(X), float), 0.0, None)
        s = p.sum(1, keepdims=True); s[s == 0] = 1.0
        return p / s
    def predict(self, X): return self.predict_proba(X).argmax(1)


def blob(pipe, feat_idx):
    return dict(pipe=pipe, win_sec=FREEZE_WIN, fs=100.0, classes=CLASSES,
                feature_names=F.FEATURE_NAMES, feat_idx=feat_idx)


def final_model(cand):
    base = cand.reg if isinstance(cand, SoftProbaRegressor) else cand
    return base.steps[-1][1]           # the estimator inside the (Scaler, model) pipeline


def footprint(cand):
    p = os.path.join(MODELS, "_fp.joblib"); joblib.dump(cand, p)
    size_kb = os.path.getsize(p) / 1024; os.remove(p)
    m = final_model(cand)
    if isinstance(m, (MLPClassifier, MLPRegressor)):
        params = int(sum(c.size for c in m.coefs_) + sum(b.size for b in m.intercepts_))
    elif isinstance(m, RandomForestClassifier):
        params = int(sum(t.tree_.node_count for t in m.estimators_))
    else:
        try: params = int(len(m.get_booster().trees_to_dataframe()))
        except Exception: params = 0
    return size_kb, params


def latency(cand, nfeat):
    x = np.zeros((1, nfeat), np.float32)
    t0 = time.perf_counter()
    for _ in range(300): cand.predict_proba(x)
    return (time.perf_counter() - t0) / 300 * 1000


def diagnostics(cand, Xte, yte, pure):
    pred = cand.predict(Xte)
    acc = accuracy_score(yte[pure], pred[pure])
    _, rc, _, _ = precision_recall_fscore_support(yte[pure], pred[pure], labels=[WATER],
                                                  average=None, zero_division=0)
    return acc, float(rc[0])


def mission_completion(cand_blob, seeds):
    """PRIMARY metric: completion of the amphibious water goal with this classifier in the FSM."""
    import amphib, controllers_amphibious as C
    path = os.path.join(MODELS, "_cand.joblib"); joblib.dump(cand_blob, path)
    reached = []
    for s in seeds:
        w = amphib.AmphibWorld(seed=s, current=0.05)
        ctrl = C.make_controller("classifier_fsm_rule", dt=w.dt, clf_path=path)
        reached.append(C.run_episode(w, ctrl)["reached"])
    os.remove(path)
    return 100.0 * np.mean(reached)


def main(mission_seeds):
    data = np.load(os.path.join(os.path.dirname(HERE), "terrain_dataset.npz"), allow_pickle=True)
    X, Ysoft, yhard, is_pure, groups = build_windows(data, FREEZE_WIN)
    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=0).split(X, yhard, groups))
    full = list(range(X.shape[1])); lite = F.LITE_FEATURE_IDX
    seeds = list(range(300, 300 + mission_seeds))
    print(f"windows: {len(X)}  train {len(tr)} / test {len(te)}   full={len(full)} lite={len(lite)} feats\n")

    teacher = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0))
    teacher.fit(X[tr], yhard[tr])
    soft_tr = teacher.predict_proba(X[tr])                 # teacher soft targets for distillation

    def mk(idx, kind):
        Xi = X[tr][:, idx]
        if kind == "rf25":  m = RandomForestClassifier(n_estimators=25, max_depth=8, n_jobs=-1, random_state=0)
        elif kind == "xgb": m = xgb.XGBClassifier(n_estimators=60, max_depth=4, tree_method="hist",
                                                  learning_rate=0.2, num_class=4, objective="multi:softprob", verbosity=0)
        elif kind == "mlp": m = MLPClassifier(hidden_layer_sizes=(16,), activation="tanh", max_iter=500, random_state=0)
        elif kind == "mlp_distill":
            rp = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(16,), activation="tanh",
                                                              max_iter=500, random_state=0)).fit(Xi, soft_tr)
            return SoftProbaRegressor(rp)
        return make_pipeline(StandardScaler(), m).fit(Xi, yhard[tr])

    cands = [
        ("baseline RF300", full, None, "-"),               # = teacher (full RF)
        ("A: RF25 full", full, "rf25", "direct"),
        ("A: RF25 lite", lite, "rf25", "direct"),
        ("A: XGB60 lite", lite, "xgb", "direct"),
        ("A: MLP16 lite", lite, "mlp", "direct"),
        ("B: MLP16 distill full", full, "mlp_distill", "distill"),
        ("B: MLP16 distill lite", lite, "mlp_distill", "distill"),
    ]

    hdr = f"{'candidate':24s}{'acc':>6s}{'water_rec':>10s}{'size(KB)':>10s}{'params':>9s}{'lat(ms)':>9s}{'MCU':>5s}{'MISSION%':>10s}"
    print(hdr); print("-" * len(hdr))
    rows = []
    for label, idx, kind, fam in cands:
        cand = teacher if kind is None else mk(idx, kind)
        cb = blob(cand, None if idx == full else idx)
        acc, wrec = diagnostics(cand, X[te][:, idx], yhard[te], is_pure[te])
        size_kb, params = footprint(cand); lat = latency(cand, len(idx))
        miss = mission_completion(cb, seeds)
        ok = size_kb < BUDGET["flash_kb"] and params < BUDGET["params"] and lat < BUDGET["lat_ms"]
        rows.append((label, cb, acc, wrec, size_kb, params, lat, ok, miss))
        print(f"{label:24s}{acc:6.2f}{wrec*100:9.0f}%{size_kb:10.1f}{params:9d}{lat:9.2f}{'OK' if ok else 'NO':>5s}{miss:9.0f}%")

    base_miss = rows[0][8]
    ok_cands = [r for r in rows[1:] if r[7] and r[8] >= base_miss - 5]
    ok_cands.sort(key=lambda r: r[5])                      # fewest params
    print("\n--- SELECTION (mission + footprint; accuracy is NOT the tiebreaker) ---")
    print(f"baseline RF300 mission = {base_miss:.0f}%")
    if not ok_cands:
        print("no candidate matched baseline mission within budget; keeping full RF."); return
    win = ok_cands[0]
    print(f"WINNER: {win[0]}  (mission {win[8]:.0f}%, {win[4]:.1f} KB, {win[5]} params, MCU-OK)")
    dep = os.path.join(MODELS, "terrain_clf.joblib")
    bak = os.path.join(MODELS, "terrain_clf_full.joblib")
    if os.path.exists(dep) and not os.path.exists(bak):
        joblib.dump(joblib.load(dep), bak)
    joblib.dump(win[1], dep)
    print(f"deployed -> {dep}  (full RF backed up to terrain_clf_full.joblib)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission-seeds", type=int, default=3)
    a = ap.parse_args()
    main(a.mission_seeds)
