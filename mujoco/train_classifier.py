"""
train_classifier.py  --  train + validate the terrain classifier.

- Windows each run with `features.py` (the SAME extractor the robot will use).
- Sweeps window length (0.25 / 0.5 / 1.0 s) and compares tree ensembles
  (RandomForest, HistGradientBoosting, XGBoost).
- Splits BY RUN (GroupShuffleSplit) so windows never leak across train/test.
- Pure terrain regions  -> accuracy / precision / recall / F1 / confusion matrix.
- Transition regions     -> log-loss, Brier score, KL divergence vs the soft label.
- Permutation importance (sklearn) [+ optional SHAP] to confirm vibration/slip dominate.
- Saves the best (window, model) + scaler to models/terrain_clf.joblib.

Usage:  python train_classifier.py --data ../terrain_dataset.npz
"""
import argparse
import os

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, log_loss)
import xgboost as xgb

import features as F
import terrain as terr

HERE = os.path.dirname(os.path.abspath(__file__))
PURE_THRESH = 0.9          # window is "pure" if max soft-label >= this
WINDOWS_SEC = [0.25, 0.5, 1.0]
FREEZE_WIN = 0.5           # deployed/frozen window length (user decision)
FREEZE_MODEL = "RandomForest"


def feature_families():
    """Split FEATURE_NAMES into vibration vs slip families (for the ablation)."""
    vib_pref = ("ax", "ay", "az", "gx", "gy", "gz", "roll", "pitch")
    slip_pref = ("speed", "wheel", "slip")
    vib = [i for i, n in enumerate(F.FEATURE_NAMES) if n.split("_")[0] in vib_pref]
    slip = [i for i, n in enumerate(F.FEATURE_NAMES) if n.split("_")[0] in slip_pref]
    return {"vibration": vib, "slip": slip, "vibration+slip": vib + slip}


def build_windows(data, win_sec, hop_sec=0.1):
    fs = float(data["fs"])
    raw, lab, rid = data["raw"], data["labels"], data["run_id"]
    win = int(round(win_sec * fs)); hop = max(1, int(round(hop_sec * fs)))
    X, Ysoft, groups = [], [], []
    for r in np.unique(rid):
        m = rid == r
        rr, ll = raw[m], lab[m]
        for end in range(win, len(rr) + 1, hop):
            X.append(F.features_vector(rr[end - win:end], fs))
            Ysoft.append(ll[end - win:end].mean(axis=0))   # avg soft label over window
            groups.append(int(r))
    X = np.array(X, np.float32); Ysoft = np.array(Ysoft, np.float32)
    groups = np.array(groups)
    yhard = Ysoft.argmax(1)
    is_pure = Ysoft.max(1) >= PURE_THRESH
    return X, Ysoft, yhard, is_pure, groups


def _make_model(name):
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)
    if name == "HistGB":
        return HistGradientBoostingClassifier(max_iter=300, random_state=0)
    if name == "XGBoost":
        return xgb.XGBClassifier(n_estimators=300, max_depth=6, tree_method="hist",
                                 learning_rate=0.1, n_jobs=-1, random_state=0,
                                 num_class=terr.N_CLASSES, objective="multi:softprob")
    raise ValueError(name)


def _transition_metrics(y_soft, proba):
    """Probability-vector quality on mixed-terrain windows."""
    eps = 1e-7
    p = np.clip(proba, eps, 1.0); y = np.clip(y_soft, eps, 1.0)
    brier = float(np.mean(np.sum((proba - y_soft) ** 2, axis=1)))
    ce = float(np.mean(-np.sum(y_soft * np.log(p), axis=1)))
    kl = float(np.mean(np.sum(y_soft * (np.log(y) - np.log(p)), axis=1)))
    return dict(brier=brier, cross_entropy=ce, kl=kl)


def evaluate(data, win_sec, models, seed=0):
    X, Ysoft, yhard, is_pure, groups = build_windows(data, win_sec)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr, te = next(gss.split(X, yhard, groups))
    out = {}
    for name in models:
        pipe = make_pipeline(StandardScaler(), _make_model(name))
        pipe.fit(X[tr], yhard[tr])
        proba = pipe.predict_proba(X[te])
        pred = proba.argmax(1)
        # pure-region metrics
        pmask = is_pure[te]
        acc = accuracy_score(yhard[te][pmask], pred[pmask])
        pr, rc, f1, _ = precision_recall_fscore_support(
            yhard[te][pmask], pred[pmask], average="macro", zero_division=0)
        # transition-region metrics
        tmask = ~pmask
        trans = _transition_metrics(Ysoft[te][tmask], proba[tmask]) if tmask.sum() else {}
        out[name] = dict(acc=acc, prec=pr, rec=rc, f1=f1,
                         n_pure=int(pmask.sum()), n_trans=int(tmask.sum()),
                         trans=trans, pipe=pipe, split=(tr, te),
                         X=X, yhard=yhard, is_pure=is_pure)
    return out


def main(data_path, do_shap=False):
    data = np.load(data_path, allow_pickle=True)
    classes = [str(c) for c in data["classes"]]
    print(f"dataset: {len(np.unique(data['run_id']))} runs, {len(data['raw'])} samples\n")

    # ---- window x model sweep (informational documentation only; no save) ----
    for win_sec in WINDOWS_SEC:
        res = evaluate(data, win_sec, ["RandomForest", "HistGB", "XGBoost"])
        print(f"=== window {win_sec:.2f}s ({int(win_sec*float(data['fs']))} samples) ===")
        for name, m in res.items():
            t = m["trans"]
            tstr = (f"trans: logloss/CE={t['cross_entropy']:.3f} Brier={t['brier']:.3f} "
                    f"KL={t['kl']:.3f}") if t else "trans: n/a"
            print(f"  {name:13s} pure acc={m['acc']:.3f} F1={m['f1']:.3f}  | {tstr}")
        print()

    # ---- FROZEN model: 0.5 s RandomForest (user decision) ----
    print(f">>> FROZEN model: {FREEZE_MODEL} @ {FREEZE_WIN:.2f}s\n")
    X, Ysoft, yhard, is_pure, groups = build_windows(data, FREEZE_WIN)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
                  .split(X, yhard, groups))

    # feature-family ablation (documentation): vibration-only / slip-only / both
    print("feature-family ablation (0.5 s RF, pure-window metrics):")
    for fam, idx in feature_families().items():
        pipe = make_pipeline(StandardScaler(), _make_model(FREEZE_MODEL))
        pipe.fit(X[tr][:, idx], yhard[tr])
        pred = pipe.predict(X[te][:, idx]); pm = is_pure[te]
        acc = accuracy_score(yhard[te][pm], pred[pm])
        _, _, f1, _ = precision_recall_fscore_support(
            yhard[te][pm], pred[pm], average="macro", zero_division=0)
        print(f"  {fam:16s} ({len(idx):2d} feats)  acc={acc:.3f}  F1={f1:.3f}")

    # frozen model uses ALL features
    pipe = make_pipeline(StandardScaler(), _make_model(FREEZE_MODEL)).fit(X[tr], yhard[tr])
    pm = is_pure[te]; pred = pipe.predict(X[te])
    acc = accuracy_score(yhard[te][pm], pred[pm])
    _, _, f1, _ = precision_recall_fscore_support(yhard[te][pm], pred[pm],
                                                  average="macro", zero_division=0)
    print(f"\nfrozen full-feature model: pure acc={acc:.3f} F1={f1:.3f}")
    cm = confusion_matrix(yhard[te][pm], pred[pm], labels=range(len(classes)))
    print("confusion matrix (pure windows, rows=true):")
    print("         " + " ".join(f"{c[:6]:>7s}" for c in classes))
    for i, c in enumerate(classes):
        print(f"  {c:7s} " + " ".join(f"{v:7d}" for v in cm[i]))

    print("\ntop features (permutation importance):")
    pi = permutation_importance(pipe, X[te], yhard[te], n_repeats=5, random_state=0, n_jobs=-1)
    for idx in np.argsort(pi.importances_mean)[::-1][:12]:
        print(f"  {F.FEATURE_NAMES[idx]:16s} {pi.importances_mean[idx]:.4f}")

    if do_shap:
        try:
            import shap  # noqa
            print("\n[shap] available (text-mode summary omitted)")
        except Exception as e:
            print(f"\n[shap] skipped: {e}")

    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    out = os.path.join(HERE, "models", "terrain_clf.joblib")
    joblib.dump(dict(pipe=pipe, win_sec=FREEZE_WIN, fs=float(data["fs"]),
                     classes=classes, feature_names=F.FEATURE_NAMES), out)
    print(f"\nsaved FROZEN model -> {out}  (win={FREEZE_WIN}s, model={FREEZE_MODEL})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(HERE), "terrain_dataset.npz"))
    ap.add_argument("--shap", action="store_true")
    a = ap.parse_args()
    main(a.data, a.shap)
