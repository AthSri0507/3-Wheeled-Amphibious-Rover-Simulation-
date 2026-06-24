"""
diag_benchmark.py  --  benchmark-validity diagnostics for the max-range task (READ-ONLY; no training).

Separates: (A) razor-edge benchmark, (B) inherently small adaptation opportunity,
(C) SAC learned-but-imperfect, (D) RL genuinely failed. Primary metric = RANGE (net forward
distance); reach% is secondary. All numbers are mean +/- 95% CI over N episodes.

Diagnostics (run one at a time):
  --diag 0   budget sensitivity 12-28 J (ranking stability -> A)
  --diag 1   oracle success-basin (throttle noise + 2-D packed/loose sweep) + failure taxonomy
  --diag 2   oracle* vs best-fixed Adaptation Gain % across budgets (-> B)
  --diag 3   SAC throttle-vs-slip Pearson/Spearman/p + slip-zeroing sensitivity (-> C vs D)
  --diag 4   graded range distributions + per-controller failure taxonomy

Usage:  python diag_benchmark.py --diag 0 --eps 20
"""
import argparse
import contextlib
import io
import math
import os

import numpy as np

import rover_sim; rover_sim.QUIET = True
import rover_env

HERE = os.path.dirname(os.path.abspath(__file__))
OP_BUDGET = 16.0                       # operating budget
BUDGETS = [12, 16, 20, 24, 28]
SLIP_IDX = 13                          # obs slip channel


# ---------------- controllers (all see only obs/state, except the *_true oracles) ----------------
def fixed(t):
    return lambda o, e: np.array([t, 0.0, 0.0], np.float32)

def oracle_true(packed=1.0, loose=0.6, thr=0.5):
    def f(o, e):
        diff = e._range_diff_at(e._range_netfwd())
        return np.array([loose if diff > thr else packed, 0.0, 0.0], np.float32)
    return f

def slip_feedback(k, slip0):
    def f(o, e):
        slip = float(o[SLIP_IDX])
        thr = float(np.clip(1.0 - k * max(0.0, slip - slip0), 0.5, 1.0))
        return np.array([thr, 0.0, 0.0], np.float32)
    return f

def random_adaptive(thr=0.3):
    rng = np.random.default_rng(0)
    def f(o, e):
        slip = float(o[SLIP_IDX])
        t = rng.uniform(0.5, 0.8) if slip > thr else rng.uniform(0.8, 1.0)
        return np.array([t, 0.0, 0.0], np.float32)
    return f

def load_net(path):
    is_sac = "sac" in os.path.basename(path).lower()
    if is_sac:
        from stable_baselines3 import SAC
        net = SAC.load(path, device="cpu")
    else:
        from stable_baselines3 import PPO
        net = PPO.load(path, device="cpu")
    return lambda o, e: net.predict(o, deterministic=True)[0], net


# ---------------- rollout + stats ----------------
def fail_mode(info, terminated, truncated, budget):
    rg = info["range"]
    if rg["reached"]:
        return "success"
    if rg["energy"] >= 0.98 * budget:
        return "depletion"
    if truncated:
        return "stuck" if rg["net_fwd"] < 0.3 else "timeout"
    return "flip_oob"

def run(env, fn, n, ood, budget, collect_pairs=False):
    out = dict(reached=[], rng=[], energy=[], steps=[], fail=[], thr=[], slip=[])
    for _ in range(n):
        obs, _ = env.reset(options={"ood": ood})
        term = trunc = False
        while True:
            a = fn(obs, env)
            if collect_pairs:
                out["thr"].append(float(a[0])); out["slip"].append(float(obs[SLIP_IDX]))
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
        rg = info["range"]
        out["reached"].append(rg["reached"]); out["rng"].append(rg["net_fwd"])
        out["energy"].append(rg["energy"]); out["steps"].append(rg["steps"])
        out["fail"].append(fail_mode(info, term, trunc, budget))
    return out

def mean_ci(a):
    a = np.asarray(a, float); m = a.mean()
    ci = 1.96 * a.std(ddof=1) / math.sqrt(len(a)) if len(a) > 1 else 0.0
    return m, ci

def fail_counts(fails):
    from collections import Counter
    c = Counter(fails); n = len(fails)
    return " ".join(f"{k}:{100*v/n:.0f}%" for k, v in sorted(c.items(), key=lambda x: -x[1]))


def base_controllers(model, include_net=True):
    ctrls = {
        "fixed 0.8":            fixed(0.8),
        "oracle_true(1.0/0.6)": oracle_true(),
        "slip_fb(best)":        None,      # filled by caller with tuned params
        "random_adaptive":      random_adaptive(),
    }
    if include_net and model and os.path.exists(model):
        fn, _ = load_net(model)
        label = "sac" if "sac" in os.path.basename(model).lower() else "ppo"
        ctrls[label] = fn
    return ctrls


def set_budget(b):
    rover_env.RANGE_BUDGET = float(b)


# ---------------- tuned slip-feedback (grid search, keep best by in-dist range) ----------------
def tune_slip_fb(env, eps):
    set_budget(OP_BUDGET)
    best = (-1, None, None)
    print("  tuning slip_feedback (k x slip0) by in-dist range:")
    for k in (0.5, 1.0, 1.5, 2.0):
        for s0 in (0.1, 0.2, 0.3, 0.4):
            with contextlib.redirect_stdout(io.StringIO()):
                r = run(env, slip_feedback(k, s0), eps, False, OP_BUDGET)
            m, _ = mean_ci(r["rng"])
            if m > best[0]:
                best = (m, k, s0)
    print(f"    best: k={best[1]} slip0={best[2]}  range={best[0]:.3f} m")
    return slip_feedback(best[1], best[2]), best[1], best[2]


# ---------------- Diag 0: budget sensitivity ----------------
def diag0(env, eps, model):
    print("=== DIAG 0: budget sensitivity (ranking stability -> razor-edge test A) ===")
    sfb, k, s0 = tune_slip_fb(env, eps)
    ctrls = base_controllers(model, include_net=False)   # net excluded: per-step NN inference too slow for sweep
    ctrls["slip_fb(best)"] = sfb
    print(f"\n{'budget':>7s} | " + " | ".join(f"{n:>16s}" for n in ctrls))
    print("-" * (9 + 19 * len(ctrls)))
    rank_by_budget = {}
    for b in BUDGETS:
        set_budget(b); cells = {}
        for name, fn in ctrls.items():
            with contextlib.redirect_stdout(io.StringIO()):
                r = run(env, fn, eps, False, b)
            m, ci = mean_ci(r["rng"]); reach = 100 * np.mean(r["reached"])
            cells[name] = (m, ci, reach)
        rank_by_budget[b] = sorted(cells, key=lambda n: -cells[n][0])
        row = " | ".join(f"{cells[n][0]:.2f}±{cells[n][1]:.2f}/{cells[n][2]:.0f}%" for n in ctrls)
        print(f"{b:>7.0f} | " + row)
    print("\n  ranking by range (best->worst) per budget:")
    for b in BUDGETS:
        print(f"    {b:>2.0f}J: " + " > ".join(rank_by_budget[b]))
    tops = {rank_by_budget[b][0] for b in BUDGETS}
    print(f"\n  VERDICT: top controller {'STABLE' if len(tops)==1 else 'UNSTABLE -> razor-edge (A)'} "
          f"across budgets (winners: {tops})")


# ---------------- Diag 1: oracle success-basin ----------------
def diag1(env, eps):
    print("=== DIAG 1: oracle success-basin + failure taxonomy (op budget) ===")
    set_budget(OP_BUDGET)
    print("\n (a) throttle NOISE on oracle_true(1.0/0.6):")
    for sig in (0.0, 0.05, 0.1, 0.2):
        def noisy(o, e, s=sig):
            base = oracle_true()(o, e)
            base[0] = float(np.clip(base[0] + np.random.normal(0, s), 0.3, 1.0))
            return base
        with contextlib.redirect_stdout(io.StringIO()):
            r = run(env, noisy, eps, False, OP_BUDGET)
        m, ci = mean_ci(r["rng"])
        print(f"   sigma={sig:.2f}: range {m:.2f}±{ci:.2f} m  reach {100*np.mean(r['reached']):.0f}%")
    print("\n (b) 2-D throttle basin: rows=loose, cols=packed (range m / reach%):")
    packs = [0.85, 0.90, 0.95, 1.00]; looses = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    print("        packed-> " + "  ".join(f"{p:>10.2f}" for p in packs))
    for lo in looses:
        cells = []
        for pk in packs:
            with contextlib.redirect_stdout(io.StringIO()):
                r = run(env, oracle_true(pk, lo), eps, False, OP_BUDGET)
            m, _ = mean_ci(r["rng"]); cells.append(f"{m:.2f}/{100*np.mean(r['reached']):.0f}%")
        print(f"  loose {lo:.2f}: " + "  ".join(f"{c:>10s}" for c in cells))
    print("\n (c) oracle failure taxonomy:")
    with contextlib.redirect_stdout(io.StringIO()):
        r = run(env, oracle_true(), eps, False, OP_BUDGET)
    print("   " + fail_counts(r["fail"]))


# ---------------- Diag 2: oracle* vs best-fixed Adaptation Gain % ----------------
def diag2(env, eps):
    print("=== DIAG 2: oracle* vs best-fixed -> Adaptation Gain % across budgets (-> B) ===")
    # find oracle* (packed x loose) once at op budget
    set_budget(OP_BUDGET); best = (-1, 1.0, 0.6)
    for pk in (0.9, 0.95, 1.0):
        for lo in (0.5, 0.55, 0.6, 0.65, 0.7):
            with contextlib.redirect_stdout(io.StringIO()):
                r = run(env, oracle_true(pk, lo), eps, False, OP_BUDGET)
            m, _ = mean_ci(r["rng"])
            if m > best[0]:
                best = (m, pk, lo)
    print(f"  oracle* = packed {best[1]} / loose {best[2]}")
    fixed_grid = [0.6, 0.7, 0.8, 0.9, 1.0]
    print(f"\n{'budget':>7s} | {'oracle* range':>16s} | {'best-fixed range':>18s} | {'AdaptGain%':>12s} | class")
    print("-" * 75)
    for b in BUDGETS:
        set_budget(b)
        with contextlib.redirect_stdout(io.StringIO()):
            ro = run(env, oracle_true(best[1], best[2]), eps, False, b)
        om, oci = mean_ci(ro["rng"])
        bf = (-1, 0, 0)
        for t in fixed_grid:
            with contextlib.redirect_stdout(io.StringIO()):
                rf = run(env, fixed(t), eps, False, b)
            m, ci = mean_ci(rf["rng"])
            if m > bf[0]:
                bf = (m, ci, t)
        gain = 100 * (om - bf[0]) / bf[0] if bf[0] > 0 else 0
        cls = ("negligible" if gain < 5 else "modest" if gain < 15 else
               "meaningful" if gain < 30 else "strong")
        print(f"{b:>7.0f} | {om:>7.2f}±{oci:.2f} | {bf[0]:>9.2f}±{bf[1]:.2f}(t={bf[2]}) "
              f"| {gain:>10.1f}% | {cls}")


# ---------------- Diag 3: SAC throttle-vs-slip ----------------
def diag3(env, eps, model):
    from scipy.stats import pearsonr, spearmanr
    print("=== DIAG 3: SAC throttle-vs-slip (C vs D) ===")
    set_budget(OP_BUDGET)
    fn, net = load_net(model)
    r = run(env, fn, eps, False, OP_BUDGET, collect_pairs=True)
    thr = np.array(r["thr"]); slip = np.array(r["slip"])
    pr, pp = pearsonr(slip, thr); sr, sp = spearmanr(slip, thr)
    print(f"  (throttle vs measured slip)  Pearson r={pr:+.3f} p={pp:.2e}   Spearman rho={sr:+.3f} p={sp:.2e}")
    sig_neg = (sr < 0 and sp < 0.05)
    print(f"  significantly NEGATIVE (throttle drops as slip rises)? {sig_neg}  "
          f"-> {'learned adaptation (C)' if sig_neg else 'NOT using slip -> (D) if slip_fb works'}")
    # slip-zeroing input sensitivity on the same observations
    obs_list = []
    obs, _ = env.reset(options={"ood": False})
    for _ in range(1500):
        a = fn(obs, env); obs, _, t, tr, _ = env.step(a); obs_list.append(obs.copy())
        if t or tr:
            obs, _ = env.reset(options={"ood": False})
    O = np.array(obs_list); Oz = O.copy(); Oz[:, SLIP_IDX] = 0.0
    a_real = net.predict(O, deterministic=True)[0][:, 0]
    a_zero = net.predict(Oz, deterministic=True)[0][:, 0]
    dmean = float(np.mean(np.abs(a_real - a_zero)))
    print(f"  slip-zeroing sensitivity: mean |d throttle| = {dmean:.4f}  "
          f"(near 0 -> SAC ignores slip input)")


# ---------------- Diag 4: graded range + failure taxonomy ----------------
def diag4(env, eps, model):
    print("=== DIAG 4: graded range distributions + failure taxonomy (op budget, in-dist & OOD) ===")
    sfb, _, _ = tune_slip_fb(env, eps)
    ctrls = base_controllers(model); ctrls["slip_fb(best)"] = sfb
    for ood in (False, True):
        set_budget(OP_BUDGET)
        print(f"\n  {'OOD' if ood else 'IN-DIST'}:")
        print(f"  {'controller':>18s} {'range(m)':>14s} {'reach%':>7s}  failure-modes")
        for name, fn in ctrls.items():
            with contextlib.redirect_stdout(io.StringIO()):
                r = run(env, fn, eps, ood, OP_BUDGET)
            m, ci = mean_ci(r["rng"])
            print(f"  {name:>18s} {m:>7.2f}±{ci:.2f} {100*np.mean(r['reached']):>6.0f}%  {fail_counts(r['fail'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", type=int, required=True)
    ap.add_argument("--eps", type=int, default=20)
    ap.add_argument("--model", default=os.path.join(HERE, "models", "sac_range.zip"))
    a = ap.parse_args()
    env = rover_env.RoverEnv(seed=123, task="range")
    {0: lambda: diag0(env, a.eps, a.model),
     1: lambda: diag1(env, a.eps),
     2: lambda: diag2(env, a.eps),
     3: lambda: diag3(env, a.eps, a.model),
     4: lambda: diag4(env, a.eps, a.model)}[a.diag]()
    env.close()


if __name__ == "__main__":
    main()
