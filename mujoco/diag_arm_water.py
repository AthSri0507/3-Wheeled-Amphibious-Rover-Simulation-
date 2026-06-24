"""
diag_arm_water.py  --  Step-1 diagnostics for the water arm_trim residual (no retrain).

On the Exp B checkpoint, in WATER:
  - per-actuator energy breakdown (where the energy goes), PPO vs rule
  - arm sweep: hold arm_trim at fixed values and measure forward progress, stability
    (roll/pitch), energy, CoT  -> tests the DECISION GATE for masking arm in water.
"""
import contextlib, io, os
import numpy as np
from stable_baselines3 import PPO
import rover_sim; rover_sim.QUIET = True
import rover_env
from baseline_rule import RuleController

HERE = os.path.dirname(os.path.abspath(__file__))
ARM_IDX = 2
EPISODES = 18


def actuator_names(m):
    return [m.actuator(i).name for i in range(m.nu)]


def run(env, action_fn, n, arm_override=None):
    """Return dict of means: net_fwd, path, energy, CoT, stability(deg^2), per-actuator energy."""
    m = env.m
    nu = m.nu
    accs = dict(net_fwd=[], path=[], energy=[], cot=[], stab=[])
    act_e = np.zeros(nu)
    n_steps = 0
    for _ in range(n):
        obs, _ = env.reset()
        start = env.d.xpos[env.rover_bid][0:2].copy()
        fwd0 = -env.d.xmat[env.rover_bid].reshape(3, 3)[:, 1][0:2]
        fwd0 = fwd0 / (np.linalg.norm(fwd0) + 1e-9)
        stab = 0.0; k = 0
        while True:
            a = np.array(action_fn(obs, env), np.float32)
            if arm_override is not None:
                a[ARM_IDX] = arm_override
            obs, r, term, trunc, info = env.step(a)
            act_e += np.abs(env.d.actuator_force * env.d.actuator_velocity) * env.dt
            n_steps += 1
            roll, pitch = env._roll_pitch()
            stab += np.degrees(roll) ** 2 + np.degrees(pitch) ** 2
            k += 1
            if term or trunc:
                break
        c = info["episode_components"]
        end = env.d.xpos[env.rover_bid][0:2]
        accs["net_fwd"].append(float((end - start) @ fwd0))
        accs["path"].append(c["distance"]); accs["energy"].append(c["energy"])
        accs["cot"].append(c["CoT"]); accs["stab"].append(stab / max(k, 1))
    out = {k: float(np.mean(v)) for k, v in accs.items()}
    out["act_e"] = act_e / max(n_steps, 1)   # per-actuator energy per step
    return out


def main():
    env = rover_env.RoverEnv(seed=11); env.set_allowed_classes(["water"])
    ppo = PPO.load(os.path.join(HERE, "models", "ppo_rover.zip"), device="cpu")
    rule = RuleController()
    names = actuator_names(env.m)

    ppo_fn = lambda o, e: ppo.predict(o, deterministic=True)[0]
    rule_fn = lambda o, e: rule.act(e.terrain_spec.cls)

    with contextlib.redirect_stdout(io.StringIO()):
        rule_r = run(env, rule_fn, EPISODES)
        ppo_r = run(env, ppo_fn, EPISODES)
        sweep = {v: run(env, ppo_fn, EPISODES, arm_override=v) for v in (-1.0, -0.5, 0.0, 0.5, 1.0)}

    print("=== WATER per-actuator energy/step (J) ===")
    print(f"{'actuator':18s}{'rule':>10s}{'ppo':>10s}")
    for i, nm in enumerate(names):
        print(f"{nm:18s}{rule_r['act_e'][i]:10.4f}{ppo_r['act_e'][i]:10.4f}")
    print(f"{'TOTAL':18s}{rule_r['act_e'].sum():10.4f}{ppo_r['act_e'].sum():10.4f}")

    print("\n=== WATER arm sweep (PPO drive/steer, arm forced) ===")
    print(f"{'arm':>6s}{'net_fwd':>10s}{'path':>9s}{'energy':>9s}{'CoT':>9s}{'stab(deg^2)':>13s}")
    print(f"{'policy':>6s}{ppo_r['net_fwd']:10.3f}{ppo_r['path']:9.3f}{ppo_r['energy']:9.3f}"
          f"{ppo_r['cot']:9.2f}{ppo_r['stab']:13.2f}")
    for v, r in sweep.items():
        print(f"{v:>6.1f}{r['net_fwd']:10.3f}{r['path']:9.3f}{r['energy']:9.3f}{r['cot']:9.2f}{r['stab']:13.2f}")
    print(f"{'rule':>6s}{rule_r['net_fwd']:10.3f}{rule_r['path']:9.3f}{rule_r['energy']:9.3f}"
          f"{rule_r['cot']:9.2f}{rule_r['stab']:13.2f}")


if __name__ == "__main__":
    main()
