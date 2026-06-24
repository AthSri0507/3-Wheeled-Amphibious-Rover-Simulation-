"""
terrain.py  --  shared terrain helper for the classifier dataset and the RL env.

Defines the four terrain CLASSES and a `TerrainManager` that applies a sampled
`TerrainSpec` to a live MuJoCo model at reset (and cheaply re-applies scalar
properties each step for temporal transitions). All randomization that both the
classifier dataset and PPO must agree on lives here, so the two stay consistent.

Terrains are realized by editing the existing scene at runtime (no scene
duplication):
  - friction          -> model.geom_friction[terrain]
  - gravel roughness  -> bumps added to the land region of model.hfield_data
  - rolling resistance -> wheel-joint dof damping
  - "battery" level   -> actuator gain + forcerange scaling
  - mass              -> body mass/inertia scaling
  - water currents    -> a slowly varying planar force (applied by the caller)

Land transitions are modelled *temporally* (friction/roughness interpolate from one
land class to another mid-run); the land->water "shoreline" transition is the real
spatial beach crossing (handled by terrain_data.py driving down the existing ramp).
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

import make_terrain  # roughness_field, ELEVATION_M, HALF, SHORE

CLASSES = ["hard", "sand", "gravel", "water"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)

# per-class physical property ranges: friction(slide), surface roughness (m),
# wheel rolling-resistance damping.
PARAMS = {
    "hard":   dict(friction=(1.10, 1.40), rough_m=(0.000, 0.003), wheel_damp=(0.001, 0.003)),
    "sand":   dict(friction=(0.55, 0.85), rough_m=(0.003, 0.010), wheel_damp=(0.012, 0.030)),
    "gravel": dict(friction=(0.70, 1.05), rough_m=(0.030, 0.055), wheel_damp=(0.004, 0.012)),
    "water":  dict(friction=(0.60, 1.00), rough_m=(0.000, 0.000), wheel_damp=(0.001, 0.003)),
}

# land classes that can blend into each other (temporal transitions)
TRANSITION_PAIRS = [("hard", "sand"), ("gravel", "sand"), ("gravel", "hard")]

# Hidden within-class "difficulty" (0 = packed/firm, 1 = loose) realized as PASSIVE
# properties only -- lower friction (more emergent slip) + higher wheel rolling resistance.
# The class LABEL stays the same (e.g. "sand"), so the rule/classifier can't see it.
# Slip is NOT applied as a force; it emerges from the physics.
DIFF_FRICTION  = (1.20, 0.35)    # packed -> loose
DIFF_WHEELDAMP = (0.004, 0.130)  # packed -> loose (rolling resistance up ~32x)


def difficulty_spec(cls, d, rng=None, mass_scale=1.0, battery_scale=1.0):
    """A TerrainSpec whose passive properties interpolate packed(0) -> loose(1) difficulty.
    `d` may exceed 1.0 for out-of-distribution evaluation (extrapolated, still passive)."""
    fr = DIFF_FRICTION[0] + (DIFF_FRICTION[1] - DIFF_FRICTION[0]) * d
    wd = DIFF_WHEELDAMP[0] + (DIFF_WHEELDAMP[1] - DIFF_WHEELDAMP[0]) * d
    return TerrainSpec(cls=cls, friction=max(0.15, fr), rough_m=0.004, wheel_damp=max(0.001, wd),
                       mass_scale=mass_scale, battery_scale=battery_scale)

# Runtime land level (< 1.0) so gravel bumps have headroom before the [0,1] hfield clip.
# (MuJoCo renormalises the PNG to land=1.0 at load; we rescale at runtime where it isn't
# renormalised, and shift the terrain geom so land-top stays at z=0.)
LAND_LEVEL = 0.8


@dataclass
class TerrainSpec:
    cls: str                          # primary class
    friction: float
    rough_m: float
    wheel_damp: float
    mass_scale: float
    battery_scale: float
    cls2: Optional[str] = None        # secondary class for a transition (land only)
    water_current: Optional[np.ndarray] = None   # base planar force (N), water only
    current_var: float = 0.0

    @property
    def is_water(self) -> bool:
        return self.cls == "water"

    def label_vector(self, blend: float = 0.0) -> np.ndarray:
        """Soft label: pure class at blend=0, mixture toward cls2 as blend->1."""
        v = np.zeros(N_CLASSES, dtype=np.float32)
        if self.cls2 is None or blend <= 0.0:
            v[CLASS_IDX[self.cls]] = 1.0
        else:
            v[CLASS_IDX[self.cls]] = 1.0 - blend
            v[CLASS_IDX[self.cls2]] += blend
        return v


def _u(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def sample_spec(rng, allowed=None, transition_prob=0.0, water_disturbance=True):
    """Sample a TerrainSpec from `allowed` classes, optionally a land transition."""
    allowed = allowed or CLASSES
    cls = str(rng.choice(allowed))
    p = PARAMS[cls]
    spec = TerrainSpec(
        cls=cls,
        friction=_u(rng, *p["friction"]),
        rough_m=_u(rng, *p["rough_m"]),
        wheel_damp=_u(rng, *p["wheel_damp"]),
        mass_scale=_u(rng, 0.95, 1.10) if cls == "water" else _u(rng, 0.80, 1.25),
        battery_scale=_u(rng, 0.70, 1.00),
    )
    # land transition (not for water; the shoreline crossing is handled spatially)
    if cls != "water" and rng.random() < transition_prob:
        pairs = [q for q in TRANSITION_PAIRS if cls in q]
        if pairs:
            a, b = pairs[int(rng.integers(len(pairs)))]
            spec.cls2 = b if a == cls else a
    if cls == "water" and water_disturbance:
        ang = rng.uniform(0, 2 * np.pi)
        mag = rng.uniform(0.0, 0.6)              # N, gentle current
        spec.water_current = mag * np.array([np.cos(ang), np.sin(ang)], dtype=np.float64)
        spec.current_var = rng.uniform(0.0, 0.15)
    return spec


def lerp_spec(a: TerrainSpec, b_cls: str, t: float, rng) -> TerrainSpec:
    """Interpolate scalar land properties from spec `a`'s class toward `b_cls` by t."""
    pa, pb = PARAMS[a.cls], PARAMS[b_cls]
    mid = lambda k: (np.mean(pa[k]) * (1 - t) + np.mean(pb[k]) * t)
    return TerrainSpec(
        cls=a.cls, cls2=b_cls,
        friction=mid("friction"), rough_m=max(a.rough_m, np.mean(pb["rough_m"])),
        wheel_damp=mid("wheel_damp"), mass_scale=a.mass_scale, battery_scale=a.battery_scale,
    )


class TerrainManager:
    """Snapshots the base model and applies/restores terrain specs at runtime."""

    def __init__(self, model):
        self.m = model
        self.geom_id = model.geom("terrain").id
        hid = model.hfield("terrain").id
        self.hf_adr = int(model.hfield_adr[hid])
        self.nrow = int(model.hfield_nrow[hid])
        self.ncol = int(model.hfield_ncol[hid])
        self.hf_n = self.nrow * self.ncol
        self.rover_bid = model.body("rover").id
        self.wheel_dofs = [int(model.joint(j).dofadr[0])
                           for j in ("front_left_wheel", "front_right_wheel", "rear_wheel")]
        self.elev = float(model.hfield_size[hid][2])

        # Rescale the (PNG-renormalised, land=1.0) field so land sits at LAND_LEVEL, giving
        # gravel bumps headroom; shift the geom so land-top stays at z=0. This becomes the
        # base that every reset restores. (basin stays at 0 -> top z=-LAND_LEVEL*elev.)
        loaded = model.hfield_data[self.hf_adr:self.hf_adr + self.hf_n].copy()
        self.base_hfield = (loaded * LAND_LEVEL).astype(loaded.dtype)
        model.hfield_data[self.hf_adr:self.hf_adr + self.hf_n] = self.base_hfield
        self.base_geom_z = -LAND_LEVEL * self.elev
        model.geom_pos[self.geom_id, 2] = self.base_geom_z

        self.base_friction = model.geom_friction[self.geom_id].copy()
        self.base_damping = model.dof_damping.copy()
        self.base_gain = model.actuator_gainprm.copy()
        self.base_frange = model.actuator_forcerange.copy()
        self.base_mass = model.body_mass.copy()
        self.base_inertia = model.body_inertia.copy()

        # Land = the flat high plateau of the heightfield. Identify it directly from the
        # base height (orientation-independent) instead of assuming the row<->y ordering.
        base_grid = self.base_hfield.reshape(self.nrow, self.ncol)
        rowmean = base_grid.mean(axis=1)
        self.land_rows = rowmean > (rowmean.max() - 0.05)

    # --- scalar properties (cheap; safe to call every step for transitions) ---
    def apply_scalars(self, spec: TerrainSpec):
        m = self.m
        m.geom_friction[self.geom_id, 0] = spec.friction
        for d in self.wheel_dofs:
            m.dof_damping[d] = spec.wheel_damp
        m.actuator_gainprm[:, 0] = self.base_gain[:, 0] * spec.battery_scale
        m.actuator_forcerange[:] = self.base_frange * spec.battery_scale
        m.body_mass[:] = self.base_mass * spec.mass_scale
        m.body_inertia[:] = self.base_inertia * spec.mass_scale

    # --- surface roughness (rebuilds hfield; call once per run, not per step) ---
    def set_surface(self, rough_m: float, rng):
        hf = self.base_hfield.copy()
        if rough_m > 0:
            grid = hf.reshape(self.nrow, self.ncol)
            amp = rough_m / make_terrain.ELEVATION_M
            # sigma ~1.2 grid cells (~4 cm) -> sharper, wheel-scale bumps -> real vibration
            bumps = make_terrain.roughness_field((self.nrow, self.ncol), amp, smooth=1.2, rng=rng)
            grid = grid.copy()
            grid[self.land_rows] = np.clip(grid[self.land_rows] + bumps[self.land_rows], 0.0, 1.0)
            hf = grid.ravel()
        self.m.hfield_data[self.hf_adr:self.hf_adr + self.hf_n] = hf

    def reset_base(self):
        m = self.m
        m.hfield_data[self.hf_adr:self.hf_adr + self.hf_n] = self.base_hfield
        m.geom_pos[self.geom_id, 2] = self.base_geom_z
        m.geom_friction[self.geom_id] = self.base_friction
        m.dof_damping[:] = self.base_damping
        m.actuator_gainprm[:] = self.base_gain
        m.actuator_forcerange[:] = self.base_frange
        m.body_mass[:] = self.base_mass
        m.body_inertia[:] = self.base_inertia

    def apply(self, spec: TerrainSpec, rng):
        """Full apply for a fresh run: surface + scalars."""
        self.reset_base()
        self.set_surface(spec.rough_m, rng)
        self.apply_scalars(spec)

    @staticmethod
    def water_force(spec: TerrainSpec, rng) -> np.ndarray:
        """Per-step planar disturbance force (N) for water episodes; zeros otherwise."""
        if not spec.is_water or spec.water_current is None:
            return np.zeros(2)
        return spec.water_current + rng.standard_normal(2) * spec.current_var
