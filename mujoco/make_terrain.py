"""
make_terrain.py  --  generate beach.png heightfield for rover.xml

Profile (varies only along Y; the rover's forward = -Y drives into the water):
  land    (y > SHORE)        : flat, full height  (drive-on surface, z = 0)
  beach   (BASIN < y < SHORE): linear ramp down into the water
  basin   (y < BASIN)        : flat deep floor (deep enough that the rover floats)

The hfield <size> in rover.xml is "radius_x radius_y elevation base"; with
elevation=0.5 and the geom placed at z=-0.5, a normalised height of 1.0 -> top z=0
(land) and 0.2 -> top z=-0.4 (basin floor, ~0.4 m of water).
"""
import os
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

N = 256                  # grid resolution (square)
HALF = 4.0               # terrain half-extent in metres (matches size radius in XML)
SHORE = -0.6             # y where the beach starts
BASIN = -2.4             # y where the flat deep basin starts
# NB: MuJoCo renormalises the PNG to [0,1] by min-max at load, so the absolute values
# below only set the SHAPE (land is always the max -> 1.0). terrain.py rescales the
# loaded field at runtime to give gravel bumps headroom (runtime edits aren't renormalised).
LAND_H = 1.0             # normalised land height
BASIN_H = 0.2            # normalised basin height
ELEVATION_M = 0.5        # hfield <size> elevation (m) per normalised unit (matches rover.xml)


def roughness_field(shape, amp_norm, smooth=2.5, rng=None):
    """Smooth low-pass random bumps in normalised-height units, peak |value| ~= amp_norm.

    terrain.py adds this to the land region of model.hfield_data at each reset to make
    'gravel' (high-frequency surface -> high-frequency IMU signature). amp_norm is in
    the same [0,1] units as the heightfield (multiply a desired metre amplitude by
    1/ELEVATION_M to convert)."""
    rng = rng or np.random.default_rng()
    noise = gaussian_filter(rng.standard_normal(shape), sigma=smooth)
    noise /= (np.abs(noise).max() + 1e-9)
    return noise.astype(np.float32) * float(amp_norm)


def height(y):
    if y >= SHORE:
        return LAND_H
    if y <= BASIN:
        return BASIN_H
    t = (y - SHORE) / (BASIN - SHORE)          # 0 at shore -> 1 at basin
    return LAND_H + t * (BASIN_H - LAND_H)


def main():
    img = np.zeros((N, N), dtype=np.uint8)
    # image row 0 = top = +Y ; row N-1 = bottom = -Y
    for r in range(N):
        y = HALF - 2.0 * HALF * r / (N - 1)     # r=0 -> +HALF, r=N-1 -> -HALF
        img[r, :] = int(round(height(y) * 255))
    # write into the project root (meshdir=".." in rover.xml resolves hfield files there)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "beach.png")
    Image.fromarray(img).save(out)
    print("wrote", out, img.shape)


if __name__ == "__main__":
    main()
