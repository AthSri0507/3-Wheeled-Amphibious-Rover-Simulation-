"""
features.py  --  THE single, transfer-critical feature extractor.

Used identically by the classifier (sim training) and, later, the real robot. The
inputs are only sensors the real rover has: IMU (accelerometer, gyro, orientation ->
roll/pitch) and wheel encoders, plus a body-speed estimate (GPS/encoder-derived) for
the slip ratio. Features are physics-meaningful (vibration energy + spectrum + slip),
so a model trained on them transfers far better than one trained on raw waveforms.

A raw log is an array of shape (T, len(RAW_COLS)); `window_features` turns one fixed
window into a feature vector. Window length is a tuned hyperparameter (see
train_classifier.py): pass `fs` and slice windows of W = round(win_sec * fs) samples.
"""
import numpy as np

# canonical raw-stream column order (terrain_data.py logs exactly this)
RAW_COLS = ["t", "ax", "ay", "az", "gx", "gy", "gz", "roll", "pitch", "wl", "wr", "wrear", "speed"]
COL = {name: i for i, name in enumerate(RAW_COLS)}

WHEEL_RADIUS = 0.031     # m, matches rover_sim
V_MIN = 0.05             # m/s floor for slip denominator (numerical stability near 0)
SLIP_MAX = 3.0           # clip slip to +/- this so startup/stop can't blow up
FS_DEFAULT = 100.0       # Hz

# spectral bands (Hz) for vertical-vibration power, at 100 Hz sampling (Nyquist 50)
BANDS = [(0.5, 5.0), (5.0, 15.0), (15.0, 50.0)]


def slip_series(wheel_speed_mean, body_speed):
    """Per-sample wheel-slip ratio, stable near zero speed and clipped."""
    wheel_lin = np.asarray(wheel_speed_mean) * WHEEL_RADIUS
    v = np.asarray(body_speed)
    denom = np.maximum(np.abs(v), V_MIN)
    return np.clip((wheel_lin - v) / denom, -SLIP_MAX, SLIP_MAX)


def _var(x, prefix):
    """Centered VARIATION stats only (no absolute level) -- for accel/gyro/orientation.

    Deliberately excludes the raw mean: absolute accel (gravity projection / tilt) and
    mean orientation are sim-specific pose cues that don't transfer. Terrain shows up in
    *how much the signal varies* (vibration / wobble), which is physical and transfers."""
    x = np.asarray(x, dtype=np.float64)
    xc = x - x.mean()
    return {
        f"{prefix}_std": xc.std(),
        f"{prefix}_ptp": np.ptp(x),
        f"{prefix}_absmean": np.mean(np.abs(xc)),
    }


def _lvl(x, prefix):
    """Level stats (mean/std/absmax) -- for wheel speed & slip, where level is meaningful."""
    x = np.asarray(x, dtype=np.float64)
    return {
        f"{prefix}_mean": x.mean(),
        f"{prefix}_std": x.std(),
        f"{prefix}_absmax": np.max(np.abs(x)),
    }


def _spectral(x, fs, prefix):
    """Dominant frequency + band powers of the (detrended) signal."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    if n < 4 or not np.any(x):
        out = {f"{prefix}_domfreq": 0.0}
        out.update({f"{prefix}_band{i}": 0.0 for i in range(len(BANDS))})
        return out
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    psd = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    total = psd.sum() + 1e-12
    out = {f"{prefix}_domfreq": float(freqs[np.argmax(psd)])}
    for i, (lo, hi) in enumerate(BANDS):
        m = (freqs >= lo) & (freqs < hi)
        out[f"{prefix}_band{i}"] = float(psd[m].sum() / total)   # fraction of power
    return out


def window_features(win, fs=FS_DEFAULT):
    """Compute the feature dict for one window (T, len(RAW_COLS))."""
    win = np.asarray(win, dtype=np.float64)
    f = {}
    # accelerometer: vibration energy per axis + spectrum of vertical axis (bumpiness)
    for ax in ("ax", "ay", "az"):
        f.update(_var(win[:, COL[ax]], ax))
    f.update(_spectral(win[:, COL["az"]], fs, "az"))
    # gyro: rotational vibration
    for ax in ("gx", "gy", "gz"):
        f.update(_var(win[:, COL[ax]], ax))
    # orientation wobble (variation, not absolute tilt)
    f.update(_var(win[:, COL["roll"]], "roll"))
    f.update(_var(win[:, COL["pitch"]], "pitch"))
    # body speed level (terrain affects achievable speed)
    f.update(_lvl(win[:, COL["speed"]], "speed"))
    # wheel speeds + slip (level + variation)
    wmean = win[:, [COL["wl"], COL["wr"], COL["wrear"]]].mean(axis=1)
    f.update(_lvl(wmean, "wheel"))
    slip = slip_series(wmean, win[:, COL["speed"]])
    f.update(_lvl(slip, "slip"))
    return f


# stable feature order, derived once from a dummy window
FEATURE_NAMES = list(window_features(np.zeros((8, len(RAW_COLS)))).keys())


def features_vector(win, fs=FS_DEFAULT):
    d = window_features(win, fs)
    return np.array([d[k] for k in FEATURE_NAMES], dtype=np.float32)


def windows_from_run(raw, win_samples, hop, fs=FS_DEFAULT):
    """Yield (feature_vector, end_index) for sliding windows over one run's raw log."""
    raw = np.asarray(raw, dtype=np.float64)
    T = len(raw)
    for end in range(win_samples, T + 1, hop):
        yield features_vector(raw[end - win_samples:end], fs), end
