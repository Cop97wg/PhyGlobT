"""Re-simulate radar observations for a scene from its ground-truth trajectories.

Takes an existing scene ``.npy`` (whose ``raw_trajectories`` hold the AIS ground
truth) and regenerates both radars' observation tracks under user-controlled
noise / detection / sampling parameters. The segment order and MMSI mapping are
preserved, so track ids and ground-truth pairs stay valid — only the point sets
change. This makes "what-if" evaluation possible: run the same models on the
same ships under a different radar error model.

Noise model (mirrors the original generator, see ``radar_sim.json``):

    σρ = sigma_rho_m · (r / noise_ref_m)^noise_exponent    (capped at noise_cap×)
    σθ = sigma_theta_deg · (r / noise_ref_m)^noise_exponent

Range (radial) and azimuth (tangential) errors are drawn independently per
plot. Each radar samples its own sweep grid (``sweep_interval_s``) with a
random ±``async_jitter_s`` offset; a sweep yields a plot with probability
``detection_rate`` (the original generator's effective detection is
1 − (1 − Pd·(1−drop)) ≈ 0.68). Points beyond the sensor's max range are gated.

Coordinates follow the same convention as ``scenes.load_scene``: observations
and truth are LOCAL degree offsets from the grid corner (absolute WGS84 inputs
are normalized in memory; ``metadata.lat_offset`` = corner). Ground truth and
observations are verified to share one time base (relative seconds).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np

from .scenes import Point, Scene, Track

_M_PER_DEG_LAT = 111320.0


@dataclass
class SimParams:
    sigma_rho_m: float = 30.0        # range noise at noise_ref_m
    sigma_theta_deg: float = 0.5     # azimuth noise at noise_ref_m
    noise_exponent: float = 0.8      # σ ∝ (r / ref)^exponent
    noise_ref_m: float = 10000.0     # reference range
    noise_cap: float = 4.0           # max multiplier of the base σ
    detection_rate: float = 0.68     # P(sweep yields a plot); 1.0 = no misses
    sweep_interval_s: float = 10.0   # antenna period (6 rpm ≙ 10 s)
    async_jitter_s: float = 2.0      # ± random sweep offset per sensor
    sensor_range_a_m: float | None = None  # None → keep metadata max_range
    sensor_range_b_m: float | None = None
    seed: int = 0                    # 0 → random entropy


def parse_sim_params(d: dict | None) -> SimParams:
    d = d or {}
    p = SimParams()

    def _num(key: str, default: float, lo: float, hi: float) -> float:
        v = d.get(key, default)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = default
        return min(max(v, lo), hi)

    p.sigma_rho_m = _num("sigma_rho_m", 30.0, 0.0, 1000.0)
    p.sigma_theta_deg = _num("sigma_theta_deg", 0.5, 0.0, 10.0)
    p.noise_exponent = _num("noise_exponent", 0.8, 0.0, 2.0)
    p.noise_ref_m = _num("noise_ref_m", 10000.0, 100.0, 100000.0)
    p.noise_cap = _num("noise_cap", 4.0, 1.0, 10.0)
    p.detection_rate = _num("detection_rate", 0.68, 0.01, 1.0)
    p.sweep_interval_s = _num("sweep_interval_s", 10.0, 1.0, 120.0)
    p.async_jitter_s = _num("async_jitter_s", 2.0, 0.0, 30.0)

    def _opt_range(key: str) -> float | None:
        v = d.get(key)
        if v in (None, ""):
            return None
        try:
            return min(max(float(v), 500.0), 500000.0)
        except (TypeError, ValueError):
            return None

    p.sensor_range_a_m = _opt_range("sensor_range_a_m")
    p.sensor_range_b_m = _opt_range("sensor_range_b_m")

    try:
        p.seed = int(d.get("seed", 0) or 0)
    except (TypeError, ValueError):
        p.seed = 0
    return p


def params_to_dict(p: SimParams) -> dict:
    return {f.name: getattr(p, f.name) for f in fields(p)}


def simulate_scene(raw: dict, params: SimParams, name: str) -> tuple[Scene, dict]:
    """Build a re-simulated :class:`Scene` from a raw scene ``.npy`` dict."""
    obs = [np.asarray(a, dtype=float) for a in raw.get("observations", [])]
    seg = list(raw.get("segment_info", []))
    truth_raw = raw.get("raw_trajectories", {}) or {}
    meta = dict(raw.get("metadata", {}))

    # ── coordinate normalization (mirrors scenes.load_scene) ──
    good = [a for a in obs if a.ndim == 2 and a.shape[0] > 0 and a.shape[1] >= 6]
    lat_all = np.concatenate([a[:, 0] for a in good]) if good else np.array([])
    absolute = bool(lat_all.size and np.median(np.abs(lat_all)) > 5.0)
    corner_lat = corner_lon = 0.0
    if absolute:
        # Include the truth extents in the corner so observations and truth
        # share ONE local base (truth may reach slightly beyond the obs cloud).
        lon_all = np.concatenate([a[:, 1] for a in good])
        corners_lat, corners_lon = [float(lat_all.min())], [float(lon_all.min())]
        for v in truth_raw.values():
            t = np.asarray(v, dtype=float)
            if t.ndim == 2 and t.shape[0] > 0 and t.shape[1] >= 6:
                corners_lat.append(float(t[:, 0].min()))
                corners_lon.append(float(t[:, 1].min()))
        corner_lat = float(np.floor(min(corners_lat)))
        corner_lon = float(np.floor(min(corners_lon)))
        for a in good:
            a[:, 0] -= corner_lat
            a[:, 1] -= corner_lon
        truth: dict[str, np.ndarray] = {}
        for k, v in truth_raw.items():
            t = np.asarray(v, dtype=float)
            if t.ndim == 2 and t.shape[0] > 0 and t.shape[1] >= 6:
                t = t.copy()
                t[:, 0] -= corner_lat
                t[:, 1] -= corner_lon
            truth[str(k)] = t
        if not meta.get("lat_offset"):
            meta["lat_offset"] = corner_lat
        if not meta.get("lon_offset"):
            meta["lon_offset"] = corner_lon
        for key, corner in (("lat_center", corner_lat), ("lon_center", corner_lon)):
            val = meta.get(key)
            if val is not None and abs(float(val)) > 5.0:
                meta[key] = float(val) - corner
    else:
        truth = {str(k): np.asarray(v, dtype=float) for k, v in truth_raw.items()}

    lat_off = float(meta.get("lat_offset", 0.0))
    lon_off = float(meta.get("lon_offset", 0.0))
    center_lat = lat_off + float(meta.get("lat_center", 0.0))
    cos_lat = max(math.cos(math.radians(center_lat)), 1e-9)
    # Local coords are relative to the grid CORNER (lat_offset), while sensor
    # positions are relative to the scene CENTER (lat_center) — offset accordingly.
    lat_c = float(meta.get("lat_center", 0.0))
    lon_c = float(meta.get("lon_center", 0.0))

    # ── sensor geometry (meters, origin = scene center) ──
    sens_xy: dict[str, tuple[float, float]] = {}
    sens_range: dict[str, float | None] = {}
    for sid, key in ((1, "a"), (2, "b")):
        pos = meta.get(f"sensor_{sid}_position")
        if pos is not None:
            sens_xy[key] = (float(pos[0]), float(pos[1]))
        override = params.sensor_range_a_m if key == "a" else params.sensor_range_b_m
        sens_range[key] = float(override) if override else meta.get(f"sensor_{sid}_max_range")

    rng = np.random.default_rng(params.seed or None)
    dt = max(params.sweep_interval_s, 0.5)

    def sweep_times(src: np.ndarray, t_lo: float, t_hi: float) -> np.ndarray:
        lo = max(t_lo, float(src[0, 5]))
        hi = min(t_hi, float(src[-1, 5]))
        if hi <= lo:
            return np.array([lo if hi <= lo else hi])
        t = np.arange(lo, hi + 0.5 * dt, dt)
        if params.async_jitter_s > 0:
            t = t + float(rng.uniform(-params.async_jitter_s, params.async_jitter_s))
        return np.unique(np.clip(t, lo, hi))

    def apply_noise(lat_loc: float, lon_loc: float, key: str) -> tuple[float, float]:
        """Range/azimuth noise in sensor polar coordinates → local degrees."""
        if key not in sens_xy or (params.sigma_rho_m == 0.0 and params.sigma_theta_deg == 0.0):
            return lat_loc, lon_loc
        sx, sy = sens_xy[key]
        dx = (lon_loc - lon_c) * _M_PER_DEG_LAT * cos_lat - sx
        dy = (lat_loc - lat_c) * _M_PER_DEG_LAT - sy
        r = math.hypot(dx, dy)
        th = math.atan2(dx, dy)  # bearing from north
        scale = min(params.noise_cap, (max(r, 1.0) / params.noise_ref_m) ** params.noise_exponent)
        r2 = r + rng.normal(0.0, params.sigma_rho_m * scale)
        th2 = th + math.radians(params.sigma_theta_deg * scale) * float(rng.normal())
        # Reconstruct in CORNER-relative meters (local-degree base), not the
        # center-relative frame used for the polar noise draw.
        y_corner = sy + r2 * math.cos(th2) + lat_c * _M_PER_DEG_LAT
        x_corner = sx + r2 * math.sin(th2) + lon_c * _M_PER_DEG_LAT * cos_lat
        return y_corner / _M_PER_DEG_LAT, x_corner / (_M_PER_DEG_LAT * cos_lat)

    scene = Scene(name=name, metadata=meta)
    scene.metadata["simulated"] = True
    scene.metadata["sim_params"] = params_to_dict(params)
    counter = {"A": 0, "B": 0}
    n_truth = n_fallback = plots = 0
    base_plots = sum(a.shape[0] for a in good)

    for i, a in enumerate(obs):
        a = np.asarray(a, dtype=float)
        if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] < 6:
            continue  # same skip rule as scenes.load_scene → ids stay aligned
        info = seg[i] if i < len(seg) and isinstance(seg[i], dict) else {}
        sid = int(info.get("sensor_id", -1))
        sensor = "A" if sid == 1 else "B"
        key = "a" if sid == 1 else "b"
        mmsi = str(info.get("original_mmsi", "") or "")

        t_lo, t_hi = float(a[0, 5]), float(a[-1, 5])
        src = truth.get(mmsi)
        if src is not None and src.ndim == 2 and src.shape[0] >= 2:
            n_truth += 1
        else:
            src = a  # fallback: original (already noisy) points as pseudo-truth
            n_fallback += 1

        t = sweep_times(src, t_lo, t_hi)
        keep = rng.random(t.size) < params.detection_rate
        if keep.sum() < 2:
            keep[:2] = True  # every track stays viable (≥2 points)
        t = t[keep]

        lat = np.interp(t, src[:, 5], src[:, 0])
        lon = np.interp(t, src[:, 5], src[:, 1])
        sog = np.interp(t, src[:, 5], src[:, 2])
        cog = np.interp(t, src[:, 5], src[:, 3])
        hdg = np.interp(t, src[:, 5], src[:, 4]) if src.shape[1] > 4 else np.zeros_like(t)

        pts: list[Point] = []
        rows: list[list[float]] = []
        for k in range(t.size):
            y_l, x_l = apply_noise(float(lat[k]), float(lon[k]), key)
            if key in sens_xy and sens_range.get(key):
                sx, sy = sens_xy[key]
                r_m = math.hypot((x_l - lon_c) * _M_PER_DEG_LAT * cos_lat - sx,
                                 (y_l - lat_c) * _M_PER_DEG_LAT - sy)
                if r_m > float(sens_range[key]):
                    continue  # beyond the range gate
            pts.append(Point(t=float(t[k]), lat=lat_off + y_l, lon=lon_off + x_l,
                             sog=float(sog[k]), cog=float(cog[k])))
            rows.append([y_l, x_l, float(sog[k]), float(cog[k]), float(hdg[k]), float(t[k])])
        if len(pts) < 2 and t.size >= 2:
            # Range gate killed almost everything — keep the first two sweeps
            # so the track (and its id / GT pair) remains viable.
            for k in range(min(2, t.size)):
                y_l, x_l = apply_noise(float(lat[k]), float(lon[k]), key)
                pts.append(Point(t=float(t[k]), lat=lat_off + y_l, lon=lon_off + x_l,
                                 sog=float(sog[k]), cog=float(cog[k])))
                rows.append([y_l, x_l, float(sog[k]), float(cog[k]), float(hdg[k]), float(t[k])])

        arr = np.ascontiguousarray(np.asarray(rows, dtype=float)) if rows else None
        tid = counter[sensor]
        counter[sensor] += 1
        scene.tracks.append(Track(id=tid, sensor=sensor, points=pts, raw=arr, mmsi=mmsi))
        plots += len(pts)

    # Ground truth is attached by the caller via attach_ground_truth()
    # (track ids are preserved, so the base scene's pair list stays valid).
    return scene, {
        "n_tracks": len(scene.tracks),
        "tracks_from_truth": n_truth,
        "tracks_fallback": n_fallback,
        "plots": plots,
        "base_plots": base_plots,
        "coordinate_base": "absolute_wgs84_normalized" if absolute else "local",
        "params": params_to_dict(params),
    }


def attach_ground_truth(scene: Scene, base: Scene) -> None:
    """Copy the GT pair list from the base scene (track ids are preserved)."""
    scene.gt = list(base.gt)
