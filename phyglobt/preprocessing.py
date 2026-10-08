"""Trajectory preprocessing for PhyGlobT.

Ported from the paper implementation (``Phy_new/preprocessing.py``).
Supports the two on-disk trajectory formats used by the dataset pipeline:

1. radar observations   ``[lat, lon, sog, cog, heading, timestamp]`` (6 cols)
2. preprocessed format  ``[time, lat, lon, sog, cog]``               (5 cols)

``preprocess_trajectory_unified`` returns a fixed-length ``[time_steps, 5]``
feature matrix ``[time_rel, lat_norm, lon_norm, sog_norm, cog_norm]`` and,
optionally, the original (absolute/relative) timestamps — used by the
physics residual encoder to compute dt between consecutive samples.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")


def compute_relative_time_encoding(time_delta, hidden_dim=16):
    """Sinusoidal relative-time encoding (used for short auxiliary signals)."""
    time_delta = np.clip(time_delta / 3600.0, -1, 1)
    position = np.arange(hidden_dim // 2)
    encoding = np.zeros(hidden_dim)
    encoding[0::2] = np.sin(time_delta / np.power(10000, 2 * position / hidden_dim))
    encoding[1::2] = np.cos(time_delta / np.power(10000, 2 * position / hidden_dim))
    return encoding


def compute_distance_encoding(distance, hidden_dim=16):
    """Log-scaled distance encoding (used for short auxiliary signals)."""
    distance = np.log1p(np.clip(distance, 0, 1000))
    position = np.arange(hidden_dim // 2)
    encoding = np.zeros(hidden_dim)
    encoding[0::2] = np.sin(distance / np.power(100, 2 * position / hidden_dim))
    encoding[1::2] = np.cos(distance / np.power(100, 2 * position / hidden_dim))
    return encoding


def preprocess_trajectory_unified(trajectory, time_steps=32, use_interpolation=False, return_original_times=False):
    """
    Unified trajectory preprocessing function - end-to-end version.

    Parameters
    ----------
    trajectory : np.ndarray
        Raw trajectory data, either (T, 6) radar observation rows
        ``[lat, lon, sog, cog, heading, timestamp]`` or (T, 5)
        ``[time, lat, lon, sog, cog]``.
    time_steps : int
        Target number of time steps.
    use_interpolation : bool
        Interpolate short trajectories instead of zero-padding.
    return_original_times : bool
        Also return the original timestamps (for dt computation).
    """
    # Convert to numpy array
    if isinstance(trajectory, list):
        trajectory = np.array(trajectory)

    # Ensure trajectory has correct feature dimensions (at least 5 features)
    if trajectory.shape[1] < 5:
        padded = np.zeros((trajectory.shape[0], 5))
        padded[:, : trajectory.shape[1]] = trajectory
        trajectory = padded
    elif trajectory.shape[1] >= 6:
        # Radar observation format: [lat, lon, sog, cog, heading, timestamp]
        original_time_feature = trajectory[:, -1].copy() if return_original_times else None
        timestamp = trajectory[:, -1].copy()
        lat = trajectory[:, 0].copy()
        lon = trajectory[:, 1].copy()
        sog = trajectory[:, 2].copy()
        cog = trajectory[:, 3].copy()
        trajectory = np.column_stack([timestamp, lat, lon, sog, cog])
    else:
        if return_original_times:
            original_time_feature = trajectory[:, 0].copy() if trajectory.shape[1] > 0 else None
        else:
            original_time_feature = None

    # Separate features
    lat = trajectory[:, 1].copy()  # Latitude
    lon = trajectory[:, 2].copy()  # Longitude
    sog = trajectory[:, 3].copy()  # Speed over ground
    cog = trajectory[:, 4].copy()  # Course over ground
    time_feature = trajectory[:, 0].copy()  # Unix timestamp

    # 1. Time feature: convert to relative time (seconds) and normalize
    if len(time_feature) > 1:
        time_rel = time_feature - time_feature[0]
        if time_rel[-1] > 0:
            time_rel = time_rel / time_rel[-1]
        else:
            time_rel = np.zeros_like(time_rel)
    else:
        time_rel = np.zeros_like(time_feature)

    # 2. Latitude and longitude: map to [-1, 1] via trigonometric functions
    lat_norm = np.sin(np.radians(lat))
    lon_norm = np.sin(np.radians(lon))

    # 3. SOG: normalize to [0, 1], assuming maximum speed of 30 knots
    sog_norm = np.clip(sog / 30.0, 0, 1)

    # 4. COG: map to [-1, 1] via trigonometric functions
    cog_norm = np.sin(np.radians(cog))

    # Combine processed features in order: time, lat, lon, sog, cog
    processed_trajectory = np.column_stack([time_rel, lat_norm, lon_norm, sog_norm, cog_norm])

    # Ensure trajectory length is time_steps
    if len(processed_trajectory) > time_steps:
        processed_trajectory = processed_trajectory[:time_steps]
        if return_original_times and original_time_feature is not None:
            original_time_feature = original_time_feature[:time_steps]
    elif len(processed_trajectory) < time_steps:
        if use_interpolation:
            old_indices = np.arange(len(processed_trajectory))
            new_indices = np.linspace(0, len(processed_trajectory) - 1, time_steps)
            interpolator = interp1d(old_indices, processed_trajectory, axis=0, kind="linear")
            processed_trajectory = interpolator(new_indices)
            if return_original_times and original_time_feature is not None:
                time_interpolator = interp1d(old_indices, original_time_feature, kind="linear")
                original_time_feature = time_interpolator(new_indices)
        else:
            pad_length = time_steps - len(processed_trajectory)
            padding = np.zeros((pad_length, processed_trajectory.shape[1]))
            processed_trajectory = np.vstack([processed_trajectory, padding])
            if return_original_times and original_time_feature is not None:
                time_padding = np.full(pad_length, original_time_feature[-1])
                original_time_feature = np.concatenate([original_time_feature, time_padding])

    if return_original_times:
        return processed_trajectory, original_time_feature
    return processed_trajectory


def latlon_to_meters(lat, lon, ref_lat, ref_lon):
    """
    Convert lat/lon (degrees) to local Cartesian offsets (meters) w.r.t. a
    reference point, using a spherical approximation.
    """
    R = 6371000.0  # Earth radius (m)
    dlat = np.radians(lat - ref_lat)
    dlon = np.radians(lon - ref_lon)
    y = dlat * R          # north displacement
    x = dlon * R * np.cos(np.radians(ref_lat))  # east displacement
    return x, y


def cartesian_preprocess(trajectory, time_steps=32, use_interpolation=False,
                         return_original_times=False, scene_normalize=True):
    """
    Convert a raw trajectory to local Cartesian coordinates (meters) and
    apply scene-level normalization. Output features: [x_norm, y_norm, vx_norm, vy_norm].
    """
    if isinstance(trajectory, list):
        trajectory = np.array(trajectory)
    if trajectory.shape[1] >= 6:
        timestamp = trajectory[:, -1].copy()
        lat = trajectory[:, 0].copy()
        lon = trajectory[:, 1].copy()
        sog = trajectory[:, 2].copy()
        cog = trajectory[:, 3].copy()
    else:
        timestamp = trajectory[:, 0].copy()
        lat = trajectory[:, 1].copy()
        lon = trajectory[:, 2].copy()
        sog = trajectory[:, 3].copy()
        cog = trajectory[:, 4].copy()

    ref_lat, ref_lon = lat[0], lon[0]
    x_m, y_m = latlon_to_meters(lat, lon, ref_lat, ref_lon)

    sog_ms = sog * 0.514444  # knots -> m/s
    cog_rad = np.radians(cog)
    vx_ms = sog_ms * np.sin(cog_rad)
    vy_ms = sog_ms * np.cos(cog_rad)

    if scene_normalize:
        max_abs = max(np.max(np.abs(x_m)), np.max(np.abs(y_m)))
        if max_abs < 1e-6:
            max_abs = 1.0
        scale = max_abs
        x_norm = x_m / scale
        y_norm = y_m / scale
        vx_norm = vx_ms / scale
        vy_norm = vy_ms / scale
    else:
        x_norm, y_norm = x_m, y_m
        vx_norm, vy_norm = vx_ms, vy_ms

    features = np.column_stack([x_norm, y_norm, vx_norm, vy_norm])
    timestamps = timestamp.copy()

    if len(features) > time_steps:
        features = features[:time_steps]
        timestamps = timestamps[:time_steps]
    elif len(features) < time_steps:
        if use_interpolation:
            old_idx = np.arange(len(features))
            new_idx = np.linspace(0, len(features) - 1, time_steps)
            interp = interp1d(old_idx, features, axis=0, kind="linear")
            features = interp(new_idx)
            time_interp = interp1d(old_idx, timestamps, kind="linear")
            timestamps = time_interp(new_idx)
        else:
            pad_len = time_steps - len(features)
            features = np.vstack([features, np.zeros((pad_len, 4))])
            timestamps = np.concatenate([timestamps, np.full(pad_len, timestamps[-1])])

    if return_original_times:
        return features, timestamps
    return features