"""Scene loading for the PhyGlobT visualization API.

Each scene ``.npy`` file (produced by the official featurization pipeline) is a
dict with:

    observations    : list[np.ndarray]  radar observation per track (T, 6):
                        [lat, lon, sog, cog, heading, timestamp] where lat/lon
                        are LOCAL offsets (degrees) from the scene's bottom-left
                        corner: row[0] → north (lat), row[1] → east (lon).
    segment_info    : list[dict]        per-track {original_mmsi, sensor_id,
                        time_range, vessel_idx, shuffled_idx}
    relation_matrix : np.ndarray        (N, N) cross-sensor ground-truth label
                        matrix, indexed by raw observation order:
                        rel[i, j] == 1 ⟺ track i and track j are seen by
                        different sensors AND share the same original_mmsi.
    raw_trajectories: dict[str, np.ndarray]  ground-truth AIS per MMSI (T, 6),
                        same local-coordinate convention as observations.
    metadata        : dict              {lat_offset, lon_offset, density_level,
                        coordinate_system, ais_source, absolute_start_time,
                        sensor_1/2_position (meters), sensor_1/2_max_range, ...}

``sensor_id == 1`` tracks form sensor A, ``sensor_id == 2`` form sensor B.
Local coordinates are converted to real lat/lon with ``lat = lat_offset + y``,
``lon = lon_offset + x``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Point:
    t: float
    lat: float
    lon: float
    sog: float
    cog: float

    def to_dict(self) -> dict[str, float]:
        return {"t": float(self.t), "lat": float(self.lat), "lon": float(self.lon),
                "sog": float(self.sog), "cog": float(self.cog)}


@dataclass
class Track:
    id: int
    sensor: str  # 'A' | 'B'
    points: list[Point] = field(default_factory=list)
    raw: np.ndarray | None = None       # original (T, 6) local-coordinate rows
    mmsi: str = ""                      # original AIS MMSI (empty if unknown)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "sensor": self.sensor, "mmsi": self.mmsi,
                "points": [p.to_dict() for p in self.points]}


@dataclass
class Scene:
    name: str
    tracks: list[Track] = field(default_factory=list)
    gt: list[tuple[int, int]] = field(default_factory=list)  # (id_A, id_B)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── queries ─────────────────────────────────────────
    def tracks_of(self, sensor: str) -> list[Track]:
        return [t for t in self.tracks if t.sensor == sensor]

    def num_tracks(self, sensor: str) -> int:
        return len(self.tracks_of(sensor))

    def has_ground_truth(self) -> bool:
        return len(self.gt) > 0

    def duration_minutes(self) -> float:
        ts = [p.t for t in self.tracks for p in t.points]
        if not ts:
            return 0.0
        return (max(ts) - min(ts)) / 60.0

    def ground_truth_as_list(self) -> list[dict[str, int]]:
        return [{"track_a": int(a), "track_b": int(b)} for a, b in self.gt]

    def track_map(self, sensor: str) -> dict[int, Track]:
        return {t.id: t for t in self.tracks_of(sensor)}

    # ── evaluation ──────────────────────────────────────
    def evaluate(self, predicted: list[tuple[int, int]]) -> dict[str, float]:
        """Compute P/R/F1 against the ground-truth set (order-insensitive)."""
        gt = set(self.gt)
        pred = set((int(a), int(b)) for a, b in predicted)
        tp = len(gt & pred)
        fp = len(pred - gt)
        fn = len(gt - pred)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        return {"overall_f1": f1, "precision": precision, "recall": recall,
                "hard_f1": f1}  # per-scene hard == overall


def load_scene(path: str) -> Scene:
    """Load a real featurization-pipeline scene ``.npy`` file into a Scene."""
    raw = np.load(path, allow_pickle=True).item()
    obs = raw.get("observations", [])
    seg = raw.get("segment_info", [])
    rel = np.asarray(raw.get("relation_matrix", []), dtype=float)
    meta = dict(raw.get("metadata", {}))

    name = str(path).replace("\\", "/").split("/")[-1]

    # ── coordinate-system normalization ──
    # The training corpus stores observations as LOCAL degree offsets from the
    # grid cell's bottom-left corner (metadata ``lat_offset`` = absolute corner).
    # Some scene dumps (e.g. ``scenes_200_plus`` from output_npy_hard) instead
    # store ABSOLUTE WGS84 lat/lon and omit lat_offset/lon_offset. The model's
    # preprocessing encodes position as sin/cos of the offset degrees, so an
    # absolute-coordinate scene is far out of distribution: the network
    # collapses to a constant ≈0.92 score for every pair (F1 ≈ 0.01). Detect
    # that case and convert to the local convention in memory; traditional
    # baselines are origin-agnostic and the map adds ``lat_offset`` back, so
    # this is transparent to everything downstream.
    arrs = [np.asarray(o, dtype=float) for o in obs]
    good = [a for a in arrs if a.ndim == 2 and a.shape[0] > 0 and a.shape[1] >= 6]
    if good:
        lat_all = np.concatenate([a[:, 0] for a in good])
        lon_all = np.concatenate([a[:, 1] for a in good])
        if lat_all.size and np.median(np.abs(lat_all)) > 5.0:  # absolute coords
            corner_lat = float(np.floor(lat_all.min()))
            corner_lon = float(np.floor(lon_all.min()))
            for a in good:
                a[:, 0] -= corner_lat
                a[:, 1] -= corner_lon
            if not meta.get("lat_offset"):
                meta["lat_offset"] = corner_lat
            if not meta.get("lon_offset"):
                meta["lon_offset"] = corner_lon
            for key, corner in (("lat_center", corner_lat), ("lon_center", corner_lon)):
                val = meta.get(key)
                if val is not None and abs(float(val)) > 5.0:
                    meta[key] = float(val) - corner
            obs = arrs  # keep order/indices; entries are now normalized arrays

    scene = Scene(name=name, metadata=meta)
    if len(obs) == 0:
        return scene

    lat_offset = float(meta.get("lat_offset", 0.0))
    lon_offset = float(meta.get("lon_offset", 0.0))
    n = len(obs)

    # ── per-track info ──
    sids: list[int] = []
    mmsis: list[str] = []
    for i in range(n):
        info = seg[i] if i < len(seg) and isinstance(seg[i], dict) else {}
        sids.append(int(info.get("sensor_id", -1)))
        mmsis.append(str(info.get("original_mmsi", "") or ""))

    # ── build tracks (sensor A: sensor_id 1, sensor B: sensor_id 2) ──
    counter = {"A": 0, "B": 0}
    obs_id_to_track: dict[int, Track] = {}
    for i, arr in enumerate(obs):
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 6:
            continue
        sensor = "A" if sids[i] == 1 else "B"
        tid = counter[sensor]
        counter[sensor] += 1
        pts = []
        for row in arr:
            y, x = float(row[0]), float(row[1])  # local: y north, x east
            lat, lon = lat_offset + y, lon_offset + x
            pts.append(Point(t=float(row[5]), lat=lat, lon=lon,
                             sog=float(row[2]), cog=float(row[3])))
        track = Track(id=tid, sensor=sensor, points=pts,
                      raw=np.ascontiguousarray(arr), mmsi=mmsis[i])
        scene.tracks.append(track)
        obs_id_to_track[i] = track

    # ── ground truth from relation_matrix (raw-obs order) ──
    seen: set[tuple[int, int]] = set()
    if rel.ndim == 2 and rel.shape[0] == n:
        rows, cols = np.nonzero(rel > 0)
        for i, j in zip(rows.tolist(), cols.tolist()):
            if i == j or sids[i] == sids[j]:
                continue  # only cross-sensor pairs count as associations
            if i < j and (i, j) in seen:
                continue
            if j < i and (j, i) in seen:
                continue
            ta = obs_id_to_track.get(i)
            tb = obs_id_to_track.get(j)
            if ta is None or tb is None:
                continue
            a, b = (ta, tb) if ta.sensor == "A" else (tb, ta)
            scene.gt.append((a.id, b.id))
            seen.add((i, j) if i < j else (j, i))
    # dedupe kept in order
    scene.gt = list(dict.fromkeys(scene.gt))
    return scene