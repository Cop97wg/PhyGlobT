"""PhyGlobT backend — FastAPI server.

Exposes the visualization API consumed by ``web/``:

    GET  /api/health                    backend status + checkpoint list
    GET  /api/datasets/scenes           list available scene .npy files
    GET  /api/datasets/scenes/{name}    scene detail (tracks + ground truth)
    POST /api/datasets/scenes/upload    import a local scene .npy file
    POST /api/simulate                  re-simulate a scene from user params
    GET  /api/model/checkpoints         list model checkpoints
    POST /api/model/infer               run association on a scene (method-selectable)

Environment:
    DATA_DIR   : directory containing scene .npy files (default ./data/scenes)
    CKPT_DIR   : directory containing model checkpoints (default ./checkpoints)
    HOST/PORT  : bind address (default 127.0.0.1:8000)

Run:
    uvicorn server.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import math
import os
import re
import time
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .scenes import load_scene
from .infer import build_inference
from .traditional import METHODS as TRADITIONAL_METHODS
from .traditional import run_traditional
from .deep_baselines import DEEP_VARIANTS, build_deep_inference
from .simulator import parse_sim_params, params_to_dict, simulate_scene, attach_ground_truth

app = FastAPI(title="PhyGlobT API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "scenes"))
CKPT_DIR = os.environ.get("CKPT_DIR", os.path.join(os.path.dirname(__file__), "..", "checkpoints"))


# ── Pydantic request/response models ────────────────────────────────
# Lazy-loaded LSTM deep-baseline engines (loaded on first inference, not import,
# so cold-starting uvicorn for the dev server is still snappy).
_DEEP_ENGINES: dict[str, Any] = {}

# In-memory registry of re-simulated scenes (POST /api/simulate). Simulated
# scenes keep the base scene's track ids / GT pairs, live only for this server
# lifetime, and are served by the same scene endpoints as disk scenes.
_SIM_SCENES: dict[str, Any] = {}
_SIM_COUNTER = {"n": 0}


def _get_deep_engine(method: str):
    if method not in _DEEP_ENGINES:
        info = DEEP_VARIANTS[method]
        path = os.path.join(os.path.dirname(__file__), "..", info["ckpt"])
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail=f"Checkpoint not found: {info['ckpt']}")
        _DEEP_ENGINES[method] = build_deep_inference(
            path, kind=info.get("kind", "lstm_pair"), threshold=info.get("threshold", 0.5)
        )
    return _DEEP_ENGINES[method]


class InferenceRequest(BaseModel):
    scene: str
    checkpoint: str = ""
    # Association method:
    #   'phyglobt'            → PhyGlobT (needs checkpoint)
    #   'nn', 'jpda', 'hungarian' → traditional baselines (no checkpoint)
    #   'lstm', 'transformer', 'lstm_sinkhorn', 'phyglobt_nophysloss'
    #                          → deep baselines / ablations (no checkpoint)
    method: str = "phyglobt"
    use_soft_threshold: bool = True


class SimulateRequest(BaseModel):
    scene: str
    params: dict[str, Any] = {}


class AssociationOut(BaseModel):
    matches: list[dict[str, int]]
    soft_matrix: list[list[float]] | None = None
    metrics: dict[str, float]
    method: str = "phyglobt"


# ── Helpers ──────────────────────────────────────────────────────────
def _domain_of(scene) -> str:
    """Infer dataset domain from the scene latitude offset.

    US scenes sit in the Caribbean / Gulf of Mexico (lat ≈ 18–30), the US East
    Coast (≈ 40–41) and the US West Coast / Puget Sound (≈ 47–49); DM (Denmark)
    scenes sit in the Baltic (lat ≈ 54–58), so DM is the 52–62 band.
    """
    lat_off = scene.metadata.get("lat_offset")
    if lat_off is None:
        return "US"  # conservative default
    try:
        return "DM" if 52.0 <= float(lat_off) < 62.0 else "US"
    except (TypeError, ValueError):
        return "US"


def _density_of(scene) -> str:
    return str(scene.metadata.get("density_level", "unknown"))


# ── Sensor + error-model helpers ──────────────────────────────────────
_M_PER_DEG_LAT = 111320.0

# Radar noise model parameters — ACTUAL generator config (radar_sim.json),
# empirically verified against the shipped scenes by radial/tangential
# decomposition (measured σρ 22.7/42.0/59.2/80.1 m vs configured
# 22.6/40.8/56.7/80.9 m across range bins). The noise is range-dependent:
#   σρ = 30 m · (r/10 km)^0.8,  σθ = 0.5° · (r/10 km)^0.8   (capped at 4×)
# Effective miss rate = 1 − Pd·(1−drop) = 1 − 0.85·0.8 ≈ 0.32 (measured 0.318).
# Difficulty levels differ in TARGET DENSITY only, not in sensor noise.
_DIFFICULTY_ERR = {
    "50_100": {"difficulty": "Easy", "sigma_rho_m": 30.0, "sigma_theta_deg": 0.5,
               "drop_rate": 0.32, "window_overlap": 0.9},
    "100_200": {"difficulty": "Medium", "sigma_rho_m": 30.0, "sigma_theta_deg": 0.5,
                "drop_rate": 0.32, "window_overlap": 0.7},
    "200_plus": {"difficulty": "Hard", "sigma_rho_m": 30.0, "sigma_theta_deg": 0.5,
                 "drop_rate": 0.32, "window_overlap": 0.5},
}


def _sensor_info(scene) -> list[dict[str, Any]]:
    """Convert stored sensor positions (local meters) to lat/lon + detection range.

    Scene metadata stores ``sensor_{1,2}_position`` as (x_east_m, y_north_m)
    offsets from the SCENE CENTER (``lat_offset + lat_center``,
    ``lon_offset + lon_center``); ``lat = center_lat + y`` and
    ``lon = center_lon + x`` after scaling meters → degrees.
    (Verified: under this origin ~100% of each sensor's observed track points
    fall inside its ``max_range`` circle; the bottom-left origin puts the
    sensors ~40 km south of the track cloud.)
    """
    meta = scene.metadata
    lat_off = float(meta.get("lat_offset", 0.0)) + float(meta.get("lat_center", 0.0))
    lon_off = float(meta.get("lon_offset", 0.0)) + float(meta.get("lon_center", 0.0))
    cos_lat = max(math.cos(math.radians(lat_off)), 1e-9)
    out: list[dict[str, Any]] = []
    for sid in (1, 2):
        pos = meta.get(f"sensor_{sid}_position")
        rng = meta.get(f"sensor_{sid}_max_range")
        if pos is None:
            continue
        x_m, y_m = float(pos[0]), float(pos[1])
        out.append({
            "id": sid,
            "lat": lat_off + y_m / _M_PER_DEG_LAT,
            "lon": lon_off + x_m / (_M_PER_DEG_LAT * cos_lat),
            "max_range_m": float(rng) if rng is not None else None,
        })
    return out


def _error_model(scene) -> dict[str, Any] | None:
    """Radar noise parameters — user sim params take precedence, then the
    generator defaults for the scene's difficulty tier."""
    sim = scene.metadata.get("sim_params")
    if isinstance(sim, dict):
        return {
            "difficulty": "Simulated",
            "sigma_rho_m": float(sim.get("sigma_rho_m", 30.0)),
            "sigma_theta_deg": float(sim.get("sigma_theta_deg", 0.5)),
            "drop_rate": round(1.0 - float(sim.get("detection_rate", 0.68)), 3),
            "window_overlap": None,
            "source": "user-configured re-simulation: σρ={:.0f} m·(r/{} km)^{}, σθ={:.2f}°, "
                      "detection {:.0%}, sweep {:.0f} s, async ±{:.0f} s".format(
                          float(sim.get("sigma_rho_m", 30.0)),
                          float(sim.get("noise_ref_m", 10000.0)) / 1000.0,
                          float(sim.get("noise_exponent", 0.8)),
                          float(sim.get("sigma_theta_deg", 0.5)),
                          float(sim.get("detection_rate", 0.68)),
                          float(sim.get("sweep_interval_s", 10.0)),
                          float(sim.get("async_jitter_s", 2.0))),
        }
    density = _density_of(scene)
    cfg = _DIFFICULTY_ERR.get(density)
    if cfg is None:
        return None
    return {
        **cfg,
        "source": "generator config (radar_sim.json): σρ=30 m·(r/10km)^0.8, σθ=0.5°·(r/10km)^0.8, "
                  "Pd=0.85, drop=0.2 → effective miss ≈0.32; sweep 10 s (6 rpm); async ±2 s",
    }


def _list_scenes() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if os.path.isdir(DATA_DIR):
        for fn in sorted(os.listdir(DATA_DIR)):
            if fn.endswith(".npy"):
                meta = _scene_summary(fn)
                meta["simulated"] = False
                out.append(meta)
    # Simulated scenes live in memory; surface them after the disk scenes.
    for name, scene in _SIM_SCENES.items():
        out.append(_sim_summary(name, scene))
    return out


def _sim_summary(name: str, scene) -> dict[str, Any]:
    return {
        "name": name,
        "domain": _domain_of(scene),
        "density": _density_of(scene),
        "num_sensor_a": scene.num_tracks("A"),
        "num_sensor_b": scene.num_tracks("B"),
        "num_timesteps": max((len(t.points) for t in scene.tracks), default=0),
        "duration_min": round(scene.duration_minutes(), 1),
        "has_gt": scene.has_ground_truth(),
        "ais_source": bool(scene.metadata.get("ais_source", True)),
        "start_time": int(scene.metadata.get("absolute_start_time", 0)),
        "simulated": True,
    }


def _scene_summary(fn: str) -> dict[str, Any]:
    path = os.path.join(DATA_DIR, fn)
    try:
        scene = load_scene(path)
        name = fn
        domain = _domain_of(scene)
        density = _density_of(scene)
        num_a = scene.num_tracks("A")
        num_b = scene.num_tracks("B")
        n_ts = max((len(t.points) for t in scene.tracks), default=0)
        dur_min = scene.duration_minutes()
        return {
            "name": name,
            "domain": domain,
            "density": density,
            "num_sensor_a": num_a,
            "num_sensor_b": num_b,
            "num_timesteps": n_ts,
            "duration_min": round(dur_min, 1),
            "has_gt": scene.has_ground_truth(),
            "ais_source": bool(scene.metadata.get("ais_source", True)),
            "start_time": int(scene.metadata.get("absolute_start_time", 0)),
        }
    except Exception as exc:  # malformed file — skip rather than crash the listing
        return {"name": fn, "domain": "?", "density": "?", "num_sensor_a": 0,
                "num_sensor_b": 0, "num_timesteps": 0, "duration_min": 0.0,
                "has_gt": False, "ais_source": True, "start_time": 0,
                "error": str(exc)}


def _scene_from_name(name: str):
    if not name.endswith(".npy"):
        name = name + ".npy"
    if name in _SIM_SCENES:
        return _SIM_SCENES[name]
    path = os.path.join(DATA_DIR, name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"Scene not found: {name}")
    return load_scene(path)


def _list_checkpoints() -> list[dict[str, Any]]:
    if not os.path.isdir(CKPT_DIR):
        return []
    out: list[dict[str, Any]] = []
    for fn in sorted(os.listdir(CKPT_DIR)):
        if fn.endswith((".pt", ".pth", ".ckpt")):
            entry: dict[str, Any] = {"id": fn, "domain": "US", "metrics": None}
            info = _checkpoint_info(os.path.join(CKPT_DIR, fn))
            if info is not None:
                entry.update(info)
            out.append(entry)
    return out


def _checkpoint_info(path: str) -> dict[str, Any] | None:
    """Best-effort metadata read from a checkpoint dict (never crashes listing)."""
    try:
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(ckpt, dict):
        return None
    info: dict[str, Any] = {}
    if ckpt.get("best_threshold") is not None:
        info["threshold"] = float(ckpt["best_threshold"])
    for key in ("precision", "recall", "f1", "accuracy", "epoch"):
        if ckpt.get(key) is not None:
            info[key] = float(ckpt[key])
    if info:
        return info
    return None


# ── Routes ────────────────────────────────────────────────────────────
@app.get("/api/health")
def health() -> dict[str, Any]:
    import torch

    return {
        "status": "ok",
        "version": "1.0.0",
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "checkpoints": [c["id"] for c in _list_checkpoints()],
        "scenes_dir": DATA_DIR,
        "num_scenes": len(_list_scenes()),
    }


@app.get("/api/datasets/scenes")
def list_scenes() -> list[dict[str, Any]]:
    return _list_scenes()


@app.get("/api/datasets/scenes/{name}")
def get_scene(name: str) -> dict[str, Any]:
    scene = _scene_from_name(name)
    sim_params = scene.metadata.get("sim_params")
    return {
        "name": name,
        "domain": _domain_of(scene),
        "density": _density_of(scene),
        "coordinate_system": scene.metadata.get("coordinate_system", "local_bottom_left"),
        "ais_source": bool(scene.metadata.get("ais_source", True)),
        "start_time": int(scene.metadata.get("absolute_start_time", 0)),
        "simulated": bool(scene.metadata.get("simulated", False)),
        "base_scene": scene.metadata.get("base_scene"),
        "sim_params": sim_params if isinstance(sim_params, dict) else None,
        "lat_offset": float(scene.metadata.get("lat_offset", 0.0)),
        "lon_offset": float(scene.metadata.get("lon_offset", 0.0)),
        "sensors": _sensor_info(scene),
        "error_model": _error_model(scene),
        "tracks": [t.to_dict() for t in scene.tracks],
        "gt": scene.ground_truth_as_list(),
    }


@app.get("/api/model/checkpoints")
def list_checkpoints() -> list[dict[str, Any]]:
    return _list_checkpoints()


def _sanitize_filename(name: str) -> str:
    """Keep only a safe basename; require the ``.npy`` scene extension."""
    base = os.path.basename(name.replace("\\", "/"))
    base = re.sub(r"[^A-Za-z0-9_.\-]", "_", base)
    if not base.lower().endswith(".npy"):
        raise HTTPException(status_code=400, detail="Only .npy scene files are supported")
    return base


@app.post("/api/datasets/scenes/upload")
async def upload_scene(file: UploadFile = File(...)) -> dict[str, Any]:
    """Import a user-supplied local scene ``.npy`` file.

    Validates the payload parses as a scene (``load_scene``), persists it into
    ``DATA_DIR`` under a sanitized name, and returns its ``SceneSummary``.
    Rejects malformed files with a 400 and a human-readable detail.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    name = _sanitize_filename(file.filename)
    dest = os.path.join(DATA_DIR, name)

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail=f"Empty file: {file.filename}")

    # Validate BEFORE persisting: parse from a temp file, then move into place.
    tmp = dest + ".uploading"
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
        load_scene(tmp)  # raises on malformed scene
        if os.path.exists(dest):
            os.remove(dest)
        os.replace(tmp, dest)
    except HTTPException:
        raise
    except Exception as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise HTTPException(status_code=400, detail=f"Invalid scene file {file.filename}: {exc}")
    return _scene_summary(name)


@app.post("/api/simulate")
def simulate_scene_endpoint(req: SimulateRequest) -> dict[str, Any]:
    """Re-simulate both radars' observations for a base scene.

    Loads the base scene's ``raw_trajectories`` (AIS ground truth), resamples
    each track on per-sensor sweep grids, and applies the user's noise /
    detection / async parameters. Track ids and GT pairs are preserved, so any
    association method can be evaluated on the result as usual.

    If ``req.scene`` is itself a previously simulated scene, its ``base_scene``
    is used as the truth source — so the user can keep re-simulating without
    manually switching back to the base scene first.
    """
    name = req.scene if req.scene.endswith(".npy") else req.scene + ".npy"

    # Resolve to the underlying base scene (simulated scenes point at one).
    base_name = name
    if name in _SIM_SCENES:
        base_name = _SIM_SCENES[name].metadata.get("base_scene") or name

    path = os.path.join(DATA_DIR, base_name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"Scene not found: {base_name}")

    try:
        import numpy as _np

        raw = _np.load(path, allow_pickle=True).item()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot read scene file: {exc}")

    params = parse_sim_params(req.params)
    base = load_scene(path)  # for the GT pair list (ids are preserved)
    _SIM_COUNTER["n"] += 1
    base_tag = base_name[:-4] if base_name.endswith(".npy") else base_name
    sim_name = f"sim_{_SIM_COUNTER['n']:03d}_{base_tag}.npy"
    if len(sim_name) > 120:
        sim_name = f"sim_{_SIM_COUNTER['n']:03d}_{_density_of(base) or 'scene'}.npy"

    scene, info = simulate_scene(raw, params, sim_name)
    attach_ground_truth(scene, base)
    scene.metadata["base_scene"] = base_name
    _SIM_SCENES[sim_name] = scene

    return {"name": sim_name, "summary": _sim_summary(sim_name, scene), "info": info}


@app.post("/api/model/infer")
def infer(req: InferenceRequest) -> AssociationOut:
    scene = _scene_from_name(req.scene)

    if req.method == "phyglobt":
        if not req.checkpoint:
            raise HTTPException(status_code=400, detail="checkpoint is required for method='phyglobt'")
        ckpt = os.path.join(CKPT_DIR, req.checkpoint)
        if not os.path.isfile(ckpt):
            raise HTTPException(status_code=404, detail=f"Checkpoint not found: {req.checkpoint}")

        run = build_inference(ckpt)
        t0 = time.perf_counter()
        matches, soft = run(scene, use_soft_threshold=req.use_soft_threshold)
        runtime_ms = (time.perf_counter() - t0) * 1000.0
    elif req.method in TRADITIONAL_METHODS:
        t0 = time.perf_counter()
        matches, soft = run_traditional(scene, method=req.method)
        runtime_ms = (time.perf_counter() - t0) * 1000.0
    elif req.method in DEEP_VARIANTS:
        run = _get_deep_engine(req.method)
        t0 = time.perf_counter()
        matches, soft = run(scene)
        runtime_ms = (time.perf_counter() - t0) * 1000.0
    else:
        choices = sorted([*TRADITIONAL_METHODS, "phyglobt", *DEEP_VARIANTS])
        raise HTTPException(status_code=400, detail=f"Unknown method: {req.method!r} (choose from {choices})")

    metrics = scene.evaluate(matches)
    metrics["runtime_ms"] = runtime_ms
    return AssociationOut(
        matches=[{"track_a": int(a), "track_b": int(b)} for a, b in matches],
        soft_matrix=[[float(x) for x in row] for row in soft] if soft is not None else None,
        metrics=metrics,
        method=req.method,
    )


# ── Static frontend (production / Docker / HF Spaces) ─────────────────
# When the React app has been built (``cd web && npm run build``), FastAPI
# serves the bundle directly so the whole demo runs as ONE service on ONE
# port — no Vite dev server needed. API routes above take precedence; any
# other GET falls back to index.html (SPA routing).
_WEB_DIST = os.environ.get(
    "WEB_DIST", os.path.join(os.path.dirname(__file__), "..", "web", "dist")
)

if os.path.isdir(_WEB_DIST):
    from fastapi.staticfiles import StaticFiles
    from starlette.exceptions import HTTPException as StarletteHTTPException

    class _SPAStaticFiles(StaticFiles):
        async def get_response(self, path: str, scope):
            try:
                return await super().get_response(path, scope)
            except StarletteHTTPException as exc:
                if exc.status_code == 404:
                    return await super().get_response("index.html", scope)
                raise

    app.mount("/", _SPAStaticFiles(directory=_WEB_DIST, html=True), name="web")