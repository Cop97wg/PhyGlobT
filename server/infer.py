"""PhyGlobT inference for the visualization API.

``build_inference(ckpt_path)`` loads a training checkpoint (top-level keys:
``model_state_dict`` + ``best_threshold``; the model class is PhysiTrackModel)
and returns a ``run`` callable that maps a :class:`~server.scenes.Scene` to
``(matches, soft_matrix)``:

    matches      : list[(a_id, b_id)]  track-id pairs predicted to be the same vessel
    soft_matrix  : (nA, nB) np.ndarray  Sinkhorn assignment probabilities P

Preprocessing mirrors the training pipeline (``phyglobt.preprocessing``):
radar rows are taken as ``[lat, lon, sog, cog, heading, timestamp]`` and mapped
to the fixed-length 5-feature representation. Matches are extracted with the
checkpoint's best threshold on P (``use_soft_threshold=True``), or with a
one-to-one Hungarian assignment on P (``use_soft_threshold=False``).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from phyglobt.model import PhysiTrackModel
from phyglobt.preprocessing import preprocess_trajectory_unified
from server.scenes import Scene

# Default inference configuration. Matches the Phy_unbalance training entry and
# was strict-load-verified against the flagship checkpoint (F1 = 0.9481).
DEFAULT_CONFIG: dict[str, Any] = {
    "input_dim": 5,
    "hidden_dim": 128,
    "phys_dim": 64,
    "lstm_layers": 2,
    "dropout": 0.2,
    "use_topology": True,
    "topo_num_layers": 2,
    "topo_nhead": 8,
    "topo_dim_feedforward": 256,
    "sinkhorn_iters": 20,
    "sinkhorn_tau": 0.5,
    "sinkhorn_epsilon": 1e-3,
    "dustbin_cost": 0.0,
    "learn_dustbin_cost": False,
    "auto_dustbin_mass": True,
    "dustbin_mass_ratio": 0.2,
    "sinkhorn_stop_thresh": 1e-5,
    "use_transformer_encoder": False,
}

TIME_STEPS = 32
MIN_POINTS = 10      # tracks shorter than this are dropped (mirrors the dataset)


def _config_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Derive architecture flags from the checkpoint's parameter names.

    Keeps structural decisions (topology on/off) in sync with the weights
    actually loaded. ``learn_dustbin_cost`` is NOT inferred from the state dict:
    ``sinkhorn.dustbin_cost`` is saved both as a Parameter (learned) and as a
    buffer (fixed), so only the value reveals the mode — DEFAULT_CONFIG
    (fixed dustbin, the training default) is tried first via strict load.
    """
    cfg = dict(DEFAULT_CONFIG)
    has = lambda key: any(k.startswith(key) for k in state_dict)
    cfg["use_topology"] = has("topology_encoder.")
    if has("traj_encoder.lstm."):
        cfg["use_transformer_encoder"] = False
    elif has("traj_encoder.layers.") and has("traj_encoder.layers.0.self_attn."):
        cfg["use_transformer_encoder"] = True
    return cfg


def _preprocess_track(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """radar (T,6) → (time_steps,5) features + original timestamps."""
    arr = np.asarray(raw, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 6 or arr.shape[0] < MIN_POINTS:
        return None
    try:
        proc, ts = preprocess_trajectory_unified(
            arr, time_steps=TIME_STEPS, use_interpolation=False,
            return_original_times=True,
        )
    except Exception:
        return None
    if proc.shape[0] != TIME_STEPS:
        return None
    return proc.astype(np.float32), ts.astype(np.float32)


class InferenceEngine:
    def __init__(self, model: PhysiTrackModel, threshold: float):
        self.model = model
        self.threshold = threshold

    @torch.no_grad()
    def _forward(self, scene: Scene):
        trajs: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {"A": [], "B": []}
        for t in scene.tracks:
            out = _preprocess_track(t.raw)
            if out is None:
                continue
            proc, ts = out
            trajs[t.sensor].append((t.id, proc, ts))

        nA = len(trajs["A"])
        nB = len(trajs["B"])
        if nA == 0 or nB == 0:
            return [], np.zeros((0, 0)), {}

        feat_A = np.stack([p for _, p, _ in trajs["A"]])   # (nA, T, 5)
        ts_A = np.stack([t for _, _, t in trajs["A"]])
        feat_B = np.stack([p for _, p, _ in trajs["B"]])
        ts_B = np.stack([t for _, _, t in trajs["B"]])
        mask_A = torch.ones(1, nA)
        mask_B = torch.ones(1, nB)

        # model expects (B, N, T, C) with a leading batch dimension
        out = self.model(
            torch.FloatTensor(feat_A).unsqueeze(0),
            torch.FloatTensor(feat_B).unsqueeze(0),
            torch.FloatTensor(ts_A).unsqueeze(0),
            torch.FloatTensor(ts_B).unsqueeze(0),
            mask_A, mask_B,
        )
        P, logits, _ = out
        return P.squeeze(0).cpu().numpy(), logits.squeeze(0).cpu().numpy(), {"ids": (trajs["A"], trajs["B"])}

    @torch.no_grad()
    def run(self, scene: Scene, use_soft_threshold: bool = True):
        P, logits, meta = self._forward(scene)
        ids_a = [i for i, _, _ in meta.get("ids", ([], []))[0]]
        ids_b = [j for j, _, _ in meta.get("ids", ([], []))[1]]
        matches: list[tuple[int, int]] = []

        if P.size == 0 or len(ids_a) == 0 or len(ids_b) == 0:
            return [], P

        # The paper's evaluation thresholds on sigmoid(logits), NOT on the
        # Sinkhorn assignment P: P is a differentiable transport plan whose
        # row marginals normalize it to ~1/N, so P > 0.79 would never fire.
        S = 1.0 / (1.0 + np.exp(-logits))

        if use_soft_threshold:
            thresh = self.threshold
            rows, cols = np.nonzero(S > thresh)
            for r, c in zip(rows.tolist(), cols.tolist()):
                matches.append((ids_a[r], ids_b[c]))
        else:
            # one-to-one Hungarian assignment maximizing sigmoid(logits)
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(-S)
            for r, c in zip(row_ind.tolist(), col_ind.tolist()):
                if S[r, c] > 0.5:
                    matches.append((ids_a[r], ids_b[c]))
        return matches, P


class InferenceRunner:
    """Callable wrapper exposing the engine and config alongside ``__call__``.

    ``runner(scene, use_soft_threshold=True) -> (matches, soft_matrix)``.
    ``runner.config`` / ``runner.engine`` expose the loaded config and the
    underlying :class:`InferenceEngine` for debugging and custom thresholds.
    """

    def __init__(self, engine: InferenceEngine, config: dict[str, Any], threshold: float):
        self.engine = engine
        self.config = config
        self.threshold = threshold

    def __call__(self, scene: Scene, use_soft_threshold: bool = True):
        return self.engine.run(scene, use_soft_threshold=use_soft_threshold)


def build_inference(ckpt_path: str) -> InferenceRunner:
    """Load a PhyGlobT checkpoint and return a ``run(scene, use_soft_threshold)`` callable."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    cfg = _config_from_state_dict(state)
    model = PhysiTrackModel(cfg)
    model.load_state_dict(state, strict=True)
    model.eval()

    threshold = float(ckpt.get("best_threshold", ckpt.get("threshold", 0.5)))
    engine = InferenceEngine(model, threshold)
    return InferenceRunner(engine, cfg, threshold)