"""Deep-learning association baselines (external checkpoints, separate from PhyGlobT).

These load the authors' trained checkpoints and expose them through the SAME
``(matches, soft_matrix)`` contract as :mod:`server.infer`, so the API and the
visualization treat them like any other method.

Registered variants (``DEEP_VARIANTS``)
---------------------------------------
lstm                2-layer LSTM pair encoder + pairwise score MLP,
                    decoded with one-to-one Hungarian on sigmoid scores
                    (trained on 100% of data).
transformer         Same pair-classifier head, but the trajectory encoder is a
                    2-layer Transformer (``TrajectoryTransformerEncoder``).
lstm_sinkhorn       PhyGlobT ablation ``no_phys_encoder_no_topology``: the
                    LSTM trajectory encoder + ``CostPredictor`` + unbalanced
                    Sinkhorn, i.e. the full pipeline WITHOUT the physics
                    encoder and WITHOUT the global topology encoder.
phyglobt_nophysloss PhyGlobT ablation ``no_phys_loss_no_focal``: the full
                    PhyGlobT architecture trained WITHOUT the physics
                    consistency / focal losses; reuses the PhyGlobT engine
                    (``server.infer.build_inference``) with a one-to-one
                    Hungarian decode.

Decode protocol: all deep baselines use one-to-one Hungarian assignment on
sigmoid scores gated at 0.5 (threshold-free, scale-robust). The flagship
PhyGlobT keeps its checkpoint-trained best-threshold decode. Note the ablation
checkpoints save ``best_threshold=0.01`` from a training-time protocol that
does not transfer to full-size scenes — hence the forced Hungarian decode.

All checkpoint files live in ``checkpoints/`` (copied from the original
training runs). Engines are loaded lazily on first inference.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from phyglobt.model import (
    CostPredictor,
    PositionalEncoding,
    TrajectoryFeatureEncoder,
    UnbalancedSinkhornSolver,
)
from phyglobt.preprocessing import preprocess_trajectory_unified
from server.scenes import Scene

TIME_STEPS = 32
MIN_POINTS = 10

# Sinkhorn hyper-parameters — identical to the PhyGlobT inference config
# (``server.infer.DEFAULT_CONFIG``), which the ablation training shared.
_SINKHORN_CFG: dict[str, Any] = {
    "num_iters": 20,
    "tau": 0.5,
    "epsilon": 1e-3,
    "dustbin_cost": 0.0,
    "learn_dustbin_cost": False,
    "auto_dustbin_mass": True,
    "dustbin_mass_ratio": 0.2,
    "stop_thresh": 1e-5,
}


# ────────────────────────── pair-classifier models ──────────────────────────


class _PairEncoder(nn.Module):
    """Bidirectional-LSTM trajectory encoder → projection (matches ``encoder.*`` keys)."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (N, T, 5) → (N, hidden)
        x = torch.nan_to_num(feat, nan=0.0, posinf=10.0, neginf=-10.0)
        x = torch.clamp(x, -10.0, 10.0)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers*2, N, hidden) → concat last layer's fwd/bwd
        return self.projection(torch.cat([h_n[-2], h_n[-1]], dim=-1))


class _LSTMPairModel(nn.Module):
    """BiLSTM pair encoder + pairwise score MLP (matches ``encoder.*`` + ``score_mlp.*``)."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.encoder = _PairEncoder(input_dim, hidden_dim, num_layers)
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, feat: torch.Tensor) -> torch.Tensor:
        return self.encoder(feat)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        emb_a, emb_b = self.encode(feat_a), self.encode(feat_b)
        nA, nB = emb_a.shape[0], emb_b.shape[0]
        pair = torch.cat(
            [emb_a.unsqueeze(1).expand(nA, nB, -1), emb_b.unsqueeze(0).expand(nA, nB, -1)],
            dim=-1,
        )
        return self.score_mlp(pair).squeeze(-1)


class _TransformerTrajEncoder(nn.Module):
    """Transformer trajectory encoder matching the standalone baseline checkpoint.

    Same key layout as ``phyglobt.model.TrajectoryTransformerEncoder``
    (``input_proj`` / ``pos_encoder.pe`` / ``transformer`` / ``output_proj``)
    but with a configurable ``dim_feedforward`` — the baseline was trained
    with 256, not the 2048 default.
    """

    def __init__(self, input_dim: int = 5, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 256,
                 dropout: float = 0.2, max_len: int = 100):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, d_model)
        self.d_model = d_model

    def forward(self, traj: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        B, N, T, D = traj.shape
        x = self.input_proj(traj.reshape(B * N, T, D))
        x = self.pos_encoder(x)
        mask_flat = time_mask.reshape(B * N, T)
        pad = mask_flat == 0
        valid = mask_flat.sum(dim=1) > 0
        out_flat = torch.zeros(B * N, T, self.d_model, device=traj.device)
        if valid.any():
            out_flat[valid] = self.transformer(x[valid], src_key_padding_mask=pad[valid])
        lengths = mask_flat.sum(dim=1, keepdim=True)
        emb = (out_flat * mask_flat.unsqueeze(-1)).sum(dim=1) / lengths.clamp(min=1)
        zero_len = lengths.squeeze(-1) == 0
        if zero_len.any():
            emb[zero_len] = 0.0
        return self.output_proj(emb).reshape(B, N, -1)


class _TransformerPairModel(nn.Module):
    """Transformer trajectory encoder + pairwise score MLP.

    Loads the standalone baseline checkpoint with ``strict=True``.
    """

    def __init__(self, input_dim: int = 5, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 256, dropout: float = 0.2):
        super().__init__()
        self.encoder = _TransformerTrajEncoder(
            input_dim=input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, max_len=100,
        )
        self.score_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, 1),
        )

    def encode(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (N, T, 5); encoder expects (B, N, T, D) + time mask (B, N, T)
        x = feat.unsqueeze(0)
        x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        x = torch.clamp(x, -10.0, 10.0)
        time_mask = torch.ones(1, x.shape[1], x.shape[2])
        return self.encoder(x, time_mask).squeeze(0)  # (N, d_model)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        emb_a, emb_b = self.encode(feat_a), self.encode(feat_b)
        nA, nB = emb_a.shape[0], emb_b.shape[0]
        pair = torch.cat(
            [emb_a.unsqueeze(1).expand(nA, nB, -1), emb_b.unsqueeze(0).expand(nA, nB, -1)],
            dim=-1,
        )
        return self.score_mlp(pair).squeeze(-1)


# ────────────────────── ablation: LSTM + Sinkhorn only ──────────────────────


class _LSTMSinkhornModel(nn.Module):
    """PhyGlobT minus physics encoder minus global topology encoder.

    Mirrors ``PhysiTrackModel`` with the physics pathway and topology
    transformer removed: LSTM trajectory encoding → ``CostPredictor`` →
    unbalanced Sinkhorn. Loads the ``no_phys_encoder_no_topology`` ablation
    checkpoint with ``strict=True``.
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.traj_encoder = TrajectoryFeatureEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.cost_predictor = CostPredictor(hidden_dim, hidden_dim)
        self.sinkhorn = UnbalancedSinkhornSolver(**_SINKHORN_CFG)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor):
        """(nA,T,5),(nB,T,5) → (P (nA,nB), logits (nA,nB))."""
        nA, nB = feat_a.shape[0], feat_b.shape[0]
        mask_a = torch.ones(1, nA)
        mask_b = torch.ones(1, nB)
        featA = self.traj_encoder(feat_a.unsqueeze(0), mask_a)
        featB = self.traj_encoder(feat_b.unsqueeze(0), mask_b)
        logits = self.cost_predictor(featA, featB)
        P = self.sinkhorn(logits, mask_a, mask_b)
        return P.squeeze(0), logits.squeeze(0)


# ────────────────────────────── engines ─────────────────────────────────────


def _collect_features(scene: Scene):
    """Scene → per-sensor (features, ids) using the shared preprocessing."""
    feats: dict[str, list[np.ndarray]] = {"A": [], "B": []}
    ids: dict[str, list[int]] = {"A": [], "B": []}
    for t in scene.tracks:
        arr = np.asarray(t.raw, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < MIN_POINTS or arr.shape[1] < 6:
            continue
        try:
            proc = preprocess_trajectory_unified(arr, TIME_STEPS, False)
        except Exception:
            continue
        if proc.shape[0] != TIME_STEPS:
            continue
        feats[t.sensor].append(proc.astype(np.float32))
        ids[t.sensor].append(t.id)
    return feats, ids


def _hungarian_decode(S: np.ndarray, ids_a: list[int], ids_b: list[int], threshold: float):
    """One-to-one assignment maximizing sigmoid scores, gated at ``threshold``."""
    from scipy.optimize import linear_sum_assignment

    row_ind, col_ind = linear_sum_assignment(-S)
    return [
        (int(ids_a[r]), int(ids_b[c]))
        for r, c in zip(row_ind.tolist(), col_ind.tolist())
        if S[r, c] > threshold
    ]


class _PairEngine:
    """LSTM / Transformer pair classifiers: sigmoid → one-to-one Hungarian."""

    def __init__(self, model: nn.Module, threshold: float = 0.5):
        self.model = model.eval()
        self.threshold = threshold

    @torch.no_grad()
    def run(self, scene: Scene):
        feats, ids = _collect_features(scene)
        if not ids["A"] or not ids["B"]:
            return [], np.zeros((len(ids["A"]), len(ids["B"])))
        logits = self.model(
            torch.FloatTensor(np.stack(feats["A"])),
            torch.FloatTensor(np.stack(feats["B"])),
        ).cpu().numpy()
        S = 1.0 / (1.0 + np.exp(-logits))
        return _hungarian_decode(S, ids["A"], ids["B"], self.threshold), S


class _SinkhornEngine:
    """LSTM+Sinkhorn ablation: one-to-one Hungarian decode on sigmoid(logits).

    The ablation checkpoints save ``best_threshold=0.01`` from a training-time
    evaluation protocol that does not transfer to full-size scenes (it would
    predict every pair). The one-to-one Hungarian assignment with a 0.5 gate
    on sigmoid scores is threshold-free, scale-robust, and matches the decode
    used by the other deep baselines.
    """

    def __init__(self, model: _LSTMSinkhornModel, threshold: float = 0.5):
        self.model = model.eval()
        self.threshold = threshold

    @torch.no_grad()
    def run(self, scene: Scene):
        feats, ids = _collect_features(scene)
        nA, nB = len(ids["A"]), len(ids["B"])
        if nA == 0 or nB == 0:
            return [], np.zeros((nA, nB))
        P, logits = self.model(
            torch.FloatTensor(np.stack(feats["A"])),
            torch.FloatTensor(np.stack(feats["B"])),
        )
        S = 1.0 / (1.0 + np.exp(-logits.cpu().numpy()))
        matches = _hungarian_decode(S, ids["A"], ids["B"], self.threshold)
        return matches, P.cpu().numpy()


# ────────────────────────────── registry ────────────────────────────────────

# kind: 'lstm_pair' | 'transformer_pair' | 'lstmsinkhorn' | 'phyglobt'
DEEP_VARIANTS: dict[str, dict[str, Any]] = {
    "lstm": {
        "label": "LSTM (deep baseline)",
        "ckpt": "checkpoints/lstm_us_best.pt",
        "kind": "lstm_pair",
        "threshold": 0.5,
    },
    "transformer": {
        "label": "Transformer (deep baseline)",
        "ckpt": "checkpoints/transformer_us_best.pt",
        "kind": "transformer_pair",
        "threshold": 0.5,
    },
    "lstm_sinkhorn": {
        "label": "LSTM+Sinkhorn (no phys/topo)",
        "ckpt": "checkpoints/ablation_lstmsinkhorn_best.pt",
        "kind": "lstmsinkhorn",
        "threshold": 0.5,
    },
    "phyglobt_nophysloss": {
        "label": "PhyGlobT (no phys-loss)",
        "ckpt": "checkpoints/phyglobt_nophysloss_best.pt",
        "kind": "phyglobt",
    },
}


def build_deep_inference(ckpt_path: str, kind: str = "lstm_pair", threshold: float = 0.5):
    """Load a deep-baseline checkpoint → ``run(scene) -> (matches, soft_matrix)``."""
    if kind == "phyglobt":
        # Full PhyGlobT architecture (training-level ablation): reuse the
        # flagship inference engine, but force the one-to-one Hungarian decode
        # (``use_soft_threshold=False``). The checkpoint's stored
        # ``best_threshold=0.01`` comes from a training-time protocol that does
        # not transfer to full-size scenes, so the threshold-free Hungarian
        # decode (S > 0.5) is used instead — same as the other deep baselines.
        from server.infer import build_inference

        runner = build_inference(ckpt_path)

        def run(scene: Scene, use_soft_threshold: bool = True):
            return runner(scene, use_soft_threshold=False)

        run.engine = runner.engine
        return run

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    if kind == "lstm_pair":
        model = _LSTMPairModel()
    elif kind == "transformer_pair":
        model = _TransformerPairModel()
    elif kind == "lstmsinkhorn":
        model = _LSTMSinkhornModel()
    else:
        raise ValueError(f"Unknown deep-baseline kind: {kind!r}")
    model.load_state_dict(state, strict=True)

    if kind == "lstmsinkhorn":
        engine = _SinkhornEngine(model, threshold)
    else:
        engine = _PairEngine(model, threshold)

    def run(scene: Scene):
        return engine.run(scene)

    run.engine = engine  # for debugging
    return run


# Backwards-compatible alias (older call sites).
def build_lstm_inference(ckpt_path: str, threshold: float = 0.5):
    return build_deep_inference(ckpt_path, kind="lstm_pair", threshold=threshold)
