"""Traditional track-to-track association baselines (pure numpy, no torch).

These re-implement the classic non-learned association baselines compared in
the PhyGlobT paper. Unlike :mod:`server.infer` they consume ONLY the raw radar
observations (no neural model, no checkpoint):

- ``nn``       : gated nearest-neighbor assignment        (one-to-one, greedy)
- ``jpda``     : joint probabilistic data association     (gated marginal
                 probabilities with a no-association mass, one-to-one)
- ``hungarian``: global one-to-one assignment via          ``linear_sum_assignment``
                 on the gated score matrix

All of them return ``(matches, soft_matrix)`` with the SAME shape as model
inference (see :func:`server.infer.InferenceEngine.run`):

    matches      : list[(id_a, id_b)]  track-id pairs predicted as same vessel
    soft_matrix  : (nA, nB) np.ndarray  gated similarity scores in [0, 1]

so :mod:`server.main` can evaluate them with the identical ``Scene.evaluate``
and serialize them through the same ``AssociationOut`` schema.
"""

from __future__ import annotations

import numpy as np

from server.scenes import Scene

# ── Gating / scoring constants (tuned on the 8 demo scenes) ──────────────
# Scene span is ~30–100 km, duration ~30 min. SOG is in knots (US up to ~102,
# DM ~16). Gates are "3-sigma" cut-offs for what can be the same vessel pair.
POS_GATE_M = 3000.0    # mean-position mismatch beyond this ⇒ not the same vessel
VEL_GATE_MS = 6.0      # mean-velocity-vector difference beyond this ⇒ rejected
T_GATE_S = 120.0       # mean-timestamp mismatch beyond this ⇒ rejected
NO_ASSOC_MASS = 1.0    # JPDA no-association (clutter) mass, parity with 1 pair
SCORE_THRESHOLD = 0.5  # nn/hungarian: keep pairs with exp(-d^2/2) above this
# JPDA probabilities are row/column-normalised (geometric mean of marginals),
# so a lone candidate pair tops out at ~0.5 — use a lower, empirically-calibrated
# threshold (mean F1 82.5% across the 8 demo scenes @ 0.25).
JPDA_THRESHOLD = 0.25

_M_PER_DEG_LAT = 111_320.0


def _scene_mid_lat(scene: Scene) -> float:
    """Mean latitude (deg) over all points — used for the lon→m scaling."""
    lats = [p.lat for t in scene.tracks for p in t.points]
    return float(np.mean(lats)) if lats else 0.0


def _track_kinematics(track, m_per_deg_lon: float) -> dict | None:
    """Single-track summary: mean local position (m), mean velocity (m/s),
    mean timestamp (s). Returns None when the track has no usable rows."""
    raw = track.raw
    if raw is None or raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] < 6:
        return None
    y = raw[:, 0].astype(float)  # north offset (deg)
    x = raw[:, 1].astype(float)  # east  offset (deg)
    sog = raw[:, 2].astype(float)          # knots
    cog = np.radians(raw[:, 3].astype(float))  # deg → rad (heading north-up)
    t = raw[:, 5].astype(float)

    vx = np.mean(sog * 0.5144444 * np.sin(cog))  # east  velocity (m/s)
    vy = np.mean(sog * 0.5144444 * np.cos(cog))  # north velocity (m/s)
    return {
        "pos_x_m": float(np.mean(x) * m_per_deg_lon),  # east  (m)
        "pos_y_m": float(np.mean(y) * _M_PER_DEG_LAT),  # north (m)
        "vel_x_m_s": float(vx),
        "vel_y_m_s": float(vy),
        "t_mean_s": float(np.mean(t)),
    }


def _score_matrix(scene: Scene) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the (nA, nB) gated similarity matrix plus the row/col track ids."""
    tracks_a = scene.tracks_of("A")
    tracks_b = scene.tracks_of("B")
    nA, nB = len(tracks_a), len(tracks_b)
    ids_a = np.array([t.id for t in tracks_a], dtype=np.int64)
    ids_b = np.array([t.id for t in tracks_b], dtype=np.int64)
    if nA == 0 or nB == 0:
        return np.zeros((nA, nB)), ids_a, ids_b, np.zeros(0)

    mid_lat = _scene_mid_lat(scene)
    m_per_deg_lon = _M_PER_DEG_LAT * float(np.cos(np.radians(mid_lat)))

    ka = [_track_kinematics(t, m_per_deg_lon) for t in tracks_a]
    kb = [_track_kinematics(t, m_per_deg_lon) for t in tracks_b]
    S = np.zeros((nA, nB), dtype=np.float64)
    for i, a in enumerate(ka):
        if a is None:
            continue
        for j, b in enumerate(kb):
            if b is None:
                continue
            d_pos = np.hypot(a["pos_x_m"] - b["pos_x_m"], a["pos_y_m"] - b["pos_y_m"])
            d_vel = np.hypot(a["vel_x_m_s"] - b["vel_x_m_s"], a["vel_y_m_s"] - b["vel_y_m_s"])
            d_t = abs(a["t_mean_s"] - b["t_mean_s"])
            if d_pos > 3 * POS_GATE_M or d_vel > 3 * VEL_GATE_MS or d_t > 3 * T_GATE_S:
                continue  # hard gate
            d2 = (d_pos / POS_GATE_M) ** 2 + (d_vel / VEL_GATE_MS) ** 2 + (d_t / T_GATE_S) ** 2
            S[i, j] = float(np.exp(-0.5 * d2))
    return S, ids_a, ids_b, np.asarray([t.raw.shape[0] for t in tracks_a])


def _one_to_one_greedy(S: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Highest-score pairs first, each row and column used at most once."""
    nA, nB = S.shape
    used_a: set[int] = set()
    used_b: set[int] = set()
    out: list[tuple[int, int]] = []
    for i, j in sorted(
        ((i, j) for i in range(nA) for j in range(nB)),
        key=lambda ij: S[ij],
        reverse=True,
    ):
        if S[i, j] < threshold:
            continue
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        out.append((i, j))
    return out


def run_nn(S: np.ndarray, ids_a: np.ndarray, ids_b: np.ndarray) -> list[tuple[int, int]]:
    """Gated nearest neighbor: greedy one-to-one on the score matrix."""
    pairs = _one_to_one_greedy(S, SCORE_THRESHOLD)
    return [(int(ids_a[i]), int(ids_b[j])) for i, j in pairs]


def run_jpda(S: np.ndarray, ids_a: np.ndarray, ids_b: np.ndarray) -> list[tuple[int, int]]:
    """Joint Probabilistic Data Association (gated marginal approximation).

    Row marginals model the probability that A_i chose B_j among its gated
    candidates + the no-association mass; col marginals do the symmetric thing
    from B's perspective. The geometric mean approximates the joint marginal
    association probability; pairs above ``JPDA_THRESHOLD`` are kept one-to-one.
    """
    nA, nB = S.shape
    if nA == 0 or nB == 0:
        return []
    row_mass = S.sum(axis=1, keepdims=True) + NO_ASSOC_MASS
    col_mass = S.sum(axis=0, keepdims=True) + NO_ASSOC_MASS
    row_prob = S / row_mass
    col_prob = S / col_mass
    beta = np.sqrt(row_prob * col_prob)  # symmetric marginal approximation
    pairs = _one_to_one_greedy(beta, JPDA_THRESHOLD)
    return [(int(ids_a[i]), int(ids_b[j])) for i, j in pairs]


def run_hungarian(S: np.ndarray, ids_a: np.ndarray, ids_b: np.ndarray) -> list[tuple[int, int]]:
    """Global one-to-one assignment maximizing total score (Hungarian method)."""
    nA, nB = S.shape
    if nA == 0 or nB == 0:
        return []
    from scipy.optimize import linear_sum_assignment

    row_ind, col_ind = linear_sum_assignment(-S)
    matches: list[tuple[int, int]] = []
    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        if S[r, c] >= SCORE_THRESHOLD:
            matches.append((int(ids_a[r]), int(ids_b[c])))
    return matches


METHODS = {"nn": run_nn, "jpda": run_jpda, "hungarian": run_hungarian}


def run_traditional(scene: Scene, method: str = "nn") -> tuple[list[tuple[int, int]], np.ndarray]:
    """Run a traditional baseline on a scene.

    Returns ``(matches, soft_matrix)`` — same contract as model inference.
    ``soft_matrix`` is the gated similarity matrix (rows sensor A, cols B).
    """
    if method not in METHODS:
        raise ValueError(f"Unknown traditional method: {method!r} (choose from {sorted(METHODS)})")
    S, ids_a, ids_b, _ = _score_matrix(scene)
    matches = METHODS[method](S, ids_a, ids_b)
    return matches, S