"""Tests for traditional baselines + the scene upload endpoint.

Run: pytest server/test_traditional.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Must import BEFORE main imports .traditional to avoid a bare-ModuleNotFoundError
# confusion on some setups; the actual test target is the pre-existing code.
from fastapi.testclient import TestClient  # noqa: E402

from server.main import app  # noqa: E402
from server.scenes import load_scene  # noqa: E402
from server.traditional import METHODS, _score_matrix, run_traditional  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "scenes")
SCENES = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".npy"))
assert len(SCENES) >= 8, "expected the 8 demo scenes"


def _load(name: str):
    return load_scene(os.path.join(DATA_DIR, name))


# ── unit: score matrix sanity ─────────────────────────────────────────────
class TestScoreMatrix:
    def test_shape_and_range(self):
        for name in SCENES:
            S, ids_a, ids_b, _ = _score_matrix(_load(name))
            nA = len(ids_a)
            nB = len(ids_b)
            assert S.shape == (nA, nB)
            assert (0.0 <= S).all() and (S <= 1.0).all()

    def test_gt_pairs_have_high_scores(self):
        """Ground-truth pairs should rank in the top of their row/column."""
        sc = _load(SCENES[0])
        S, ids_a, ids_b, _ = _score_matrix(sc)
        idx_a = {int(a): i for i, a in enumerate(ids_a)}
        idx_b = {int(b): j for j, b in enumerate(ids_b)}
        above_half = 0
        for a, b in sc.gt:
            i, j = idx_a.get(a), idx_b.get(b)
            if i is None or j is None:
                continue
            if S[i, j] > 0.5:
                above_half += 1
        assert above_half >= 0.5 * len(sc.gt), f"GT scores too low: {above_half}/{len(sc.gt)}"


# ── unit: each method returns matches + valid metrics ─────────────────────
class TestRunTraditional:
    @pytest.mark.parametrize("method", sorted(METHODS))
    @pytest.mark.parametrize("fixture", SCENES)
    def test_all_methods_all_scenes(self, method, fixture):
        sc = _load(fixture)
        matches, S = run_traditional(sc, method=method)
        track_ids = {t.id for t in sc.tracks}
        for a, b in matches:
            assert a in track_ids and b in track_ids
        metrics = sc.evaluate(matches)
        for key in ("overall_f1", "precision", "recall", "hard_f1"):
            assert 0.0 <= metrics[key] <= 1.0

    def test_results_differ_by_method(self):
        """The three baselines should not be degenerate-identical on every scene."""
        sc = _load(SCENES[0])
        outs = {m: run_traditional(sc, method=m)[0] for m in sorted(METHODS)}
        as_sets = {m: set(o) for m, o in outs.items()}
        assert len(set(tuple(sorted(s)) for s in as_sets.values())) >= 2, (
            "all methods produced identical matches — suspicious"
        )

    def test_unknown_method_raises(self):
        sc = _load(SCENES[0])
        with pytest.raises(ValueError):
            run_traditional(sc, method="nope")


# ── integration: /api/model/infer method dispatch ─────────────────────────
class TestInferMethodDispatch:
    def test_phyglobt_needs_checkpoint(self):
        with TestClient(app) as c:
            r = c.post("/api/model/infer", json={"scene": SCENES[0], "method": "phyglobt"})
            assert r.status_code == 400
            assert "checkpoint" in r.json()["detail"]

    def test_unknown_method_400(self):
        with TestClient(app) as c:
            r = c.post(
                "/api/model/infer",
                json={"scene": SCENES[0], "method": "bogus", "checkpoint": ""},
            )
            assert r.status_code == 400
            assert "Unknown method" in r.json()["detail"]

    @pytest.mark.parametrize("method", sorted(METHODS))
    def test_traditional_method_runs(self, method):
        with TestClient(app) as c:
            r = c.post(
                "/api/model/infer",
                json={"scene": SCENES[0], "method": method, "checkpoint": ""},
            )
            assert r.status_code == 200
            body = r.json()
            assert body["method"] == method
            assert set(body["metrics"]) >= {"overall_f1", "precision", "recall", "runtime_ms"}
            for m in body["matches"]:
                assert set(m) == {"track_a", "track_b"}


# ── integration: scene upload endpoint ────────────────────────────────────
class TestUploadScene:
    def test_upload_valid_scene(self, tmp_path):
        src = os.path.join(DATA_DIR, SCENES[0])
        with open(src, "rb") as fh, TestClient(app) as c:
            files = {"file": ("my_local_scene.npy", fh.read(), "application/octet-stream")}
            r = c.post("/api/datasets/scenes/upload", files=files)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["name"] == "my_local_scene.npy"

    def test_upload_rejects_non_npy(self):
        with TestClient(app) as c:
            files = {"file": ("notes.txt", b"not a scene", "text/plain")}
            r = c.post("/api/datasets/scenes/upload", files=files)
        assert r.status_code == 400
        assert ".npy" in r.json()["detail"]

    def test_upload_rejects_garbage_npy(self):
        import io

        buf = io.BytesIO()
        np.save(buf, np.zeros((3, 3)))  # valid npy but NOT a scene dict
        with TestClient(app) as c:
            files = {"file": ("bad.npy", buf.getvalue(), "application/octet-stream")}
            r = c.post("/api/datasets/scenes/upload", files=files)
        assert r.status_code == 400