"""Guards for running on a match other than BAR-ATH: corner origin, per-match artifacts and caches."""

import json

import data_io
import numpy as np
import pipeline
import pytest


def test_corner_origin_passes_centred_origin_fails():
    rng = np.random.default_rng(0)
    pitch = (105.0, 68.0)
    ball = rng.uniform(0, 1, (500, 2)) * np.array(pitch)
    data_io.check_origin(ball, pitch)
    with pytest.raises(ValueError, match="corner-origin"):
        data_io.check_origin(ball - np.array(pitch) / 2, pitch)


def test_artifact_of_another_match_is_refitted(tmp_path, monkeypatch):
    path = tmp_path / "model-B.json"
    body = {"game": "OTHER-GAME", "members": [], "corrector": {}}
    path.write_text(json.dumps(body))
    monkeypatch.setattr(pipeline, "artifact_path", lambda variant: path)
    assert pipeline.load("B") is None
    body["game"] = pipeline.GAME
    path.write_text(json.dumps(body))
    assert pipeline.load("B") == body


def test_match_cache_is_per_game():
    assert data_io.MATCH_CACHE.name == data_io.GAME
