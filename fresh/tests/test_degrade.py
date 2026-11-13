"""Degradation v2 (docs/r2-rule.md E2) on a synthetic match: no frozen stoppage ball, ends pinned,
state-dependent gaps."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import public

from possession.official import FPS, A, D, H


def _match():
    truth = np.concatenate([np.full(n, k, dtype=np.int8)
                            for k, n in ((H, 400), (D, 300), (A, 400), (D, 250), (H, 300))])
    rng = np.random.default_rng(1)
    ball = np.cumsum(rng.normal(0, 0.3, (len(truth), 2)), axis=0).astype(np.float32)
    for a, b in ((400, 700), (1100, 1350)):  # DFL: the ball frozen through each stoppage
        ball[a:b] = ball[a - 1]
    return ball, truth, [(0, len(truth))]


def test_unfreeze_moves_the_stoppage_ball():
    ball, truth, runs = _match()
    out = public.unfreeze(ball, truth, runs, np.random.default_rng(0))
    speed = np.linalg.norm(np.diff(out, axis=0), axis=1) * FPS
    dead = (truth[1:] == D) & (truth[:-1] == D)
    assert np.mean(speed[dead] < 0.03) < 0.01
    np.testing.assert_array_equal(out[truth != D], ball[truth != D])
    # no jump into the restart: the last dead frame is near the first live one
    assert np.linalg.norm(out[699] - ball[700]) < 1.0


def test_hide_by_state_rates():
    ball, truth, runs = _match()
    rng = np.random.default_rng(0)
    out = public.hide_by_state(ball, truth, runs, rng, p_dead=0.85, p_live=0.0)
    missing = ~np.isfinite(out[:, 0])
    assert missing[truth == D].mean() > 0.7
    assert missing[truth != D].mean() < 0.02


def test_version3_is_version1_with_its_blur():
    ball = np.cumsum(np.random.default_rng(1).normal(size=(400, 2)), axis=0).astype(np.float32)
    t = public.Tracking(ball, np.zeros((400, 2, 2)), np.array([0, 1]))
    saved = public.V3_BLUR_JITTER
    public.V3_BLUR_JITTER = (public.BLUR, public.JITTER)
    try:
        a = public.degrade(t, [(0, 400)], 3, version=3).ball
        b = public.degrade(t, [(0, 400)], 3, version=1).ball
    finally:
        public.V3_BLUR_JITTER = saved
    assert np.array_equal(a, b, equal_nan=True)
