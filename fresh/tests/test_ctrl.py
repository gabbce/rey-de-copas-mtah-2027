"""ctrl.project fills a gap on the straight line; compute stays finite."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ctrl


def test_project_and_finite():
    ball = np.array([[0.0, 0.0], [np.nan, np.nan], [np.nan, np.nan], [3.0, 3.0]])
    assert np.allclose(ctrl.project(ball)[1:3], [[1, 1], [2, 2]])
    players = np.zeros((4, 2, 2)) + 1.0
    x = ctrl.compute(ball, players, np.array([0, 1]), 25)
    assert x.shape == (4, len(ctrl.NAMES)) and np.isfinite(x).all()
