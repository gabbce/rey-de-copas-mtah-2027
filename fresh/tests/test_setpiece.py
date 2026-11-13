"""setpiece.compute: finite, team- and end-neutral, and each shape lights its own column."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import setpiece

PITCH = (105.0, 68.0)
FPS = 25


def _scene(n=60, seed=0):
    rng = np.random.default_rng(seed)
    home = rng.uniform([10, 10], [50, 58], size=(n, 11, 2))
    away = rng.uniform([55, 10], [95, 58], size=(n, 11, 2))
    players = np.concatenate([home, away], axis=1)
    players[:, 3] = np.nan  # an empty slot
    return players, np.array([0] * 11 + [1] * 11)


def _col(name):
    return setpiece.NAMES.index(name)


def test_finite_and_neutral():
    players, team = _scene()
    x = setpiece.compute(players, team, FPS, PITCH)
    assert x.shape == (len(players), len(setpiece.NAMES)) and np.isfinite(x).all()
    swapped = setpiece.compute(players, 1 - team, FPS, PITCH)
    assert np.allclose(x, swapped)
    flipped = players.copy()
    flipped[..., 0] = PITCH[0] - flipped[..., 0]
    flipped[..., 1] = PITCH[1] - flipped[..., 1]
    assert np.allclose(x, setpiece.compute(flipped, team, FPS, PITCH), atol=1e-6)


def test_shapes():
    players, team = _scene()
    x = setpiece.compute(players, team, FPS, PITCH)
    assert x[:, _col("sp_halves_now")].min() > 0.99  # every player in his own half

    throw = players.copy()
    throw[:, 0] = [40.0, -1.0]  # one player a metre beyond the touchline
    assert setpiece.compute(throw, team, FPS, PITCH)[30, _col("sp_side_now")] > 0.9

    wall = players.copy()
    wall[:, 12:16] = [[90.0, 30.0], [90.0, 30.7], [90.0, 31.4], [90.0, 32.1]]
    assert setpiece.compute(wall, team, FPS, PITCH)[30, _col("sp_wall_now")] > 0.6

    corner = players.copy()
    corner[:, 0] = [104.5, 0.5]
    corner[:, 13:22] = np.column_stack([np.full(9, 98.0), np.linspace(25, 43, 9)])
    y = setpiece.compute(corner, team, FPS, PITCH)[30]
    assert y[_col("sp_flag_now")] > 0.5 and y[_col("sp_box_now")] >= 0.9
