"""The segment corrector's hand-back rule and its features, on synthetic codes."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import corrector
import feats

from possession.official import A, D, H


def _codes(*spells):
    return np.concatenate([np.full(n, k, dtype=np.int8) for k, n in spells])


def _rows(codes, runs):
    scores = np.zeros((len(codes), 3))
    x = np.zeros((len(codes), len(feats.NAMES)))
    return corrector.segment_rows(codes, scores, x, runs)


def test_one_row_per_segment_with_its_bounds():
    codes = _codes((H, 50), (A, 10), (H, 40), (D, 30))
    rows = _rows(codes, [(0, len(codes))])
    assert [(a, b, k) for a, b, k, *_ in rows] == [(0, 50, H), (50, 60, A), (60, 100, H),
                                                   (100, 130, D)]
    assert all(len(r[3]) == len(rows[0][3]) for r in rows)


def test_a_wrong_possession_goes_to_its_longer_live_neighbour():
    codes = _codes((H, 50), (A, 10), (H, 40))
    rows = _rows(codes, [(0, 100)])
    out = corrector.corrected(codes, rows, np.array([0.0, 0.9, 0.0]), 0.5)
    assert (out == H).all()


def test_a_wrong_stoppage_goes_to_the_possession_before_it():
    codes = _codes((A, 50), (D, 10), (H, 40))
    rows = _rows(codes, [(0, 100)])
    out = corrector.corrected(codes, rows, np.array([0.0, 0.9, 0.0]), 0.5)
    assert (out[:60] == A).all() and (out[60:] == H).all()


def test_below_the_threshold_nothing_changes():
    codes = _codes((H, 50), (A, 10), (D, 40))
    rows = _rows(codes, [(0, 100)])
    out = corrector.corrected(codes, rows, np.array([0.4, 0.4, 0.4]), 0.5)
    assert (out == codes).all()


def test_segments_never_reach_across_runs():
    codes = _codes((H, 50), (A, 50))
    rows = _rows(codes, [(0, 50), (50, 100)])
    out = corrector.corrected(codes, rows, np.array([0.9, 0.9]), 0.5)
    assert (out == codes).all()  # each is alone in its run: no neighbour to give it to
