"""run_stack: feature width and the home/away mirror."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_stack


def test_rows_and_mirror():
    z = run_stack.rows(np.random.default_rng(0).normal(size=(300, 3)))
    assert z.shape == (300, 3 * (len(run_stack.OFFSETS) + len(run_stack.MEANS)))
    assert np.array_equal(run_stack.mirrored(run_stack.mirrored(z)), z)
