"""The segment forward-backward against brute-force enumeration of every path."""

import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hsmm


def brute(evidence, cost, lengths):
    n, k = evidence.shape
    scores, paths = [], []
    for path in itertools.product(range(k), repeat=n):
        segs, start = [], 0
        for t in range(1, n + 1):
            if t == n or path[t] != path[t - 1]:
                segs.append((start, t, path[start]))
                start = t
        if any(b - a < lengths[y] for a, b, y in segs[1:-1]):
            continue
        s = sum(evidence[t, path[t]] for t in range(n))
        s -= sum(cost[a[2]][b[2]] for a, b in itertools.pairwise(segs))
        scores.append(s)
        paths.append(path)
    scores = np.array(scores)
    log_z = np.logaddexp.reduce(scores)
    w = np.exp(scores - log_z)
    marg = np.zeros((n, k))
    for weight, path in zip(w, paths, strict=True):
        marg[np.arange(n), path] += weight
    return marg, log_z


def test_matches_brute_force():
    rng = np.random.default_rng(3)
    for lengths in ([1, 1, 1], [2, 2, 3], [3, 3, 1], [4, 2, 2]):
        evidence = rng.normal(size=(7, 3))
        cost = rng.uniform(0.2, 2.0, size=(3, 3))
        marg, log_z = hsmm.posteriors(evidence, cost, lengths)
        want_marg, want_z = brute(evidence, cost, lengths)
        assert math.isclose(log_z, want_z, rel_tol=1e-9, abs_tol=1e-9)
        np.testing.assert_allclose(marg, want_marg, atol=1e-9)
        np.testing.assert_allclose(marg.sum(1), 1.0, atol=1e-9)
