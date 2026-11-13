"""The duration-aware Viterbi against the base decoder (weight 0) and a brute force (weight > 0)."""

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hdecode

from possession.decode import DecodeParams, _quantise, transition_costs, viterbi
from possession.official import D

PARAMS = DecodeParams(team_switch=3.0, to_dead=2.0, from_dead=2.5, min_possession=3, min_dead=4)
DUR = {"live": [np.log(6.0), 0.6], "dead": [np.log(8.0), 0.5]}


def _tables(weight, dmax=12):
    old = dict(hdecode.DMAX)
    hdecode.DMAX.update({0: dmax, 1: dmax, 2: dmax})
    try:
        return hdecode.bonus_tables(DUR, weight, [3, 3, 4])
    finally:
        hdecode.DMAX.update(old)


def test_weight_zero_is_the_base_decoder():
    rng = np.random.default_rng(0)
    for _ in range(20):
        scores = rng.normal(0, 1, size=(200, 3))
        assert (hdecode.viterbi(scores, PARAMS, _tables(0.0, 60)) == viterbi(scores, PARAMS)).all()


def _score(path, scores, tables):
    q = _quantise(scores)
    cost = transition_costs(PARAMS)
    total = int(q[np.arange(len(path)), path].sum())
    runs, start = [], 0
    for t in range(1, len(path) + 1):
        if t == len(path) or path[t] != path[start]:
            runs.append((start, t, int(path[start])))
            start = t
    for i, (a, b, k) in enumerate(runs):
        if i:
            total -= int(cost[runs[i - 1][2], k])
        if 0 < i < len(runs) - 1:
            d = b - a
            table = tables[k]
            v = table[min(d, len(table)) - 1]
            if v <= hdecode.NEVER // 2:
                return None
            total += int(v)
    return total


def test_brute_force():
    rng = np.random.default_rng(1)
    tables = _tables(1.5, dmax=7)
    for _ in range(15):
        scores = rng.normal(0, 1.5, size=(11, 3))
        best = max(s for p in itertools.product(range(3), repeat=11)
                   if (s := _score(np.array(p), scores, tables)) is not None)
        got = _score(hdecode.viterbi(scores, PARAMS, tables), scores, tables)
        assert got == best
    assert D == 2
