"""Rehearsal hash: decode synthetic evidence end to end and compare the CSV's SHA-256 with a
constant. CI runs it on Linux, we develop on macOS: it fails if the output depends on the machine.
No match data is involved; the evidence is integers from a seeded generator."""

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hdecode

from possession.decode import DecodeParams, decode
from possession.official import N_FRAMES, X
from possession.submission import validate, write_submission

PARAMS = DecodeParams(team_switch=3.0, to_dead=2.0, from_dead=2.5, min_possession=25,
                      min_dead=40, shift=2)
DUR = {"live": [float(np.log(150.0)), 0.6], "dead": [float(np.log(200.0)), 0.5]}
RUNS = [(1000, 7000), (12000, 18000)]
EXPECTED = "319c24ce7d66aaee82e8f9363e09b9120b16ee3c40b10ff26d452a83c3b1bd8f"


def _codes():
    rng = np.random.default_rng(2026)
    scores = rng.integers(-300, 301, size=(N_FRAMES, 3)) / 100.0
    codes = np.full(N_FRAMES, X, dtype=np.int8)
    codes[slice(*RUNS[0])] = decode(scores[slice(*RUNS[0])], [(0, RUNS[0][1] - RUNS[0][0])],
                                    PARAMS)[: RUNS[0][1] - RUNS[0][0]]
    tables = hdecode.bonus_tables(DUR, 1.0, [25, 25, 40])
    codes[slice(*RUNS[1])] = hdecode.viterbi(scores[slice(*RUNS[1])], PARAMS, tables)
    return codes


def test_output_hash(tmp_path):
    out = write_submission(tmp_path / "rehearsal.csv", _codes())
    assert validate(out) is None
    assert hashlib.sha256(out.read_bytes()).hexdigest() == EXPECTED
