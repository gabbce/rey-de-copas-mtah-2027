"""E4 features (docs/r2-rule.md E4): projected ball, visibility, control history.

- pj_adv / pj_near: inside a ball gap, the ball projected on the straight line between its last
  and next seen positions, and the nearest-player advantage (away minus home distance, clipped
  to 10 m) and closeness at that point; the seen ball's own values elsewhere;
- vis_past2 / vis_next2: share of frames with the ball seen over the past / next 2 s;
- vis_gaps10: ball gaps starting within 10 s either side (/5);
- ct_past / ct_next: the seen ball's nearest-player advantage carried backward in time / forward
  with a 3 s exponential decay through gaps (signed toward home).
"""

import numpy as np
from feats import HOME, rolling_mean, runs_of

NAMES = ("pj_adv", "pj_near", "vis_past2", "vis_next2", "vis_gaps10", "ct_past", "ct_next")
SIGNED = ("pj_adv", "ct_past", "ct_next")
TAU = 3.0  # seconds


def project(ball):
    """The ball with each gap filled on the straight line between its neighbours (edges held)."""
    out = ball.astype(np.float64).copy()
    seen = np.isfinite(out[:, 0])
    if not seen.any():
        return out
    idx = np.arange(len(out))
    for k in range(2):
        out[:, k] = np.interp(idx, idx[seen], out[seen, k])
    return out


def _advantage(ball, players, team):
    home = team == HOME
    dist = np.hypot(players[..., 0] - ball[:, None, 0], players[..., 1] - ball[:, None, 1])
    dh = np.fmin.reduce(np.where(home, dist, np.nan), axis=1)
    da = np.fmin.reduce(np.where(~home, dist, np.nan), axis=1)
    adv = np.nan_to_num(np.clip(da - dh, -10, 10))
    near = np.nan_to_num(np.exp(-np.fmin(dh, da) / 2.0))
    return adv, near


def _carry(values, seen, fps, reverse=False):
    decay = np.exp(-1.0 / (TAU * fps))
    out = np.zeros(len(values))
    last = 0.0
    order = range(len(values) - 1, -1, -1) if reverse else range(len(values))
    for i in order:
        last = values[i] if seen[i] else last * decay
        out[i] = last
    return out


def compute(ball, players, team, fps):
    seen = np.isfinite(ball[:, 0])
    adv, near = _advantage(project(ball), players, team)
    w = int(2 * fps)
    vis = seen.astype(float)
    starts = np.zeros(len(ball))
    starts[runs_of(~seen)[0]] = 1.0
    w10 = int(10 * fps)
    gaps10 = np.nan_to_num(rolling_mean(starts, w10, w10)) * (2 * w10 + 1) / 5
    signal = adv * near / 10
    return np.column_stack([
        adv / 10, near, np.nan_to_num(rolling_mean(vis, w, -1), nan=1.0),
        np.nan_to_num(rolling_mean(vis, -1, w), nan=1.0), gaps10,
        _carry(signal, seen, fps), _carry(signal, seen, fps, reverse=True),
    ]).astype(np.float32)
