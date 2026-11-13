"""Touch-detector evidence (Oct 23 2026): where the degraded ball changes direction or speed next
to a player, the side of that player. Signed toward home, so DFL and the match read them alike.

- tt_last / tt_next: side of the last / next touch, decayed with its distance in time (2 s);
- tt_count5: touches within 5 s either side (unsigned);
- tt_balance5: home minus away touches within 5 s, over their count.
"""

import numpy as np
from feats import HOME

NAMES = ("tt_last", "tt_next", "tt_count5", "tt_balance5")
SIGNED = ("tt_last", "tt_next", "tt_balance5")
LAG = 3  # frames
REACH = 2.5  # m


def touches(ball, players, team):
    """Per frame: +1 a home touch, -1 an away touch, 0 none."""
    n = len(ball)
    out = np.zeros(n)
    if n <= 2 * LAG:
        return out
    v1 = ball[LAG:-LAG] - ball[:-2 * LAG]
    v2 = ball[2 * LAG:] - ball[LAG:-LAG]
    s1, s2 = np.hypot(*v1.T), np.hypot(*v2.T)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (v1 * v2).sum(1) / (s1 * s2)
        speed_gap = np.abs(s2 - s1) * 25 / LAG
        turn = ((cos < np.cos(np.radians(35))) & (s1 > 0.24) & (s2 > 0.24)) | (speed_gap > 3)
        dist = np.hypot(*(players[LAG:-LAG] - ball[LAG:-LAG, None, :]).transpose(2, 0, 1))
    home = team == HOME
    dh = np.fmin.reduce(np.where(home, dist, np.nan), axis=1)
    da = np.fmin.reduce(np.where(~home, dist, np.nan), axis=1)
    near = np.fmin(dh, da) < REACH
    side = np.where(np.nan_to_num(dh, nan=99) <= np.nan_to_num(da, nan=99), 1.0, -1.0)
    hit = np.nan_to_num(turn, nan=0).astype(bool) & near
    # one touch per burst: the first frame of each run of hits
    first = hit & ~np.concatenate([[False], hit[:-1]])
    out[LAG:-LAG] = np.where(first, side, 0.0)
    return out


def _carry(t, fps, reverse=False):
    vals = t[::-1] if reverse else t
    out = np.zeros(len(t))
    last, age = 0.0, np.inf
    for i, v in enumerate(vals):
        if v:
            last, age = v, 0
        else:
            age += 1
        out[i] = last * np.exp(-age / (2 * fps)) if np.isfinite(age) else 0.0
    return out[::-1] if reverse else out


def compute(ball, players, team, fps):
    t = touches(ball, players, team)
    w = int(5 * fps)
    kernel = np.ones(2 * w + 1)
    count = np.convolve(np.abs(t), kernel, "same")
    bal = np.convolve(t, kernel, "same") / np.maximum(count, 1)
    return np.column_stack([_carry(t, fps), _carry(t, fps, True), count / 10, bal]).astype(
        np.float32)
