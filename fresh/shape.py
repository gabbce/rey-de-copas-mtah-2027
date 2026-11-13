"""Team-shape evidence from the players only (C1, Oct 23 2026), for frames with or without the ball.

Depth is each team's coordinate toward the goal it attacks, relative to the median over the
whole recording of the centroid of all players (about the halfway line), so no pitch length and no labels are needed. Differences are home minus away (signed
toward home, negated when the teams are exchanged); the rest is unsigned.
"""

import numpy as np
from feats import HOME, _nanreduce, rolling_mean

NAMES = ("sh_centroid", "sh_line", "sh_width", "sh_compact", "sh_gap", "sh_spread",
         "sh_line_past5", "sh_line_next5", "sh_compact_past5", "sh_compact_next5")
SIGNED = ("sh_centroid", "sh_line", "sh_width", "sh_compact", "sh_line_past5", "sh_line_next5",
          "sh_compact_past5", "sh_compact_next5")


def _team(xy, depth_sign):
    depth = depth_sign[:, None] * xy[..., 0]
    centroid = _nanreduce(np.nanmean, depth, 1)
    line = _nanreduce(lambda v, axis: np.nanpercentile(v, 20, axis=axis), depth, 1)
    width = _nanreduce(np.nanstd, xy[..., 1], 1)
    c = _nanreduce(np.nanmean, xy, 1)
    compact = _nanreduce(np.nanmean, np.linalg.norm(xy - c[:, None, :], axis=2), 1)
    return centroid, line, width, compact, c


def compute(players, team, sign, fps):
    home = team == HOME
    centre = np.nanmedian(_nanreduce(np.nanmean, players[..., 0], 1))  # the pitch's middle, roughly
    rel = players.astype(np.float64).copy()
    rel[..., 0] -= centre
    hc, hl, hw, hk, hxy = _team(rel[:, home], sign)
    ac, al, aw, ak, axy = _team(rel[:, ~home], -sign)
    line, compact = hl - al, hk - ak
    f5 = int(5 * fps)
    cols = [hc - ac, line, hw - aw, compact, np.linalg.norm(hxy - axy, axis=1),
            _nanreduce(np.nanstd, rel[..., 0], 1) + _nanreduce(np.nanstd, rel[..., 1], 1),
            rolling_mean(line, f5, -1), rolling_mean(line, -1, f5),
            rolling_mean(compact, f5, -1), rolling_mean(compact, -1, f5)]
    return np.clip(np.nan_to_num(np.column_stack(cols)), -105, 105).astype(np.float32)
