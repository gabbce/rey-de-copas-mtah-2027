"""Set-piece geometry, from the players only.

A restart is set up before it is taken, and the set-up has a shape the players make whatever happens
to the ball: the throw-in taker beyond a touchline, the corner taker at a flag with the box crowded,
the goal kick with the box emptied, a wall, both teams in their own halves before a kickoff. The
current features see stoppages through the players' speeds and the ball's gaps; these columns add
the shape. All team-neutral and end-neutral (unchanged when home and away, or the two ends, are
exchanged), so nothing is signed and the mirrored training rows need no change.

Seven per-frame readings, each as a mean over +-1 s, over the 5 s before and over the 5 s after:
- sp_side: how far the player furthest beyond a touchline is (0-3 m);
- sp_end: the same beyond a goal line (0-3 m);
- sp_flag: closeness of the nearest player to a corner flag, exp(-d / 2 m);
- sp_box: players inside the fuller penalty area (/10);
- sp_gkick: a player in a goal area with at most one other player in that penalty area (0/1);
- sp_wall: players with two team-mates within 1.5 m (/3, clipped at 2);
- sp_halves: both teams in opposite halves, the smaller of the two teams' shares, best of the
  two orientations (0-1).
"""

import warnings

import numpy as np
from feats import HOME, rolling_mean

BASE = ("side", "end", "flag", "box", "gkick", "wall", "halves")
NAMES = tuple(f"sp_{b}_{w}" for b in BASE for w in ("now", "past5", "next5"))
SIGNED = ()
BOX_DEPTH, BOX_HALF_WIDTH = 16.5, 20.16  # penalty area, metres (Laws of the Game)
AREA_DEPTH, AREA_HALF_WIDTH = 5.5, 9.16  # goal area
WALL_RADIUS = 1.5
CHUNK = 20000  # frames per block in the pairwise wall computation


def _beyond(players, length, width):
    """(side, end): distance of the furthest player beyond a touchline / a goal line, 0-3 m."""
    x, y = players[..., 0], players[..., 1]
    side = np.fmax(-y, y - width)
    end = np.fmax(-x, x - length)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        side = np.nanmax(side, axis=1)
        end = np.nanmax(end, axis=1)
    return np.clip(np.nan_to_num(side, nan=0.0), 0, 3), np.clip(np.nan_to_num(end, nan=0.0), 0, 3)


def _flag(players, length, width):
    corners = np.array([[0, 0], [0, width], [length, 0], [length, width]], dtype=np.float64)
    dist = np.full(players.shape[:2], np.inf)
    for cx, cy in corners:
        dist = np.fmin(dist, np.hypot(players[..., 0] - cx, players[..., 1] - cy))
    near = np.min(np.where(np.isfinite(dist), dist, np.inf), axis=1)
    return np.where(np.isfinite(near), np.exp(-near / 2.0), 0.0)


def _boxes(players, length, width):
    """(box, gkick): the fuller penalty area's count /10, and a goal kick's emptied box at either end."""
    x, y = players[..., 0], players[..., 1]
    central = np.abs(y - width / 2) <= BOX_HALF_WIDTH
    in_area_y = np.abs(y - width / 2) <= AREA_HALF_WIDTH
    box_counts, gkick = [], np.zeros(len(players))
    for depth in (x, length - x):
        in_box = (depth >= -1) & (depth <= BOX_DEPTH) & central
        in_area = (depth >= -1) & (depth <= AREA_DEPTH) & in_area_y
        n_box, n_area = in_box.sum(1), in_area.sum(1)
        box_counts.append(n_box)
        gkick = np.maximum(gkick, ((n_area >= 1) & (n_box <= 2)).astype(float))
    return np.maximum(*box_counts) / 10.0, gkick


def _wall(players, team):
    out = np.zeros(len(players))
    for side in (HOME, 1 - HOME):
        pts = players[:, team == side]
        best = np.zeros(len(players))
        for start in range(0, len(players), CHUNK):
            block = pts[start:start + CHUNK]
            d = np.hypot(block[:, :, None, 0] - block[:, None, :, 0],
                         block[:, :, None, 1] - block[:, None, :, 1])
            close = (d <= WALL_RADIUS) & np.isfinite(d)
            idx = np.arange(close.shape[1])
            close[:, idx, idx] = False
            best[start:start + CHUNK] = (close.sum(2) >= 2).sum(1)
        out = np.maximum(out, best)
    return np.clip(out / 3.0, 0, 2)


def _halves(players, team, length):
    x = players[..., 0]
    tracked = np.isfinite(x)
    left = (x < length / 2) & tracked
    shares = []
    for side in (HOME, 1 - HOME):
        mine = team == side
        n = np.maximum(tracked[:, mine].sum(1), 1)
        shares.append(left[:, mine].sum(1) / n)
    home_left, away_left = shares
    return np.maximum(np.minimum(home_left, 1 - away_left), np.minimum(1 - home_left, away_left))


def compute(players, team, fps, pitch):
    """(n_frames, len(NAMES)) float32, finite everywhere."""
    length, width = pitch
    players = players.astype(np.float64)
    team = np.asarray(team)
    side, end = _beyond(players, length, width)
    box, gkick = _boxes(players, length, width)
    base = {"side": side, "end": end, "flag": _flag(players, length, width), "box": box,
            "gkick": gkick, "wall": _wall(players, team), "halves": _halves(players, team, length)}
    one, five = round(fps), round(5 * fps)
    cols = []
    for name in BASE:
        values = base[name]
        for before, after in ((one, one), (five, -1), (-1, five)):
            cols.append(np.nan_to_num(rolling_mean(values, before, after), nan=0.0))
    matrix = np.column_stack(cols).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("setpiece.compute produced non-finite values")
    return matrix
