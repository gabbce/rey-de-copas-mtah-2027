"""Player-only, long-context readings of whether the ball is in play (after Lang et al. 2022).

Lang et al. predict the in-game status from the players alone, with their movement up to +-80 s
around the frame: play stopping shows in everyone slowing down and regrouping long before and after
the restart, whether or not the ball is seen. These columns are that idea in the fresh pipeline:
team-neutral (unchanged when home and away are exchanged), computed from the players only, so the
same numbers come out of DFL and of the match whatever happens to the ball. They are added to the
regression's inputs (set "long" in fresh/run_lomo.py) and judged on DFL only.
"""

import warnings

import numpy as np
from feats import MAX_PLAYER_SPEED, player_step, rolling_mean

WINDOWS = (4, 10, 20, 40, 80)  # seconds, each looked at before and after the frame
SLOW_WINDOWS = (4, 20, 80)


def _names():
    out = []
    for w in WINDOWS:
        out += [f"lc_speed_past{w}", f"lc_speed_next{w}"]
    for w in SLOW_WINDOWS:
        out += [f"lc_slow_past{w}", f"lc_slow_next{w}"]
    out += ["lc_spread", "lc_spread_past10", "lc_spread_next10", "lc_off_pitch",
            "lc_centre", "lc_still_past2", "lc_still_next2"]
    return tuple(out)


NAMES = _names()


def compute(players, fps, pitch):
    """(n_frames, len(NAMES)) float32, finite everywhere."""
    length, width = pitch
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        step = player_step(players, fps)
        vel = np.concatenate([np.full((1, *step.shape[1:]), np.nan), step])
        speed = np.hypot(vel[..., 0], vel[..., 1])
        speed[speed > MAX_PLAYER_SPEED] = np.nan
        med = np.nanmedian(speed, axis=1)
        slow = np.nanmean(np.where(np.isfinite(speed), speed < 1.0, np.nan), axis=1)
        still = np.nanmean(np.where(np.isfinite(speed), speed < 0.5, np.nan), axis=1)
        centroid = np.nanmean(players, axis=1)
        off = players - centroid[:, None, :]
        spread = np.nanmean(np.hypot(off[..., 0], off[..., 1]), axis=1)
        x, y = players[..., 0], players[..., 1]
        outside = ((x < -1) | (x > length + 1) | (y < -1) | (y > width + 1)).sum(1).astype(float)
        centre = np.abs(centroid[:, 0] - length / 2)

    def roll(values, before_s, after_s, fill, lo, hi):
        before = round(before_s * fps) if before_s >= 0 else -1
        after = round(after_s * fps) if after_s >= 0 else -1
        return np.clip(np.nan_to_num(rolling_mean(values, before, after), nan=fill), lo, hi)

    cols = []
    for w in WINDOWS:
        cols += [roll(med, w, -1, 2.0, 0, 6), roll(med, -1, w, 2.0, 0, 6)]
    for w in SLOW_WINDOWS:
        cols += [roll(slow, w, -1, 0.5, 0, 1), roll(slow, -1, w, 0.5, 0, 1)]
    cols += [roll(spread, 2, 2, 15.0, 0, 40) / 10, roll(spread, 10, -1, 15.0, 0, 40) / 10,
             roll(spread, -1, 10, 15.0, 0, 40) / 10, roll(outside, 2, 2, 0.0, 0, 5),
             roll(centre, 2, 2, 20.0, 0, 60) / 10, roll(still, 2, -1, 0.3, 0, 1),
             roll(still, -1, 2, 0.3, 0, 1)]
    matrix = np.column_stack(cols).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("longctx.compute produced non-finite values")
    return matrix
