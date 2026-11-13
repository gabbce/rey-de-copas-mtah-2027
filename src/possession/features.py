"""Per-frame measurements of the tracking that say something about possession.

Everything is NaN-aware: the ball is missing on 41% of graded frames and track slots come and go,
so a feature that cannot be measured on a frame is NaN there rather than a guess.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np

from possession.tracking import HOME

MAX_SPEED = 12.0  # m/s; faster than any player, so a jump this big is a tracker identity switch
BALL_SPEED_HALF_WINDOW = 2  # frames each side; one-frame differences are mostly tracker noise
# Players' collective speed separates live play from stoppages best when averaged over a few
# seconds (Lang et al. 2022 median-filter over ~3.6 s; docs/research/possession-methods.md).
PLAYER_SPEED_HALF_WINDOW = 37  # frames each side: a 3 s centred window
PLAYER_SPEED_SIDE_WINDOW = 75  # frames: the 3 s just before, and just after, a frame
# A touch (pass, control, clearance) shows as a jump between the ball's velocity just before and
# just after a frame (docs/research/possession-methods.md, Q2).
TOUCH_HALF_WINDOW = 3  # frames each side: crisp tracking (Metrica's corrected sample data)
# The competition's automatic tracking smooths the ball, spreading a kick over several frames:
# its 99th-percentile 3-frame velocity change is 4.4 m/s, against 17.8 in the sample game.
TOUCH_HALF_WINDOW_LONG = 8
# When possession changes hands the whole block turns: the side that wins the ball pushes up and
# the side that loses it drops back, visible in the players even when the ball is not.
BLOCK_HALF_WINDOW = 12  # frames each side: about a second
BLOCK_SIDE_WINDOW = 50  # frames: the 2 s just before, and just after, a frame


@dataclass(frozen=True)
class Features:
    ball_seen: np.ndarray  # bool
    d_home: np.ndarray  # metres from the ball to the nearest home player
    d_away: np.ndarray  # metres from the ball to the nearest away player
    ball_speed: np.ndarray  # m/s, over a centred 5-frame window
    ball_out: np.ndarray  # bool, ball seen outside the pitch
    player_speed: np.ndarray  # m/s, median over tracked players
    player_speed_smooth: np.ndarray  # m/s, player_speed averaged over a centred 3 s window
    player_speed_before: np.ndarray  # m/s, averaged over the 3 s before the frame
    player_speed_after: np.ndarray  # m/s, averaged over the 3 s after the frame
    spread_home: np.ndarray  # metres, mean distance of home players to their centroid
    spread_away: np.ndarray  # metres, the same for the away side
    ball_velocity_change: np.ndarray  # m/s, |velocity over the next 3 frames - over the last 3|
    ball_velocity_change_long: np.ndarray  # m/s, the same over 8 frames each side
    # Optional so hand-built Features in tests need not carry them; compute() always fills them.
    block_vx: np.ndarray | None = field(default=None)  # m/s along x, the two teams' mean
    # x-velocities averaged, over a centred ~1 s window
    block_vx_before: np.ndarray | None = field(default=None)  # the same over the 2 s before
    block_vx_after: np.ndarray | None = field(default=None)  # the same over the 2 s after
    block_x: np.ndarray | None = field(default=None)  # metres past the halfway line along x, mean
    # x of every tracked player
    reach_home: np.ndarray | None = field(default=None)  # (n, 2) metres, min and max x of home
    reach_away: np.ndarray | None = field(default=None)  # (n, 2) metres, the same for away
    # Not from the tracking: 1 where the broadcast shows the pitch, 0 where it cuts away, NaN where
    # it does not cover the frame (possession/broadcast.py). None without the broadcast.
    live_image: np.ndarray | None = field(default=None)


def _speed(positions, fps):
    """Speed along axis 0, NaN on the first frame and wherever either neighbour is missing."""
    step = np.diff(positions, axis=0)
    speed = np.hypot(step[..., 0], step[..., 1]) * fps
    pad = np.full((1, *speed.shape[1:]), np.nan, dtype=speed.dtype)
    return np.concatenate([pad, speed])


def _window_speed(positions, fps, half):
    """Displacement from t-half to t+half over its duration; NaN near the edges or where either
    end of the window is missing."""
    speed = np.full(len(positions), np.nan, dtype=np.float32)
    if len(positions) > 2 * half:
        step = positions[2 * half:] - positions[:-2 * half]
        speed[half:-half] = np.hypot(step[:, 0], step[:, 1]) * fps / (2 * half)
    return speed


def _velocity_change(ball, fps, half):
    """|v_after - v_before| per frame, each velocity over `half` frames; NaN near gaps."""
    change = np.full(len(ball), np.nan, dtype=np.float32)
    if len(ball) > 2 * half:
        before = (ball[half:-half] - ball[:-2 * half]) * fps / half
        after = (ball[2 * half:] - ball[half:-half]) * fps / half
        jump = after - before
        change[half:-half] = np.hypot(jump[:, 0], jump[:, 1])
    return change


def _rolling_mean(values, before, after):
    """Mean of the finite values in [t - before, t + after] for every t, NaN where there are
    none. Negative `before` or `after` exclude the frame itself."""
    finite = np.isfinite(values)
    sums = np.concatenate([[0.0], np.cumsum(np.where(finite, values, 0.0), dtype=np.float64)])
    counts = np.concatenate([[0], np.cumsum(finite)])
    index = np.arange(len(values))
    low = np.clip(index - before, 0, len(values))
    high = np.clip(index + after + 1, 0, len(values))
    count = counts[high] - counts[low]
    total = sums[high] - sums[low]
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _spread(positions):
    """Mean distance of a team's tracked players to their centroid, per frame."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames with nobody tracked
        centroid = np.nanmean(positions, axis=1, keepdims=True)
        offset = positions - centroid
        return np.nanmean(np.hypot(offset[..., 0], offset[..., 1]), axis=1)


def _team_vx(players, slot_speed, fps):
    """Mean x-velocity of a team's tracked players per frame, identity switches left out."""
    vx = np.diff(players[..., 0], axis=0) * fps
    vx = np.concatenate([np.full((1, vx.shape[1]), np.nan, dtype=vx.dtype), vx])
    vx[~(slot_speed <= MAX_SPEED)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames with nobody tracked
        return np.nanmean(vx, axis=1)


def _reach(positions):
    """(n, 2): min and max x of a team's tracked players per frame."""
    if positions.shape[1] == 0:
        return np.full((len(positions), 2), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.column_stack([np.nanmin(positions[..., 0], axis=1),
                                np.nanmax(positions[..., 0], axis=1)])


def home_attack(measured, halves):
    """+1 on the frames of a half where home attacks towards growing x, -1 where it attacks the
    other way, 0 outside the halves.

    Each side's goalkeeper is its deepest player, so in a half where home defends the low-x goal,
    home's minimum x sits below away's and away's maximum above home's. Inferred per half from the
    data, so it holds whatever orientation a source uses and whether or not it flips at halftime.
    """
    out = np.zeros(len(measured.ball_seen))
    if measured.reach_home is None or measured.reach_away is None:
        return out
    for start, stop in halves:
        home, away = measured.reach_home[start:stop], measured.reach_away[start:stop]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lean = np.nansum([np.nanmedian(away[:, 0]) - np.nanmedian(home[:, 0]),
                              np.nanmedian(away[:, 1]) - np.nanmedian(home[:, 1])])
        out[start:stop] = np.sign(lean)
    return out


def compute(tracking, fps, pitch):
    ball, players = tracking.ball, tracking.players
    offset = players - ball[:, None, :]
    distance = np.hypot(offset[..., 0], offset[..., 1])
    home = tracking.team == HOME

    seen = ~np.isnan(ball[:, 0])
    length, width = pitch
    out = seen & ((ball[:, 0] < 0) | (ball[:, 0] > length) | (ball[:, 1] < 0) | (ball[:, 1] > width))

    slot_speed = _speed(players, fps)
    slot_speed[slot_speed > MAX_SPEED] = np.nan
    team_vx = np.column_stack([_team_vx(players[:, home], slot_speed[:, home], fps),
                               _team_vx(players[:, ~home], slot_speed[:, ~home], fps)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN frames are expected
        player_speed = np.nanmedian(slot_speed, axis=1)
        block_vx = np.nanmean(team_vx, axis=1)
        block_x = np.nanmean(players[..., 0], axis=1) - length / 2

    return Features(
        ball_seen=seen,
        d_home=np.fmin.reduce(np.where(home, distance, np.nan), axis=1),
        d_away=np.fmin.reduce(np.where(~home, distance, np.nan), axis=1),
        ball_speed=_window_speed(ball, fps, BALL_SPEED_HALF_WINDOW),
        ball_out=out,
        player_speed=player_speed,
        player_speed_smooth=_rolling_mean(player_speed, PLAYER_SPEED_HALF_WINDOW,
                                          PLAYER_SPEED_HALF_WINDOW),
        player_speed_before=_rolling_mean(player_speed, PLAYER_SPEED_SIDE_WINDOW, -1),
        player_speed_after=_rolling_mean(player_speed, -1, PLAYER_SPEED_SIDE_WINDOW),
        spread_home=_spread(players[:, home]),
        spread_away=_spread(players[:, ~home]),
        ball_velocity_change=_velocity_change(ball, fps, TOUCH_HALF_WINDOW),
        ball_velocity_change_long=_velocity_change(ball, fps, TOUCH_HALF_WINDOW_LONG),
        block_vx=_rolling_mean(block_vx, BLOCK_HALF_WINDOW, BLOCK_HALF_WINDOW),
        block_vx_before=_rolling_mean(block_vx, BLOCK_SIDE_WINDOW, -1),
        block_vx_after=_rolling_mean(block_vx, -1, BLOCK_SIDE_WINDOW),
        block_x=block_x,
        reach_home=_reach(players[:, home]),
        reach_away=_reach(players[:, ~home]),
    )
