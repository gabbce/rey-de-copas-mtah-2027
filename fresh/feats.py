"""Per-frame measurements for the label-free pipeline, written from scratch.

The same function runs on the match, on DFL and on Metrica's sample games, so a model trained on
public data reads the same columns on the match. Everything is NaN-aware: a measurement that cannot
be taken on a frame is filled with a neutral value and, where it matters, flagged.

Nothing here was chosen by looking at labels of the provided match: every column is a plain physical reading
(distances, speeds, the block's movement, the ball's gaps), and which ones help is judged on public
data only (fresh/README.md).
"""

import warnings

import numpy as np

HOME = 0
MAX_PLAYER_SPEED = 12.0  # m/s: a faster jump is a tracker identity switch


def player_step(players, fps):
    """Per-frame player velocity (n-1 rows), after the E9-smooth centred position mean
    (config.SMOOTH frames; 0 or 1 = the raw positions). Works on (n, ...) arrays."""
    import config
    k = int(config.SMOOTH)
    if k > 1:
        flat = players.reshape(len(players), -1)
        half = k // 2
        flat = np.column_stack([rolling_mean(flat[:, j], half, k - 1 - half)
                                for j in range(flat.shape[1])])
        players = flat.reshape(players.shape)
    return np.diff(players, axis=0) * fps


def rolling_mean(values, before, after):
    """Mean of the finite values in [t - before, t + after]; NaN where there are none. A negative
    bound excludes the frame itself (before=-1: only the future; after=-1: only the past)."""
    finite = np.isfinite(values)
    sums = np.concatenate([[0.0], np.cumsum(np.where(finite, values, 0.0), dtype=np.float64)])
    counts = np.concatenate([[0], np.cumsum(finite)])
    index = np.arange(len(values))
    low = np.clip(index - before, 0, len(values))
    high = np.clip(index + after + 1, 0, len(values))
    count = counts[high] - counts[low]
    return np.where(count > 0, (sums[high] - sums[low]) / np.maximum(count, 1), np.nan)


def runs_of(mask):
    """(starts, stops) of the True runs of a boolean array."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return edges[::2], edges[1::2]


def _nanreduce(fn, values, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(values, axis=axis)


def attack_sign(players, team, runs):
    """+1 on frames of a half where home attacks toward growing x, -1 the other way, 0 outside.

    Each side's goalkeeper is its deepest player: in a half where home defends the low-x goal, home's
    minimum x sits below away's and away's maximum above home's."""
    home = team == HOME
    sign = np.zeros(len(players))
    x = players[..., 0]
    for start, stop in runs:
        hmin = _nanreduce(np.nanmin, x[start:stop, home], 1)
        hmax = _nanreduce(np.nanmax, x[start:stop, home], 1)
        amin = _nanreduce(np.nanmin, x[start:stop, ~home], 1)
        amax = _nanreduce(np.nanmax, x[start:stop, ~home], 1)
        lean = (_nanreduce(np.nanmedian, amin, 0) - _nanreduce(np.nanmedian, hmin, 0)
                + _nanreduce(np.nanmedian, amax, 0) - _nanreduce(np.nanmedian, hmax, 0))
        sign[start:stop] = 1.0 if lean >= 0 else -1.0
    return sign


def _gap_context(seen, values, fps):
    """For every frame inside a ball gap: log seconds of the whole gap, seconds since it began and
    until it ends (log), and `values` at the last frame before it and the first after it."""
    n = len(seen)
    length = np.zeros(n)
    since = np.zeros(n)
    until = np.zeros(n)
    before = np.zeros((n, values.shape[1]))
    after = np.zeros((n, values.shape[1]))
    starts, stops = runs_of(~seen)
    for start, stop in zip(starts, stops, strict=True):
        span = np.arange(start, stop)
        length[start:stop] = (stop - start) / fps
        since[start:stop] = (span - start + 1) / fps
        until[start:stop] = (stop - span) / fps
        if start > 0:
            before[start:stop] = values[start - 1]
        if stop < n:
            after[start:stop] = values[stop]
    return np.log1p(length), np.log1p(since), np.log1p(until), before, after


NAMES = (
    "seen", "adv", "near", "adv_near", "dmin", "ball_speed", "fast", "adv_near_fast", "ball_out",
    "line_dist", "press", "adv_near_mid", "adv_near_past", "adv_near_next", "seen_near",
    "gap_len", "gap_since", "gap_until", "gap_adv_before", "gap_adv_after", "gap_out_before",
    "gap_line_before", "speed_mid", "speed_past", "speed_next", "slow_share", "push_mid",
    "push_past", "push_next", "block_x", "spread_diff", "length_diff", "tracked",
)


def compute(tracking, runs, fps, pitch):
    """(n_frames, len(NAMES)) float32 matrix, finite everywhere, plus the attack sign."""
    ball, players, team = tracking.ball, tracking.players, tracking.team
    n = len(ball)
    home = team == HOME
    length, width = pitch
    seen = np.isfinite(ball[:, 0])

    offset = players - ball[:, None, :]
    dist = np.hypot(offset[..., 0], offset[..., 1])
    dh = np.fmin.reduce(np.where(home, dist, np.nan), axis=1)
    da = np.fmin.reduce(np.where(~home, dist, np.nan), axis=1)
    dmin = np.fmin(dh, da)
    adv = np.where(seen, np.nan_to_num(np.clip(da - dh, -10, 10)), 0.0)
    near = np.where(seen, np.nan_to_num(np.exp(-dmin / 2.0)), 0.0)
    adv_near = adv * near
    press = np.where(np.isfinite(dist), dist, 99) < 4
    press = np.where(seen, press[:, home].sum(1) - press[:, ~home].sum(1), 0).astype(float)

    speed = np.full(n, np.nan)
    if n > 4:
        step = ball[4:] - ball[:-4]
        speed[2:-2] = np.hypot(step[:, 0], step[:, 1]) * fps / 4
    fast = (np.nan_to_num(speed) > 10).astype(float)
    ball_speed = np.clip(np.nan_to_num(speed), 0, 30) / 10
    out = seen & ((ball[:, 0] < 0) | (ball[:, 0] > length) | (ball[:, 1] < 0) | (ball[:, 1] > width))
    line = np.minimum.reduce([ball[:, 0], length - ball[:, 0], ball[:, 1], width - ball[:, 1]])
    line_dist = np.where(seen, np.clip(np.nan_to_num(line), 0, 10) / 10, 0.0)

    adv_seen = np.where(seen, adv_near, np.nan)
    adv_mid = np.nan_to_num(rolling_mean(adv_seen, 12, 12))
    adv_past = np.nan_to_num(rolling_mean(adv_seen, 50, -1))
    adv_next = np.nan_to_num(rolling_mean(adv_seen, -1, 50))
    seen_near = rolling_mean(seen.astype(float), 50, 50)

    gap_len, gap_since, gap_until, before, after = _gap_context(
        seen, np.column_stack([adv_near, out.astype(float), line_dist]), fps)

    step = player_step(players, fps)
    vel = np.concatenate([np.full((1, *step.shape[1:]), np.nan), step])
    slot_speed = np.hypot(vel[..., 0], vel[..., 1])
    vel[slot_speed > MAX_PLAYER_SPEED] = np.nan
    slot_speed[slot_speed > MAX_PLAYER_SPEED] = np.nan
    med = _nanreduce(np.nanmedian, slot_speed, 1)
    speed_mid = np.clip(np.nan_to_num(rolling_mean(med, 37, 37), nan=2.0), 0, 6)
    speed_past = np.clip(np.nan_to_num(rolling_mean(med, 75, -1), nan=2.0), 0, 6)
    speed_next = np.clip(np.nan_to_num(rolling_mean(med, -1, 75), nan=2.0), 0, 6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        slow = np.nanmean(np.where(np.isfinite(slot_speed), slot_speed < 1.0, np.nan), axis=1)
    slow_share = np.nan_to_num(rolling_mean(slow, 25, 25), nan=0.5)

    sign = attack_sign(players, team, runs)
    team_vx = np.column_stack([_nanreduce(np.nanmean, vel[:, home, 0], 1),
                               _nanreduce(np.nanmean, vel[:, ~home, 0], 1)])
    block_vx = _nanreduce(np.nanmean, team_vx, 1)
    push_mid = sign * np.clip(np.nan_to_num(rolling_mean(block_vx, 12, 12)), -6, 6)
    push_past = sign * np.clip(np.nan_to_num(rolling_mean(block_vx, 50, -1)), -6, 6)
    push_next = sign * np.clip(np.nan_to_num(rolling_mean(block_vx, -1, 50)), -6, 6)
    block_x = sign * np.clip(np.nan_to_num(_nanreduce(np.nanmean, players[..., 0], 1) - length / 2),
                             -60, 60) / 10

    def spread(group):
        centroid = _nanreduce(np.nanmean, group, 1)
        off = group - centroid[:, None, :]
        return _nanreduce(np.nanmean, np.hypot(off[..., 0], off[..., 1]), 1)

    def reach(group):
        return (_nanreduce(np.nanmax, group[..., 0], 1) - _nanreduce(np.nanmin, group[..., 0], 1))

    spread_diff = np.clip(np.nan_to_num(spread(players[:, home]) - spread(players[:, ~home])),
                          -20, 20) / 5
    length_diff = np.clip(np.nan_to_num(reach(players[:, home]) - reach(players[:, ~home])),
                          -40, 40) / 10
    tracked = np.isfinite(players[..., 0]).sum(1) / 22.0

    columns = [
        seen.astype(float), adv, near, adv_near, np.where(seen, np.clip(np.nan_to_num(dmin), 0, 20)
                                                          / 10, 0.0),
        ball_speed, fast, adv_near * fast, out.astype(float), line_dist, press, adv_mid, adv_past,
        adv_next, seen_near, gap_len, gap_since, gap_until, before[:, 0], after[:, 0], before[:, 1],
        before[:, 2], speed_mid, speed_past, speed_next, slow_share, push_mid, push_past, push_next,
        block_x, spread_diff, length_diff, tracked,
    ]
    matrix = np.column_stack(columns).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("feats.compute produced non-finite values")
    return matrix, sign


# Columns whose meaning flips when home and away are exchanged (mirroring a training match).
TEAM_SIGNED = ("adv", "adv_near", "adv_near_fast", "press", "adv_near_mid", "adv_near_past",
               "adv_near_next", "gap_adv_before", "gap_adv_after", "push_mid", "push_past",
               "push_next", "block_x", "spread_diff", "length_diff")
