"""Team evidence around changes of possession, with longer context (idea 1, Oct 22).

All columns are signed toward home (negated when home and away are exchanged) and computed from the
degraded ball and the players, so DFL and the match read them the same way:
- who is about to receive: the nearest home and away players to where the ball will be 0.5 s and
  1 s ahead (its current velocity projected), as a clipped distance advantage, only while seen;
- the advantage at the ball over 5 and 10 s before and after the frame (a sustained control, not
  a touch), and its change across the frame (a handover signature);
- the block's push toward home's attacking goal over 5 and 10 s before and after.
"""

import numpy as np
from feats import HOME, _nanreduce, player_step, rolling_mean

NAMES = ("tc_recv05", "tc_recv10", "tc_adv_past5", "tc_adv_next5", "tc_adv_past10",
         "tc_adv_next10", "tc_adv_flip5", "tc_push_past5", "tc_push_next5", "tc_push_past10",
         "tc_push_next10")
SIGNED = NAMES


def compute(ball, players, team, sign, fps):
    n = len(ball)
    home = team == HOME
    seen = np.isfinite(ball[:, 0])
    vel = np.full_like(ball, np.nan, dtype=np.float64)
    if n > 4:
        vel[2:-2] = (ball[4:] - ball[:-4]) * fps / 4

    def adv_at(point):
        off = players - point[:, None, :]
        dist = np.hypot(off[..., 0], off[..., 1])
        dh = np.fmin.reduce(np.where(home, dist, np.nan), axis=1)
        da = np.fmin.reduce(np.where(~home, dist, np.nan), axis=1)
        return np.clip(da - dh, -10, 10)

    adv_now = adv_at(ball.astype(np.float64))
    near = np.exp(-np.fmin.reduce(np.hypot(*(players - ball[:, None, :]).transpose(2, 0, 1)),
                                  axis=1) / 2.0)
    signal = np.where(seen, adv_now * near, np.nan)
    cols = []
    for ahead in (0.5, 1.0):
        v = adv_at(ball + np.nan_to_num(vel) * ahead)
        cols.append(np.where(seen, np.nan_to_num(v), 0.0) / 5)
    f5, f10 = round(5 * fps), round(10 * fps)
    past5 = np.nan_to_num(rolling_mean(signal, f5, -1))
    next5 = np.nan_to_num(rolling_mean(signal, -1, f5))
    cols += [past5, next5, np.nan_to_num(rolling_mean(signal, f10, -1)),
             np.nan_to_num(rolling_mean(signal, -1, f10)), next5 - past5]
    step = player_step(players[..., 0], fps)
    vx = np.concatenate([np.full((1, step.shape[1]), np.nan), step])
    vx[np.abs(vx) > 12] = np.nan
    block = _nanreduce(np.nanmean, vx, 1)
    for w, past in ((f5, True), (f5, False), (f10, True), (f10, False)):
        r = rolling_mean(block, w, -1) if past else rolling_mean(block, -1, w)
        cols.append(sign * np.clip(np.nan_to_num(r), -6, 6))
    m = np.column_stack(cols).astype(np.float32)
    if not np.isfinite(m).all():
        raise ValueError("teamctx.compute produced non-finite values")
    return m
