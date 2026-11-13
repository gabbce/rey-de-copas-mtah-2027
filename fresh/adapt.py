"""Experiment B: adapt the evidence to the match by EM, with no labels at all.

The learned evidence comes from DFL, whose tracking says nothing about how this match's tracker
loses the ball, how the broadcast cuts away, or how Metrica's Smart Tagging relates to possession.
Those relations are learned here from the match itself. Each frame carries a few discrete channels:

- `gap`: the ball seen, or missing in a gap of a given length;
- `broadcast`: the broadcast showing the pitch, cutting away, or not covering the frame;
- `phase`: the Smart Tagging phases naming home, away, both or neither as the side on the ball;
- `setpiece`: inside a tagged set piece (its first seconds or the rest) and which side it names.

The path model is the decoder's own (costs and minimum lengths), the evidence is the DFL model's
plus, per channel, log q(channel | state) tempered by the same scale as the model's. EM alternates
posterior state probabilities (fresh/hsmm.py) with q re-estimated from them. Home and away share
their tables up to exchanging the side a channel names, so no channel can learn a side.
"""

from dataclasses import dataclass

import feats
import hsmm
import numpy as np

from possession.decode import transition_costs
from possession.official import FPS, GRADED_RUNS, A, D, H

GAP_EDGES = (3.0, 6.0, 12.0, 25.0, 50.0)  # seconds: bins of a missing ball's gap
SETPIECE_HEAD = 5.0  # seconds: the first part of a tagged set piece, mostly dead ball
ATTACKING = {"build up", "progression", "creating chances", "attacking transition",
             "ball in final third", "ball in the box", "players in the box", "long ball"}
DEFENDING = {"defending in defensive third", "defending in middle third",
             "defending in attacking third", "defensive transition"}


@dataclass(frozen=True)
class Channel:
    name: str
    codes: np.ndarray  # (n,) int bin per frame
    n_bins: int
    mirror: tuple  # bin each bin becomes when home and away are exchanged


def gap_channel(seen):
    codes = np.zeros(len(seen), dtype=np.int64)
    starts, stops = feats.runs_of(~seen)
    for start, stop in zip(starts, stops, strict=True):
        codes[start:stop] = 1 + int(np.searchsorted(GAP_EDGES, (stop - start) / FPS))
    n = len(GAP_EDGES) + 2
    return Channel("gap", codes, n, tuple(range(n)))


def broadcast_channel(live):
    shown = feats.rolling_mean(np.asarray(live, dtype=np.float64), 12, 12)
    codes = np.where(np.isnan(shown), 2, np.where(shown >= 0.5, 0, 1)).astype(np.int64)
    return Channel("broadcast", codes, 3, (0, 1, 2))


def phase_channel(tags, n):
    side = np.zeros((n, 2), dtype=bool)  # [home on the ball, away on the ball]
    for tag in tags:
        if tag.team is None:
            continue
        if tag.code in ATTACKING:
            on_ball = tag.team
        elif tag.code in DEFENDING:
            on_ball = A if tag.team == H else H
        else:
            continue
        side[max(tag.start, 0):min(tag.stop, n), 0 if on_ball == H else 1] = True
    codes = side[:, 0].astype(np.int64) + 2 * side[:, 1]
    return Channel("phase", codes, 4, (0, 2, 1, 3))


def setpiece_channel(tags, n):
    """0 outside set pieces; else 1 + 3 * part + team, part 0 = first seconds, 1 = the rest, team
    0 = home, 1 = away, 2 = not named."""
    codes = np.zeros(n, dtype=np.int64)
    head = round(SETPIECE_HEAD * FPS)
    for tag in tags:
        if tag.code != "set pieces":
            continue
        team = 2 if tag.team is None else (0 if tag.team == H else 1)
        a, b = max(tag.start, 0), min(tag.stop, n)
        codes[a:min(a + head, b)] = 1 + team
        codes[min(a + head, b):b] = 4 + team
    mirror = (0, 2, 1, 3, 5, 4, 6)
    return Channel("setpiece", codes, 7, mirror)


def channels_for(tracking_seen, live, tags, use=("gap", "broadcast", "phase", "setpiece")):
    n = len(tracking_seen)
    build = {"gap": lambda: gap_channel(tracking_seen),
             "broadcast": lambda: broadcast_channel(live),
             "phase": lambda: phase_channel(tags, n),
             "setpiece": lambda: setpiece_channel(tags, n)}
    return [build[name]() for name in use]


def _channel_scores(channels, tables, alpha):
    n = len(channels[0].codes)
    out = np.zeros((n, 3))
    for channel, table in zip(channels, tables, strict=True):
        out += alpha * np.log(table)[channel.codes]  # table (n_bins, 3) -> (n, 3)
    return out


def _m_step(channels, gamma, mask, prior=1.0):
    tables = []
    for channel in channels:
        counts = np.zeros((channel.n_bins, 3))
        for y in (H, A, D):
            counts[:, y] = np.bincount(channel.codes[mask], weights=gamma[mask, y],
                                       minlength=channel.n_bins)
        mirror = np.array(channel.mirror)
        live = counts[:, H] + counts[mirror, A]  # home's table; away's is its mirror image
        table = np.zeros((channel.n_bins, 3))
        table[:, H] = (live + prior) / (live + prior).sum()
        table[:, A] = table[mirror, H]
        table[:, D] = (counts[:, D] + prior) / (counts[:, D] + prior).sum()
        tables.append(table)
    return tables


def posterior(evidence, decode_params, runs=GRADED_RUNS):
    """(n, 3) state marginals under the decoder's path model; 0 outside `runs`."""
    cost = transition_costs(decode_params) * 1e-4
    lengths = [decode_params.min_possession, decode_params.min_possession,
               decode_params.min_dead]
    gamma = np.zeros((len(evidence), 3))
    for start, stop in runs:
        gamma[start:stop], _ = hsmm.posteriors(evidence[start:stop], cost, lengths)
    return gamma


def em(base, channels, decode_params, alpha, iterations=8, runs=GRADED_RUNS, log=None):
    """(adapted evidence, tables, history). `base` is (n, 3) evidence already scaled; the result
    adds the channels' tempered log-likelihoods. The kickoff rule is not applied here."""
    cost = transition_costs(decode_params) * 1e-4  # nats, exactly the decoder's grid
    lengths = [decode_params.min_possession, decode_params.min_possession,
               decode_params.min_dead]
    mask = np.zeros(len(base), dtype=bool)
    for start, stop in runs:
        mask[start:stop] = True
    tables = [np.full((c.n_bins, 3), 1.0 / c.n_bins) for c in channels]
    history = []
    for it in range(iterations):
        evidence = base + _channel_scores(channels, tables, alpha)
        gamma = np.zeros_like(base)
        for start, stop in runs:
            gamma[start:stop], _ = hsmm.posteriors(evidence[start:stop], cost, lengths)
        new = _m_step(channels, gamma, mask)
        change = max(float(np.abs(np.log(a) - np.log(b)).max()) for a, b in zip(new, tables,
                                                                               strict=True))
        tables = new
        dead_share = float(gamma[mask, D].mean())
        history.append((it, change, dead_share))
        if log:
            log(f"   EM {it}: dead share {dead_share:.3f}, largest table change {change:.3f}")
        if change < 1e-3:
            break
    return base + _channel_scores(channels, tables, alpha), tables, history
