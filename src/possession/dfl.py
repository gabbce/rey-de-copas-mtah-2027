"""DFL / IDSSE open data as labelled real football (Bassek et al. 2025, CC BY 4.0).

Seven 2022/23 German matches (doi.org/10.6084/m9.figshare.28196177), downloaded to
external/dfl/ and never committed. Unlike the Metrica sample games, every frame carries the
league's own flags -- BallStatus (alive or not) and BallPossession (home or away) -- so the
H/A/D truth comes straight from the data rather than being derived from events. Its definition of
possession is the DFL's, not the competition's, so it is good for training and for comparing
versions of the pipeline, not for tuning to the frame. Attribution: Deutsche Fussball Liga (DFL);
Bassek, M. et al., Scientific Data 12, 195 (2025).
"""

from pathlib import Path

import numpy as np

from possession.official import ROOT, A, D, H, X, segments, valid_runs
from possession.tracking import AWAY, HOME, Tracking

DFL_DIR = ROOT / "external" / "dfl"
MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")


def _paths(match_id, root):
    meta = next(Path(root).glob(f"*matchinformation*{match_id}.xml"))
    raw = next(Path(root).glob(f"*positions_raw_observed*{match_id}.xml"))
    return meta, raw


def _convert(dataset):
    """Arrays in metres plus H/A/D codes; the two halves separated by one ungraded frame."""
    from kloppy.domain import BallState

    home_team = dataset.metadata.teams[0]
    length = dataset.metadata.pitch_dimensions.pitch_length or 105.0
    width = dataset.metadata.pitch_dimensions.pitch_width or 68.0
    players = sorted({p for frame in dataset.records for p in frame.players_data},
                     key=lambda p: (p.team != home_team, str(p.player_id)))
    slot = {p: i for i, p in enumerate(players)}
    team = np.array([HOME if p.team == home_team else AWAY for p in players], dtype=np.int8)

    periods = sorted({frame.period.id for frame in dataset.records})
    sizes = [sum(1 for f in dataset.records if f.period.id == pid) for pid in periods]
    n_frames = sum(sizes) + len(periods) - 1
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    xy = np.full((n_frames, len(players), 2), np.nan, dtype=np.float32)
    codes = np.full(n_frames, X, dtype=np.int8)
    runs, row = [], 0
    for pid, size in zip(periods, sizes, strict=True):
        frames = [f for f in dataset.records if f.period.id == pid]
        runs.append((row, row + size))
        for frame in frames:
            if frame.ball_coordinates is not None:
                ball[row] = (frame.ball_coordinates.x * length, frame.ball_coordinates.y * width)
            for player, data in frame.players_data.items():
                if data.coordinates is not None:
                    xy[row, slot[player]] = (data.coordinates.x * length,
                                             data.coordinates.y * width)
            if frame.ball_state == BallState.ALIVE and frame.ball_owning_team is not None:
                codes[row] = H if frame.ball_owning_team == home_team else A
            else:
                codes[row] = D
            row += 1
        row += 1  # the gap between halves
    return Tracking(ball, xy, team), codes, tuple(runs), (length, width)


def load_dfl_match(match_id, root=DFL_DIR):
    """(Tracking, truth, runs, pitch) for one match, cached as .npz after the first slow parse."""
    cache = Path(root) / "cache" / f"{match_id}.npz"
    if cache.exists():
        with np.load(cache) as saved:
            tracking = Tracking(saved["ball"], saved["players"], saved["team"])
            runs = tuple(tuple(int(v) for v in run) for run in saved["runs"])
            return tracking, saved["truth"], runs, tuple(float(v) for v in saved["pitch"])
    from kloppy import sportec

    meta, raw = _paths(match_id, root)
    dataset = sportec.load_tracking(meta_data=str(meta), raw_data=str(raw), coordinates="kloppy")
    tracking, truth, runs, pitch = _convert(dataset)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, ball=tracking.ball, players=tracking.players, team=tracking.team,
             truth=truth, runs=np.array(runs), pitch=np.array(pitch))
    return tracking, truth, runs, pitch


def merge_short(truth, seconds, fps=25):
    """DFL's truth with every H/A spell shorter than `seconds` given to the team of the live spell
    right before it (or, after a stoppage, right after it). DFL's own flag changes possession
    more often than a controlled-possession definition, so short spells stay with the previous
    team. `seconds` is chosen on DFL. Written for fresh/ (fresh/public.py)."""
    out = truth.copy()
    limit = seconds * fps
    for a, b in valid_runs(truth):
        merged = []  # [start, stop, label], adjacent spells of one label joined as we go
        for s, e, k in segments(truth, [(a, b)]):
            if merged and merged[-1][1] == s:
                prev = merged[-1]
                if (k in (H, A) and e - s < limit and prev[2] in (H, A)) or prev[2] == k:
                    prev[1] = e  # a brief touch keeps the ball with the last team
                    continue
            merged.append([s, e, k])
        for i, (s, e, k) in enumerate(merged):  # a brief spell after a stoppage: the next team's
            if (k in (H, A) and e - s < limit and i + 1 < len(merged)
                    and merged[i + 1][0] == e and merged[i + 1][2] in (H, A)):
                merged[i][2] = merged[i + 1][2]
        for s, e, k in merged:
            out[s:e] = k
    return out
