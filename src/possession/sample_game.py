"""Metrica's public sample games as a rehearsal on real football.

Source: github.com/metrica-sports/sample-data (anonymised; shared for the analytics community to use
responsibly, with the source acknowledged). It is not competition data. Games 1 and 2 come as CSV
(one tracking file per team, plus events); game 3 as FIFA EPTS tracking plus JSON events. Two
things differ from the competition bundle, so they get their own readers:

- the tracking names players and has substitutions (in game 3 the order of the groups on a line
  changes between stretches of frames, each described by a <DataFormatSpecification>);
- there is event data instead of a possession ground truth. possession_truth() derives an
  approximate one from Metrica's own event definitions: the team of the latest controlling event
  has the ball, and play is dead from the event that stopped it to the next restart. It follows
  Metrica's event conventions, not the competition's corrected-tracking ones, so it is good for
  comparing versions of the pipeline on real movement, not for tuning thresholds to the frame.
"""

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from possession.official import A, D, H, X
from possession.tracking import AWAY, HOME, Tracking

PITCH = (105.0, 68.0)  # metres, from the sample metadata
CONTROL = {"SET PIECE", "PASS", "CARRY", "SHOT", "RECOVERY"}  # events by the team in possession


def _layouts(root):
    """[(first_frame, last_frame, [player ids in group order])] from the format specifications."""
    channel_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    layouts = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        players = [channel_player[r] for r in refs if r and r.endswith("_x")]
        layouts.append((int(spec.get("startFrame")), int(spec.get("endFrame")), players))
    return layouts


def read_sample_tracking(meta_path, raw_path):
    """(Tracking, n_frames). One slot per player who ever appears; video frame = frame id - 1."""
    root = ET.parse(meta_path).getroot()
    home_id = next(root.iter("Team")).get("id")
    team_of = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    layouts = _layouts(root)
    slots = sorted({pid for _, _, players in layouts for pid in players})
    slot_of = {pid: i for i, pid in enumerate(slots)}
    team = np.array([HOME if team_of[p] == home_id else AWAY for p in slots], dtype=np.int8)

    with open(raw_path) as handle:
        lines = handle.read().splitlines()
    n_frames = max(int(line[:line.index(":")]) for line in lines if line)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, len(slots), 2), np.nan, dtype=np.float32)
    starts = np.array([first for first, _, _ in layouts])
    columns = [np.array([slot_of[p] for p in order]) for _, _, order in layouts]
    for line in lines:
        frame_part, people, ball_part = line.split(":")
        frame = int(frame_part)
        layout = int(np.searchsorted(starts, frame, side="right")) - 1
        xy = np.array([group.split(",")[:2] for group in people.split(";")], dtype=np.float32)
        players[frame - 1, columns[layout][:len(xy)]] = xy
        ball[frame - 1] = np.array(ball_part.split(",")[:2], dtype=np.float32)
    scale = np.array(PITCH, dtype=np.float32)
    return Tracking(ball * scale, players * scale, team), n_frames


def read_csv_tracking(home_path, away_path):
    """(Tracking, n_frames) from games 1-2: one CSV per team, three header rows, then Period,
    Frame, Time and an x,y pair per player (NaN when off the pitch), the ball last."""
    def side(path):
        values = pd.read_csv(path, skiprows=2).to_numpy(dtype=np.float32)
        return values[:, 1].astype(int), values[:, 3:].reshape(len(values), -1, 2)

    home_frames, home_xy = side(home_path)
    away_frames, away_xy = side(away_path)
    if not np.array_equal(home_frames, away_frames):
        raise ValueError("home and away tracking files cover different frames")
    n_frames = int(home_frames.max())
    rows = home_frames - 1
    people = np.concatenate([home_xy[:, :-1], away_xy[:, :-1]], axis=1)
    players = np.full((n_frames, people.shape[1], 2), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players[rows], ball[rows] = people, home_xy[:, -1]
    team = np.array([HOME] * (home_xy.shape[1] - 1) + [AWAY] * (away_xy.shape[1] - 1),
                    dtype=np.int8)
    scale = np.array(PITCH, dtype=np.float32)
    return Tracking(ball * scale, players * scale, team), n_frames


@dataclass(frozen=True)
class Event:
    index: int
    kind: str  # upper case, e.g. "PASS"
    subtypes: frozenset  # upper case names
    side: int  # official H or A code
    period: int
    start: int  # frame, counting from 1 as the files do
    end: int


def read_json_events(path, home_name="Team A"):
    """Game 3's JSON events."""
    with open(path) as handle:
        raw = json.load(handle)["data"]
    events = []
    for e in raw:
        items = e.get("subtypes")
        items = items if isinstance(items, list) else [items] if items else []
        events.append(Event(e["index"], e["type"]["name"], frozenset(i["name"] for i in items if i),
                            H if e["team"]["name"] == home_name else A, e["period"],
                            e["start"]["frame"], e["end"]["frame"]))
    return events


def read_csv_events(path):
    """Games 1-2's events: Team is Home or Away, Subtype joins its names with '-'."""
    table = pd.read_csv(path, dtype={"Subtype": "object"})
    columns = zip(table["Team"].tolist(), table["Type"].tolist(), table["Subtype"].tolist(),
                  table["Period"].tolist(), table["Start Frame"].tolist(),
                  table["End Frame"].tolist(), strict=True)
    return [Event(i, str(kind), frozenset(str(sub).split("-")) if isinstance(sub, str)
                  else frozenset(), H if team == "Home" else A, int(period), int(first),
                  int(last))
            for i, (team, kind, sub, period, first, last) in enumerate(columns)]


def possession_truth(events, n_frames, home_name="Team A"):
    """(codes, runs): approximate H/A/D per frame and the half-open playing stretch of each half.

    `events` is a list of Event, or a path to game 3's JSON or games 1-2's CSV events.
    """
    if not isinstance(events, list):
        path = Path(events)
        events = read_csv_events(path) if path.suffix == ".csv" else read_json_events(path, home_name)
    codes = np.full(n_frames, X, dtype=np.int8)
    runs = []
    for period in sorted({e.period for e in events}):
        half = sorted((e for e in events if e.period == period), key=lambda e: (e.start, e.index))
        restarts = [e for e in half if e.kind == "SET PIECE"
                    or (e.kind == "PASS" and "GOAL KICK" in e.subtypes)]
        start = restarts[0].start - 1  # the kickoff, as a 0-based video frame
        stop = max(e.end for e in half)  # last event's end, exclusive
        runs.append((start, stop))
        for e in half:  # who has the ball, event by event
            if e.kind in CONTROL:
                codes[e.start - 1:stop] = e.side
        for restart in restarts[1:]:  # dead from what stopped play to the restart
            before = [e for e in half if e.start < restart.start]
            dead_from = max(e.end for e in before[-1:]) if before else start
            codes[dead_from - 1:restart.start - 1] = D
    return codes, tuple(runs)


SAMPLE = Path(__file__).resolve().parents[2] / "external" / "metrica-sample"


def load_sample_game(number, root=SAMPLE):
    """(Tracking, truth, runs) for Metrica sample game 1, 2 or 3."""
    folder = Path(root) / f"Sample_Game_{number}"
    stem = folder / f"Sample_Game_{number}_"
    if number == 3:
        tracking, n_frames = read_sample_tracking(Path(f"{stem}metadata.xml"),
                                                  Path(f"{stem}tracking.txt"))
        truth, runs = possession_truth(Path(f"{stem}events.json"), n_frames)
    else:
        tracking, n_frames = read_csv_tracking(Path(f"{stem}RawTrackingData_Home_Team.csv"),
                                               Path(f"{stem}RawTrackingData_Away_Team.csv"))
        truth, runs = possession_truth(Path(f"{stem}RawEventsData.csv"), n_frames)
    return tracking, truth, runs


def hide_ball_in_bursts(tracking, rng, missing=0.41, mean_hidden_seconds=1.5, fps=25):
    """A copy whose ball is hidden in bursts over ~`missing` of frames, like the competition's
    automatic tracking, to test the pipeline where it is weakest."""
    ball = tracking.ball.copy()
    mean_hidden = mean_hidden_seconds * fps
    mean_shown = mean_hidden * (1 - missing) / missing
    t, shown = 0, True
    while t < len(ball):
        span = int(rng.geometric(1.0 / (mean_shown if shown else mean_hidden)))
        if not shown:
            ball[t:t + span] = np.nan
        t, shown = t + span, not shown
    return Tracking(ball, tracking.players, tracking.team)


def smooth_ball(tracking, window, noise=0.0, rng=None):
    """A copy whose ball is averaged over a centred `window` of frames and, optionally, jittered
    by Gaussian noise of `noise` metres -- the way the competition's automatic tracking blurs it
    (its 3-frame velocity jumps: median 0.68 m/s, 99th percentile 4.4, against 0.1 and 20 here)."""
    from possession.features import _rolling_mean
    half = window // 2
    ball = np.stack([_rolling_mean(tracking.ball[:, axis], half, half) for axis in (0, 1)], axis=1)
    if noise:
        ball = ball + (rng or np.random.default_rng(0)).normal(0.0, noise, size=ball.shape)
    ball[np.isnan(tracking.ball[:, 0])] = np.nan  # blurring must not invent a ball
    return Tracking(ball.astype(np.float32), tracking.players, tracking.team)
