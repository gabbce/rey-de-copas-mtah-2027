"""Metrica's Smart Tagging as extra per-frame evidence for the decoder.

A SportsCode XML of automatic phases and set pieces, timed in seconds from the first frame of the
tactical video (DATA.md: seconds x 25 is a video frame). It is inferred, not ground truth, and its
intervals are phases rather than possessions, so nothing here copies an interval into a label: each
tag only adds a little evidence per frame, and the tracking still decides where a change happens.

- Attacking phases say the tagged team has the ball; defending phases say the other one does.
- A set piece is a restart: dead ball just before it, the tagged team just after it. Throw-ins are
  tagged N/A, so after those only the tracking speaks. The interval is long (median 35 s) and
  starts in the dead ball, with the tagged side first on the ball a few seconds after its start;
  moving the team evidence there did not score better than leaving it after the end
  (docs/history.md §33).
- A first-half or second-half kickoff means everything graded before it is dead ball.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from possession.official import FPS, GRADED_RUNS, A, D, H

# The Team tag names clubs; DATA.md spells the away side "Atletic Club" in this export.
TEAM_NAMES = {"barcelona": H, "atletic club": A, "athletic club": A}

ATTACKING = {"build up", "progression", "creating chances", "attacking transition",
             "ball in final third", "ball in the box", "players in the box", "long ball"}
DEFENDING = {"defending in defensive third", "defending in middle third",
             "defending in attacking third", "defensive transition"}
SET_PIECES = "set pieces"
KICKOFF = "kick off (start)"


@dataclass(frozen=True)
class Tag:
    code: str  # lower case
    start: int  # first video frame
    stop: int  # first frame after
    team: int | None  # H, A, or None for N/A and unknown names
    kind: str  # the Type tag, lower case, "" when absent


@dataclass(frozen=True)
class TagParams:
    phase_evidence: float = 0.05  # per frame, toward the team a phase implies
    restart_dead: float = 0.5  # per frame, toward D in the seconds before a set piece
    restart_dead_seconds: float = 2.0
    restart_team: float = 0.3  # per frame, toward the restarting team after a set piece
    restart_team_seconds: float = 2.0
    kickoff_dead: float = 2.0  # per frame, toward D from the start of a half to its kickoff


DEFAULT_TAGS = TagParams()


def read_tags(path, fps=FPS):
    """Every instance of the XML, in file order."""
    tags = []
    for instance in ET.parse(path).iter("instance"):
        labels = {(label.findtext("group") or "").strip().lower():
                  (label.findtext("text") or "").strip().lower()
                  for label in instance.findall("label")}
        tags.append(Tag(
            code=(instance.findtext("code") or "").strip().lower(),
            start=round(float(instance.findtext("start") or 0) * fps),
            stop=round(float(instance.findtext("end") or 0) * fps),
            team=TEAM_NAMES.get(labels.get("team", "")),
            kind=labels.get("type", ""),
        ))
    return tags


# Channels of tag_channels(), one count per frame each.
PHASE_H, PHASE_A, DEAD_BEFORE, RESTART_H, RESTART_A, BEFORE_KICKOFF = range(6)
N_CHANNELS = 6


def _add(channels, start, stop, column):
    start, stop = max(start, 0), min(stop, len(channels))
    if start < stop:
        channels[start:stop, column] += 1.0


def tag_channels(tags, n_frames, halves=GRADED_RUNS, dead_frames=50, team_frames=50):
    """(n_frames, N_CHANNELS) counts of what the tags claim about each frame.

    Overlapping tags add up. `halves` are the graded halves, not whatever stretch is being
    decoded: a kickoff's dead ball runs from the start of its half.
    """
    channels = np.zeros((n_frames, N_CHANNELS))
    side = {H: (PHASE_H, RESTART_H), A: (PHASE_A, RESTART_A)}
    for tag in tags:
        if tag.code in ATTACKING and tag.team is not None:
            _add(channels, tag.start, tag.stop, side[tag.team][0])
        elif tag.code in DEFENDING and tag.team is not None:
            _add(channels, tag.start, tag.stop, side[A if tag.team == H else H][0])
        elif tag.code == SET_PIECES:
            if tag.kind == KICKOFF:
                half_start = next((a for a, b in halves if a <= tag.start < b), None)
                if half_start is not None:
                    _add(channels, half_start, tag.start, BEFORE_KICKOFF)
            _add(channels, tag.start - dead_frames, tag.start, DEAD_BEFORE)
            if tag.team is not None:
                _add(channels, tag.stop, tag.stop + team_frames, side[tag.team][1])
    return channels


def tag_evidence(tags, n_frames, params=DEFAULT_TAGS, halves=GRADED_RUNS, fps=FPS):
    """(n_frames, 3) scores to add to the tracking evidence, columns by official H, A, D."""
    c = tag_channels(tags, n_frames, halves, round(params.restart_dead_seconds * fps),
                     round(params.restart_team_seconds * fps))
    scores = np.zeros((n_frames, 3))
    scores[:, H] = params.phase_evidence * c[:, PHASE_H] + params.restart_team * c[:, RESTART_H]
    scores[:, A] = params.phase_evidence * c[:, PHASE_A] + params.restart_team * c[:, RESTART_A]
    scores[:, D] = (params.restart_dead * c[:, DEAD_BEFORE]
                    + params.kickoff_dead * c[:, BEFORE_KICKOFF])
    return scores
