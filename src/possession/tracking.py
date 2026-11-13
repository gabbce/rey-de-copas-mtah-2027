"""Read Metrica's FIFA/EPTS tracking export into arrays, in metres.

Raw line: "<frame_id>:<x,y,speed>;<x,y,speed>;...:<ball>", NaN for a missing observation, one line
per video frame, frame_id counting from 1 (DATA.md). Slots follow the order of the <Player>
entries in the metadata XML, and the first <Team> is home -- the convention Metrica's own baseline
reads the file with. Coordinates arrive normalised to 0-1 and leave scaled to the pitch.
"""

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HOME, AWAY = 0, 1
PARSER_VERSION = 1  # bump whenever read_tracking's output changes for the same input


@dataclass(frozen=True)
class Tracking:
    ball: np.ndarray  # (n_frames, 2) metres, NaN where the ball is not seen
    players: np.ndarray  # (n_frames, n_slots, 2) metres, NaN where the slot is empty
    team: np.ndarray  # (n_slots,) HOME or AWAY


def read_teams(meta_path):
    """HOME or AWAY for every track slot, in slot order."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError(f"{meta_path}: expected 2 teams, found {len(team_ids)}")
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError(f"{meta_path}: no Player entries")
    return np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8)


def _xy(group):
    return group.split(",", 2)[:2]


def read_tracking(meta_path, raw_path, n_frames, pitch):
    """Every frame of the raw file as arrays. Frames the file does not mention stay NaN."""
    team = read_teams(meta_path)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, len(team), 2), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1
            if not 0 <= row < n_frames:
                continue
            groups = parts[1].split(";")[: len(team)]
            players[row, : len(groups)] = np.array([_xy(g) for g in groups], dtype=np.float32)
            ball[row] = np.array(_xy(parts[2]), dtype=np.float32)
    scale = np.array(pitch, dtype=np.float32)
    return Tracking(ball * scale, players * scale, team)


def _cache_key(meta_path, raw_path, n_frames, pitch):
    """Changes whenever anything read_tracking depends on does: both files, the frame count,
    the pitch, or the parser itself."""
    raw = Path(raw_path).stat()
    digest = hashlib.sha256(Path(meta_path).read_bytes())
    digest.update(f"{raw.st_size}:{raw.st_mtime_ns}:{n_frames}:{tuple(pitch)}:"
                  f"{PARSER_VERSION}".encode())
    return digest.hexdigest()[:16]


def load_tracking(meta_path, raw_path, n_frames, pitch, cache_dir):
    """read_tracking, cached as .npz next to the data so the slow parse runs once."""
    raw_path = Path(raw_path)
    key = _cache_key(meta_path, raw_path, n_frames, pitch)
    cache = Path(cache_dir) / f"{raw_path.stem}-{key}.npz"
    if cache.exists():
        with np.load(cache) as saved:
            return Tracking(saved["ball"], saved["players"], saved["team"])
    tracking = read_tracking(meta_path, raw_path, n_frames, pitch)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, ball=tracking.ball, players=tracking.players, team=tracking.team)
    return tracking
