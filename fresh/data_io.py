"""Every input of the label-free pipeline, loaded in one place.

Allowed sources only: the match's own unlabelled assets (tactical tracking, Smart Tagging,
broadcast tracking and homography), the public 5-minute ground truth, and the public datasets in
external/ (DFL / IDSSE, Metrica sample games). No hand label is read anywhere in fresh/.

Match-derived caches live next to the match data (DATA_DIR/cache/fresh), never in the repository;
public-data caches live in fresh/out/cache, which a local .gitignore keeps untracked.
"""

import os
from pathlib import Path

import config
import numpy as np

from possession.broadcast import live_image
from possession.dfl import MATCHES as DFL_MATCHES
from possession.dfl import load_dfl_match
from possession.official import (
    DATA_DIR,
    GAME,
    GRADED_RUNS,
    KICKOFFS,
    N_FRAMES,
    PITCH,
    asset_paths,
    load_public_gt,
)
from possession.sample_game import PITCH as SAMPLE_PITCH
from possession.sample_game import load_sample_game
from possession.smart_tagging import read_tags
from possession.tracking import load_tracking

FRESH = Path(__file__).resolve().parent
OUT = FRESH / "out"
_train_degrade = os.environ.get("FRESH_TRAIN_DEGRADE")
PUBLIC_CACHE = OUT / (f"cache-v{_train_degrade}" if _train_degrade and _train_degrade != "1"
                      else ("cache" if config.DEGRADE == 1 else f"cache-v{config.DEGRADE}"))
ARTIFACTS = FRESH / "artifacts"  # tracked: the official decoder search, candidates and rankings only (no score)
MATCH_CACHE = DATA_DIR / "cache" / "fresh" / GAME  # one directory per match: no stale features
if config.SMOOTH > 1:  # E9-smooth changes the velocity features: never mix them with raw caches
    PUBLIC_CACHE = PUBLIC_CACHE.with_name(PUBLIC_CACHE.name + f"-s{config.SMOOTH}")
    MATCH_CACHE = MATCH_CACHE.with_name(MATCH_CACHE.name + f"-s{config.SMOOTH}")

__all__ = [
    "ARTIFACTS",
    "DFL_MATCHES",
    "GAME",
    "GRADED_RUNS",
    "KICKOFFS",
    "MATCH_CACHE",
    "N_FRAMES",
    "OUT",
    "PITCH",
    "PUBLIC_CACHE",
    "SAMPLE_PITCH",
    "dfl",
    "load_public_gt",
    "match",
    "sample",
]


def match():
    """(tracking, tags, live) for BAR-ATH: tactical tracking in metres, Smart Tagging instances,
    and per tactical frame whether the broadcast shows the pitch (NaN where it does not cover)."""
    paths = asset_paths()
    cache = DATA_DIR / "cache"
    tracking = load_tracking(paths["atd_meta"], paths["atd_raw"], N_FRAMES, PITCH, cache)
    check_origin(tracking.ball, PITCH)
    tags = read_tags(paths["smart_tagging"])
    live = live_image(tracking, DATA_DIR, cache)
    return tracking, tags, live


def check_origin(ball, pitch):
    """Stop with a clear message unless positions run from a corner, (0, 0) to (length, width):
    every feature assumes that convention, and a centred origin silently ruins the output."""
    centre = np.nanmedian(ball, axis=0)
    for value, size, axis in zip(centre, pitch, "xy", strict=True):
        if not 0.25 * size <= value <= 0.75 * size:
            raise ValueError(f"tracking {axis} median {value:.1f} m is not near the pitch centre "
                             f"{size / 2:.1f} m: expected corner-origin coordinates in metres")


def dfl(match_id):
    """(tracking, truth, runs, pitch) for one DFL match, the league's own flags as truth."""
    return load_dfl_match(match_id)


def sample(number):
    """(tracking, truth, runs, pitch) for Metrica sample game 1-3, truth derived from events."""
    tracking, truth, runs = load_sample_game(number)
    return tracking, truth, runs, SAMPLE_PITCH


def runs_mask(n, runs):
    mask = np.zeros(n, dtype=bool)
    for start, stop in runs:
        mask[start:stop] = True
    return mask
