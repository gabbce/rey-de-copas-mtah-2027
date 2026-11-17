"""The TV feed as dead-ball evidence.

The broadcast cuts away from the pitch -- replays, close-ups, the bench -- mostly while the ball is
dead. Its homography file already says, frame by frame, whether the picture shows the pitch
(`valid`), so no video needs decoding. The homography is an automatic record of
whether the picture shows the pitch.

DATA.md gives no mapping between the two recordings, but within each half the broadcast is the
tactical video shifted by a constant number of frames. `align` finds that shift from the two ball
tracks, which agree to within 1.5 m on about two thirds of the frames both see the ball at the
right shift and on about 1% at any other.
"""

import csv
from pathlib import Path

import numpy as np

from possession.official import GRADED_RUNS, N_FRAMES, PITCH, games
from possession.tracking import load_tracking

AGREE_METRES = 1.5
MIN_COVISIBLE = 200
OFFSETS = (-20000, 60000)  # frames; tactical = broadcast + offset
COARSE_STEP = 25  # frames, both for the offsets tried and the frames compared
REFINE = 50  # frames each side of the coarse best, tried one by one


def read_shows_pitch(path, n_frames):
    """True on the broadcast frames whose homography is valid, i.e. that show the pitch."""
    shown = np.zeros(n_frames, dtype=bool)
    with open(path) as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            if 0 <= frame < n_frames:
                shown[frame] = row["valid"] == "1"
    return shown


def _gaps(tactical_ball, broadcast_ball, frames, offset):
    """Distances between the two balls at this offset, on the frames both see it; None with too
    few such frames to tell."""
    b = frames - offset
    ok = (b >= 0) & (b < len(broadcast_ball))
    t, b = frames[ok], b[ok]
    both = np.isfinite(tactical_ball[t, 0]) & np.isfinite(broadcast_ball[b, 0])
    if both.sum() < MIN_COVISIBLE:
        return None
    gap = tactical_ball[t[both]] - broadcast_ball[b[both]]
    return np.hypot(gap[:, 0], gap[:, 1])


def agreement(tactical_ball, broadcast_ball, frames, offset):
    """Share of co-visible frames where the balls are within AGREE_METRES at this offset."""
    gaps = _gaps(tactical_ball, broadcast_ball, frames, offset)
    return 0.0 if gaps is None else float((gaps < AGREE_METRES).mean())


def median_gap(tactical_ball, broadcast_ball, frames, offset):
    """Median distance between the balls at this offset; infinite with too few frames."""
    gaps = _gaps(tactical_ball, broadcast_ball, frames, offset)
    return np.inf if gaps is None else float(np.median(gaps))


def align(tactical_ball, broadcast_ball, runs=GRADED_RUNS, offsets=OFFSETS):
    """The offset of each run: coarse by agreement every COARSE_STEP frames, then the exact frame
    by the smallest median gap, since neighbouring offsets agree almost as often. Deterministic,
    so the submission stays exact."""
    found = []
    for start, stop in runs:
        coarse = np.arange(start, stop, COARSE_STEP)
        candidates = np.arange(offsets[0], offsets[1], COARSE_STEP)
        shares = [agreement(tactical_ball, broadcast_ball, coarse, o) for o in candidates]
        best = int(candidates[int(np.argmax(shares))])
        fine = np.arange(start, stop, 5)
        found.append(min(range(best - REFINE, best + REFINE + 1),
                         key=lambda o: median_gap(tactical_ball, broadcast_ball, fine, o)))
    return tuple(found)


def shows_pitch_on_tactical(shown, offsets, n_frames=N_FRAMES, runs=GRADED_RUNS):
    """1 where the broadcast shows the pitch at that tactical frame, 0 where it does not, NaN
    outside the runs or beyond the broadcast."""
    out = np.full(n_frames, np.nan)
    for (start, stop), offset in zip(runs, offsets, strict=True):
        t = np.arange(start, stop)
        b = t - offset
        ok = (b >= 0) & (b < len(shown))
        out[t[ok]] = shown[b[ok]]
    return out


def live_image(tactical, data_dir, cache_dir):
    """Per tactical frame, whether the broadcast shows the pitch; None without the broadcast."""
    paths = games.asset_paths(source="broadcast", data_dir=str(data_dir))
    if not all(Path(paths[k]).exists() for k in ("atd_meta", "atd_raw", "homography")):
        return None
    n_frames = games.source_frames("broadcast")
    broadcast = load_tracking(paths["atd_meta"], paths["atd_raw"], n_frames, PITCH, cache_dir)
    offsets = align(tactical.ball, broadcast.ball)
    return shows_pitch_on_tactical(read_shows_pitch(paths["homography"], n_frames), offsets)
