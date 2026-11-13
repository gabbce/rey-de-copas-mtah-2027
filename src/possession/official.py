"""The official scorer and match constants, loaded from the pinned upstream submodule.

Everything we report is computed by Metrica's own `possession_metrics.py`, never by a
reimplementation, so a number here is the number the board would give. File names, frame counts
and the graded halves come from their `games.py` for the same reason.
"""

import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "vendor" / "mtah-2027"
PUBLIC_GT = UPSTREAM / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
DATA_DIR = Path(os.environ.get("MTAH_DATA_DIR") or ROOT / "data")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if not path.exists() or spec is None or spec.loader is None:
        raise ImportError(f"{path} missing: run `git submodule update --init`")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pm = load_module("possession_metrics", UPSTREAM / "evaluation" / "possession_metrics.py")
games = load_module("mtah_games", UPSTREAM / "games.py")

H, A, D, X = pm.H, pm.A, pm.D, pm.X
segments = pm.segments
valid_runs = pm.valid_runs

GAME = games.GAME
N_FRAMES = games.source_frames("tactical", GAME)
FPS = games.GAMES[GAME]["fps"]
PITCH = games.GAMES[GAME]["pitch"]  # metres, (length, width)
# The graded halves as half-open [start, stop) video-frame runs, the convention valid_runs uses.
GRADED_RUNS = tuple((first, last + 1) for _, first, last in games.half_video_blocks(GAME))


def _anchor_frame(anchor):
    """games.py kickoff anchors, (minute, second, frame) at its ANCHOR_FPS, as a video frame."""
    return (anchor["minute"] * 60 + anchor["second"]) * games.ANCHOR_FPS + anchor["frame"]


# The video frames where each half kicks off, from the official repository's anchors ("the ball is
# struck on this frame"). The second half is graded from 2,934 frames before its kickoff, and the
# ball is dead until the kickoff.
KICKOFFS = (_anchor_frame(games.GAMES[GAME]["kickoff"]),
            _anchor_frame(games.GAMES[GAME]["kickoff_2nd"]))


def asset_paths(data_dir=DATA_DIR):
    """Every tactical-source file of the match, keyed by asset name."""
    return {name: Path(path) for name, path in games.asset_paths(GAME, str(data_dir)).items()}


def load_public_gt():
    """The 5-minute public ground truth as int8 codes, full length (199,319 frames)."""
    return pm.label_codes(pm.read_label_csv(PUBLIC_GT))


def score(gt, pred):
    """The two board numbers plus the possession counts behind them."""
    result = pm.evaluate(gt, pred)
    return {
        "spell_f1": result["spell_f1_mean"],
        "macro_f1": result["macro_f1"],
        "n_pred": result["n_pred_spells"],
        "n_gt": result["n_gt_spells"],
    }
