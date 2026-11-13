"""Label-free facts that shape the design: how the match's tracking looks, and how often the
public truths change state. Aggregates only.

    uv run python fresh/stats.py
"""

import itertools

import data_io
import numpy as np

from possession.official import FPS, A, D, H, X, segments, valid_runs


def gap_runs(missing):
    edges = np.flatnonzero(np.diff(np.concatenate([[0], missing.astype(np.int8), [0]])))
    return edges[1::2] - edges[::2]


def velocity_change(ball, half=1):
    before = (ball[half:-half] - ball[:-2 * half]) * FPS / half
    after = (ball[2 * half:] - ball[half:-half]) * FPS / half
    jump = np.hypot(*(after - before).T)
    return jump[np.isfinite(jump)]


def tracking_facts(name, tracking, runs):
    mask = data_io.runs_mask(len(tracking.ball), runs)
    seen = np.isfinite(tracking.ball[:, 0])
    gaps = np.concatenate([gap_runs(~seen[a:b]) for a, b in runs]) / FPS
    if not len(gaps):
        gaps = np.zeros(1)
    weights = gaps / max(gaps.sum(), 1e-9)
    order = np.argsort(gaps)
    cum = np.cumsum(weights[order])
    tracked = np.isfinite(tracking.players[..., 0]).sum(axis=1)[mask]
    jumps = velocity_change(np.where(mask[:, None], tracking.ball, np.nan), half=1)
    print(f"{name}: ball seen {seen[mask].mean():.3f}; gaps n={len(gaps)} median {np.median(gaps):.2f}s"
          f" mean {gaps.mean():.2f}s p90 {np.quantile(gaps, 0.9):.1f}s; missing frames in gaps "
          f">1s {weights[gaps > 1].sum():.2f} >3s {weights[gaps > 3].sum():.2f} "
          f">6s {weights[gaps > 6].sum():.2f} >15s {weights[gaps > 15].sum():.2f}; half the "
          f"missing frames in gaps over {gaps[order][min(np.searchsorted(cum, 0.5), len(gaps) - 1)]:.1f}s")
    print(f"   tracked players mean {tracked.mean():.1f}; ball 1-frame velocity change "
          f"median {np.median(jumps):.2f} p99 {np.quantile(jumps, 0.99):.1f} m/s")


def truth_facts(name, truth):
    runs = valid_runs(truth)
    segs = segments(truth, runs)
    poss = np.array([b - a for a, b, k in segs if k in (H, A)]) / FPS
    dead = np.array([b - a for a, b, k in segs if k == D]) / FPS
    graded = int((truth != X).sum())
    live = int(np.isin(truth, (H, A)).sum())
    changes = 0
    for (_, b1, k1), (a2, _, k2) in itertools.pairwise(segs):
        if b1 == a2 and {k1, k2} == {H, A}:
            changes += 1
    print(f"{name}: {graded / FPS / 60:.1f} graded min, dead share {1 - live / graded:.2f}; "
          f"{len(poss)} possessions = {len(poss) / (live / FPS / 60):.2f} per live min, "
          f"{len(poss) / (graded / FPS / 60):.2f} per graded min; median {np.median(poss):.1f}s; "
          f"{len(dead)} stoppages = {len(dead) / (graded / FPS / 60):.2f} per min, median "
          f"{np.median(dead) if len(dead) else 0:.1f}s; live team changes {changes}")


def main():
    truth_facts("public", data_io.load_public_gt())
    tracking, _, live = data_io.match()
    tracking_facts("match tactical", tracking, data_io.GRADED_RUNS)
    covered = np.isfinite(live)
    print(f"   broadcast covers {covered[data_io.runs_mask(len(live), data_io.GRADED_RUNS)].mean():.3f}"
          f" of graded frames, shows the pitch on {np.nanmean(live):.3f} of those")
    for match_id in data_io.DFL_MATCHES:
        tracking, truth, runs, _ = data_io.dfl(match_id)
        truth_facts(f"dfl {match_id}", truth)
        if match_id == data_io.DFL_MATCHES[0]:
            tracking_facts(f"dfl {match_id} tracking", tracking, runs)
    for number in (1, 2, 3):
        tracking, truth, runs, _ = data_io.sample(number)
        truth_facts(f"metrica sample {number}", truth)
        tracking_facts(f"metrica sample {number} tracking", tracking, runs)


if __name__ == "__main__":
    main()
