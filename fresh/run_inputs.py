"""Which DFL degradation makes inputs most like the match's? Label-free.

    uv run python fresh/run_inputs.py
"""

import data_io
import numpy as np
import public

from possession.official import FPS, GRADED_RUNS

FROZEN_STEP = 0.01  # metres
FROZEN_RUN = 25  # frames


def _stretches(mask):
    """Lengths (frames) of maximal True stretches."""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def stats(ball, runs):
    """(visible share, frozen share, gap secs, stretch secs, speeds) over the runs."""
    vis_n = frozen_n = total = 0
    gaps, stretches, speeds = [], [], []
    for a, b in runs:
        x = ball[a:b]
        vis = ~np.isnan(x).any(axis=1)
        total += len(x)
        vis_n += vis.sum()
        gaps.append(_stretches(~vis) / FPS)
        stretches.append(_stretches(vis) / FPS)
        both = vis[1:] & vis[:-1]
        step = np.linalg.norm(np.diff(x, axis=0), axis=1)
        speeds.append(step[both] * FPS)
        still = both & (step < FROZEN_STEP)
        long = _stretches(still)
        frozen_n += long[long >= FROZEN_RUN].sum()
    return (vis_n / total, frozen_n / max(vis_n, 1), np.concatenate(gaps),
            np.concatenate(stretches), np.concatenate(speeds))


def ks(a, b):
    grid = np.sort(np.concatenate([a, b]))
    fa = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    fb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.abs(fa - fb).max())


def distances(s, t):
    return [abs(s[0] - t[0]), abs(s[1] - t[1]), ks(s[2], t[2]), ks(s[3], t[3]), ks(s[4], t[4])]


NAMES = ["visible share", "frozen share", "gap lengths KS", "stretch lengths KS", "speed KS"]


def fit():
    """E8: the (window, jitter) of version 3 whose degraded DFL is closest to the match."""
    tracking, _, _ = data_io.match()
    ref = stats(tracking.ball, GRADED_RUNS)
    dfl = [data_io.dfl(m) for m in data_io.DFL_MATCHES]
    best = None
    for window in (21, 11, 5):
        for jitter in (0.004, 0.01, 0.02, 0.03, 0.05):
            public.V3_BLUR_JITTER = (window, jitter)
            rows = [distances(ref, stats(public.degrade(t, runs, j, version=3).ball, runs))
                    for j, (t, _, runs, _) in enumerate(dfl)]
            med = np.median(rows, axis=0)
            print(f"window {window} jitter {jitter}: " + " ".join(f"{d:.4f}" for d in med)
                  + f" sum {med.sum():.4f}")
            if best is None or med.sum() < best[0]:
                best = (med.sum(), window, jitter)
    print(f"chosen: window {best[1]} jitter {best[2]}")


def main():
    tracking, _, _ = data_io.match()
    ref = stats(tracking.ball, GRADED_RUNS)
    print(f"match: visible {ref[0]:.3f} frozen {ref[1]:.3f} gaps {len(ref[2])} stretches {len(ref[3])}")
    med = {}
    for v in (1, 2):
        rows = []
        for j, match_id in enumerate(data_io.DFL_MATCHES):
            t, truth, runs, _ = data_io.dfl(match_id)
            s = stats(public.degrade(t, runs, j, truth=truth, version=v).ball, runs)
            rows.append(distances(ref, s))
            print(f"v{v} {match_id}: visible {s[0]:.3f} frozen {s[1]:.3f} d=" + " ".join(f"{d:.3f}" for d in rows[-1]))
        med[v] = np.median(rows, axis=0)
    wins2 = int((med[2] < med[1]).sum())
    for k, name in enumerate(NAMES):
        print(f"{name}: v1 {med[1][k]:.4f} v2 {med[2][k]:.4f}")
    verdict = "v2 clearly closer" if wins2 >= 4 else "v1 clearly closer" if 5 - wins2 >= 4 else "inconclusive"
    print(f"v2 closer on {wins2}/5: {verdict}")


if __name__ == "__main__":
    import sys
    fit() if "--fit" in sys.argv else main()
