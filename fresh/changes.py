"""Evidence of a hand-over from one team to the other, learned on DFL (label-free), idea 1 of Oct 22.

The possession evidence says which team has the ball on each frame; the decoder then charges a
constant cost for every change of team. This model looks for the change itself -- a pass or a
reception by the other team, a block that turns round -- and says, frame by frame, how likely a
direct hand-over (H to A, or A to H, without a stoppage between) is within WINDOW frames, trained on
DFL's merged truth with the same degraded, mirrored frames and the same columns as the evidence
plus the team context (fresh/teamctx.py). The decoder turns it into a discount on the change cost
at that frame (bench.change_gain), whose weight the decoder search samples (FRESH_CHANGE=1).

    uv run python fresh/changes.py      # the held-out change evidence of each DFL match, cached
"""

import time
from multiprocessing import get_context

import bench
import config
import data_io
import learn
import numpy as np
import public
import teamctx
from run_lomo import FEATURE_SETS

from possession.official import A, H, X

WINDOW = 12  # frames either side of a hand-over: half a second
MERGE = config.MERGE
NAMES = tuple(FEATURE_SETS["bend"]) + teamctx.NAMES
COLUMNS = bench.columns_of(NAMES)


def targets(truth, runs):
    """0: no direct hand-over within WINDOW; 1: one into H (A to H); 2: one into A (H to A)."""
    y = np.zeros(len(truth), dtype=np.int64)
    for start, stop in runs:
        w = truth[start:stop]
        live = np.isin(w, (H, A))
        at = np.flatnonzero((w[1:] != w[:-1]) & live[1:] & live[:-1]) + 1
        for k in at:
            y[start + max(0, k - WINDOW):start + min(len(w), k + WINDOW)] = 1 if w[k] == H else 2
    return y


def training_rows(items):
    xs, ys = [], []
    for _, x, truth, runs in items:
        y = targets(public.merge_short(truth, MERGE), runs)
        keep = np.flatnonzero(truth != X)[::bench.SUBSAMPLE]
        rows, labels = x[keep], y[keep]
        flipped, _ = public.mirror_rows(rows, truth[keep])
        xs += [rows, flipped]
        ys += [labels, np.where(labels == 1, 2, np.where(labels == 2, 1, labels))]
    return np.concatenate(xs)[:, COLUMNS], np.concatenate(ys)


def train(exclude=None):
    """The change model on every DFL match but `exclude` (index), degraded with seed 0 + i."""
    items = [it for i, it in enumerate(bench.dfl_set()) if i != exclude]
    return learn.fit(*training_rows(items), k=3)


def probabilities(model, x):
    """(n, 2) float32: P(a hand-over into H), P(one into A) within WINDOW of each frame."""
    return np.exp(model.log_proba(x[:, COLUMNS])[:, 1:]).astype(np.float32)


def _fold(i):
    match_id, x, truth, runs = bench.dfl_set()[i]
    q = probabilities(train(exclude=i), x)
    y = targets(public.merge_short(truth, MERGE), runs)
    near = y > 0
    return match_id, q, float(q[near].sum(1).mean()), float(q[~near & (truth != X)].sum(1).mean())


def main():
    began = time.time()
    with get_context("fork").Pool(bench.WORKERS) as pool:
        results = pool.map(_fold, range(len(data_io.DFL_MATCHES)))
    np.savez(data_io.PUBLIC_CACHE / "lomo-change.npz", **{m: q for m, q, _, _ in results})
    near = np.mean([r[2] for r in results])
    far = np.mean([r[3] for r in results])
    print(f"held-out change evidence, {len(NAMES)} columns: mean P(hand-over) {near:.3f} near a "
          f"true hand-over, {far:.3f} elsewhere; {time.time() - began:.0f} s")


if __name__ == "__main__":
    main()
