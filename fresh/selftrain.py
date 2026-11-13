"""Experiment C: self-training on the match, out of block.

The adapted decoder's posteriors (experiment B) label the frames it is confident about; a logistic
regression on the match's own measurements learns from them, so its team evidence speaks this
tracker's dialect rather than DFL's. Each block of the match is scored by a model that never saw
its pseudo-labels, which keeps the model from simply memorising the decoder's own mistakes, and
every frame also enters mirrored, so no side can be learned.
"""

import learn
import numpy as np
import public


def fit_blocks(x, gamma, runs, columns, blocks=8, threshold=0.9, subsample=3, l2=1.0):
    """([(start, stop)] frame ranges, one model per range trained on every other range's
    confident frames). The ranges split the frames inside `runs` into equal contiguous parts."""
    n = len(x)
    labels = gamma.argmax(1).astype(np.int8)
    confident = gamma.max(1) >= threshold
    inside = np.zeros(n, dtype=bool)
    for start, stop in runs:
        inside[start:stop] = True
    frames = np.flatnonzero(inside)
    edges = np.linspace(0, len(frames), blocks + 1).astype(int)
    ranges = [(int(frames[edges[b]]), int(frames[edges[b + 1] - 1]) + 1) for b in range(blocks)]
    index = np.arange(n)
    models = []
    for start, stop in ranges:
        outside = (index < start) | (index >= stop)
        keep = np.flatnonzero(inside & confident & outside)[::subsample]
        full, ys = x[keep], labels[keep]
        flipped, swapped = public.mirror_rows(full, ys)
        train_x = np.concatenate([full[:, columns], flipped[:, columns]])
        train_y = np.concatenate([ys, swapped]).astype(np.int64)
        models.append(learn.fit(train_x, train_y, l2=l2))
    return ranges, models


def out_of_block(x, gamma, runs, columns, **kw):
    """(n, 3) log-likelihoods, each range predicted by its own out-of-block model; 0 elsewhere."""
    ranges, models = fit_blocks(x, gamma, runs, columns, **kw)
    out = np.zeros((len(x), 3))
    for (start, stop), model in zip(ranges, models, strict=True):
        out[start:stop] = model.log_likelihood(x[start:stop, columns])
    return out
