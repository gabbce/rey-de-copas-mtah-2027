"""A second-stage evidence model stacked on the held-out evidence.

For every DFL match m, a gbm reads the first stage's held-out log-posteriors of the other six
matches at offsets of up to +-4 s and their means over +-1 s and +-5 s, and re-estimates each
frame's H / A / D; it is then applied to m's held-out evidence (and to its evidence on another
degradation's inputs when that file exists). Writes lomo-{set}st-{classifier}-1[-testN].npz, read
by run_decoder.py / run_corrector.py with --set {set}st.

    uv run python fresh/run_stack.py --set bend
"""

import argparse
from multiprocessing import Pool

import bench
import classifiers
import data_io
import numpy as np
import public

from possession.official import A, H, X

OFFSETS = (-100, -50, -25, -10, 0, 10, 25, 50, 100)  # frames
MEANS = (25, 125)  # half-widths, frames


def _posterior(loglik):
    z = loglik - loglik.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def rows(loglik):
    """Stacking features of one match's first-stage evidence."""
    p = _posterior(loglik.astype(np.float64))
    n = len(p)
    idx = np.clip(np.arange(n)[:, None] + np.array(OFFSETS)[None, :], 0, n - 1)
    cols = [p[idx].reshape(n, -1)]
    csum = np.vstack([np.zeros((1, 3)), np.cumsum(p, axis=0)])
    for w in MEANS:
        lo = np.clip(np.arange(n) - w, 0, n)
        hi = np.clip(np.arange(n) + w + 1, 0, n)
        cols.append((csum[hi] - csum[lo]) / (hi - lo)[:, None])
    return np.hstack(cols).astype(np.float32)


def mirrored(z):
    """Home and away exchanged: swap the H and A columns of every block of three."""
    out = z.copy()
    out[:, H::3], out[:, A::3] = z[:, A::3], z[:, H::3]
    return out


def fit_stack(lomo, held):
    """The second stage trained on the first stage's held-out evidence of every match but `held`."""
    xs, ys = [], []
    for i, match_id in enumerate(data_io.DFL_MATCHES):
        if i == held:
            continue
        _, truth, _, _ = data_io.dfl(match_id)
        keep = np.flatnonzero(truth != X)[::bench.SUBSAMPLE]
        z, y = rows(lomo[match_id])[keep], truth[keep].astype(np.int64)
        swapped = np.where(y == H, A, np.where(y == A, H, y))
        xs += [z, mirrored(z)]
        ys += [y, swapped]
    return classifiers.fit_gbm(np.concatenate(xs), np.concatenate(ys))


def first_stage(tag):
    """The first stage's held-out evidence file of a stacked tag ("bendst-gbm" -> bend-gbm's)."""
    return dict(np.load(data_io.PUBLIC_CACHE / f"lomo-{tag.replace('st-', '-', 1)}-1.npz"))


def _fold(args):
    held, lomo, test = args
    model = fit_stack(lomo, held)
    match_id = data_io.DFL_MATCHES[held]
    out = {"": model.log_likelihood(rows(lomo[match_id])).astype(np.float32)}
    if test is not None:
        out["test"] = model.log_likelihood(rows(test[match_id])).astype(np.float32)
    return match_id, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="bend")
    args = parser.parse_args()
    tag = bench.evidence_tag(args.set)
    out_tag = bench.evidence_tag(f"{args.set}st")
    src = data_io.PUBLIC_CACHE / f"lomo-{tag}-1.npz"
    lomo = dict(np.load(src))
    suffix = public.test_suffix()
    test = dict(np.load(str(src).replace(".npz", f"{suffix}.npz"))) if suffix else None
    with Pool(bench.WORKERS) as pool:
        results = pool.map(_fold, [(i, lomo, test) for i in range(len(data_io.DFL_MATCHES))])
    np.savez(data_io.PUBLIC_CACHE / f"lomo-{out_tag}-1.npz", **{m: r[""] for m, r in results})
    if suffix:
        np.savez(data_io.PUBLIC_CACHE / f"lomo-{out_tag}-1{suffix}.npz",
                 **{m: r["test"] for m, r in results})
    print(f"stacked {tag} -> {out_tag}")


if __name__ == "__main__":
    main()
