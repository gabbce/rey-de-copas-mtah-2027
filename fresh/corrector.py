"""A second stage that corrects the decoder's segments, learned on DFL (label-free). Adopted Oct 22.

The decoder decides frame by frame under a path model; some of the segments it outputs are wrong
as a whole (a possession that never happened, an invented stoppage). This stage looks at each
decoded segment -- its length, its neighbours, how firmly the evidence backed it, how much of it
the ball was seen -- and learns, on DFL's merged truth, how likely it is to be wrong; a segment
judged wrong is given to its neighbour. Each DFL match it learns from is decoded the way the
nested bench decodes it: held-out evidence, and the median of the ten best decoders ranked without
that match. The threshold is chosen by leave-one-match-out inside the training matches.

The experiment that judged it is fresh/run_corrector.py; fresh/pipeline.py applies it to the match
(FRESH_CORRECT, on by default).
"""

import json
import subprocess
import sys
from multiprocessing import get_context

import bench
import config
import data_io
import feats
import hdecode
import learn
import numpy as np
import public
from run_adapt import load_search, median_params, search_order, to_decode

from possession.official import FPS, A, D, H, segments

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.01)  # 1.01: no correction
SEEN = feats.NAMES.index("seen")
SPEED = feats.NAMES.index("speed_mid")
SLOW = feats.NAMES.index("slow_share")
GAP = feats.NAMES.index("gap_len")
STATE = {}


def durations_without(excluded, merge):
    truths, runs_list = [], []
    for i, match_id in enumerate(data_io.DFL_MATCHES):
        if i in excluded:
            continue
        _, truth, runs, _ = data_io.dfl(match_id)
        truths.append(public.merge_short(truth, bench.durations_merge(merge)))
        runs_list.append(runs)
    return hdecode.fit(truths, runs_list)


def pick(excluded):
    order = search_order(STATE["search"], tuple(excluded), 10)
    return median_params([STATE["candidates"][i] for i in order])


def decoded(args):
    """Codes of DFL match j decoded by the pick (and durations) that saw neither j nor m."""
    j, m = args
    excluded = {j} if m is None else {j, m}
    match_id = data_io.DFL_MATCHES[j]
    _, _, runs, _ = data_io.dfl(match_id)
    c = pick(excluded)
    test = m is None and public.test_suffix()
    loglik = STATE["test_lomo" if test else "lomo"][match_id].astype(np.float64)
    durations = durations_without(excluded, STATE["merge"]) if bench.DUR else None
    if test:
        with public.degradation(public.test_version()):
            seen = public.dfl_frames(match_id, j)[0][:, SEEN] > 0.5
    else:
        seen = public.dfl_frames(match_id, j)[0][:, SEEN] > 0.5
    scores = loglik * c["scale"]
    codes = bench.decode_runs(scores, runs, to_decode(c), tables=bench.dur_tables(durations, c),
                              gain=bench.gains(seen, bench.dfl_changes(match_id), c))
    return j, m, codes, scores


def segment_rows(codes, scores, x, runs):
    """One row per decoded segment: (start, stop, state, feature vector, run index)."""
    out = []
    for r, (start, stop) in enumerate(runs):
        segs = [s for s in segments(codes, [(start, stop)]) if s[2] in (H, A, D)]
        for i, (a, b, k) in enumerate(segs):
            prev = segs[i - 1] if i else None
            nxt = segs[i + 1] if i + 1 < len(segs) else None
            s = scores[a:b]
            others = np.delete(s, k, axis=1).max(axis=1)
            margin = s[:, k] - others
            row = [
                float(k == D),
                np.log1p((b - a) / FPS),
                np.log1p((prev[1] - prev[0]) / FPS) if prev else 0.0,
                np.log1p((nxt[1] - nxt[0]) / FPS) if nxt else 0.0,
                float(prev is not None and prev[2] == D),
                float(nxt is not None and nxt[2] == D),
                float(prev is not None and nxt is not None and prev[2] == nxt[2] != k),
                float(np.clip(margin, -5, 5).mean()),
                float((margin > 0).mean()),
                float(x[a:b, SEEN].mean()),
                float(x[a:b, GAP].mean()),
                float(x[a:b, SPEED].mean()),
                float(x[a:b, SLOW].mean()),
            ]
            out.append((a, b, k, row, r))
    return out


def corrected(codes, rows, p_wrong, threshold):
    """Each segment judged wrong goes to a neighbour: a possession to the longer of its live
    neighbours (or the dead one when it has no live neighbour), a stoppage to the possession before
    it (after it at the start)."""
    out = codes.copy()
    for i, ((a, b, k, _, r), p) in enumerate(zip(rows, p_wrong, strict=True)):
        if p < threshold:
            continue
        prev = rows[i - 1] if i and rows[i - 1][4] == r else None
        nxt = rows[i + 1] if i + 1 < len(rows) and rows[i + 1][4] == r else None
        if k == D:
            new = prev[2] if prev and prev[2] != D else (nxt[2] if nxt and nxt[2] != D else None)
        else:
            live = [r for r in (prev, nxt) if r and r[2] != D]
            pool = live or [r for r in (prev, nxt) if r]
            new = max(pool, key=lambda r: r[1] - r[0])[2] if pool else None
        if new is not None:
            out[a:b] = new
    return out


def labels(rows, truth):
    return np.array([int(np.mean(truth[a:b] == k) < 0.5) for a, b, k, *_ in rows])


def p_wrong(model, rows):
    return np.exp(model.log_proba(np.array([r[3] for r in rows]))[:, 1])


def fit(train_sets):
    x = np.array([r[3] for _, rows, _, _ in train_sets for r in rows])
    y = np.concatenate([y for _, _, _, y in train_sets])
    return learn.fit(x, y, k=2)


def candidate_thresholds():
    """THRESHOLDS at or above config.CORR_MIN (E12 floor; default 0.5 keeps the full grid)."""
    floor = config.CORR_MIN
    cands = tuple(t for t in THRESHOLDS if t + 1e-12 >= floor)
    return cands or (1.01,)


def threshold(train_sets):
    """The threshold with the best mean spellF1 over the training matches, each corrected by a
    corrector that did not see it (leave one out inside the training matches)."""
    cands = candidate_thresholds()
    table = []
    for i, (c, rows, target, _) in enumerate(train_sets):
        model = fit(train_sets[:i] + train_sets[i + 1:])
        p = p_wrong(model, rows)
        table.append([bench.spell(target, corrected(c, rows, p, t)) for t in cands])
    return cands[int(np.argmax(np.mean(table, axis=0)))]


def train_sets_without(codes, data, m):
    """(codes, rows, target, labels) of every DFL match but m, each decoded without itself and m."""
    out = []
    for j in range(len(data_io.DFL_MATCHES)):
        if j == m:
            continue
        c, s = codes[(j, m)]
        target, runs, x = data[j]
        rows = segment_rows(c, s, x, runs)
        out.append((c, rows, target, labels(rows, target)))
    return out


def correct(model, best, codes, scores, x, runs):
    rows = segment_rows(codes, scores, x, runs)
    return corrected(codes, rows, p_wrong(model, rows), best)


def setup(tag, merge):
    """Load the decoder search and the held-out evidence of `tag` (also each worker's initializer)."""
    name = f"decoder-{bench.dtag(tag)}-m{merge:g}-s{config.DECODER_SEED}-all.npz"
    saved = load_search(name)
    lomo = data_io.PUBLIC_CACHE / f"lomo-{tag}-1.npz"
    if not lomo.exists():  # a clean clone: DFL's held-out evidence, deterministic (fresh/run_lomo.py)
        subprocess.run([sys.executable, str(data_io.FRESH / "run_lomo.py"), "--set",
                        tag.removesuffix(f"-{bench.CLASSIFIER}")], check=True)
    STATE.update(search=saved, candidates=[json.loads(c) for c in saved["candidates"]],
                 lomo=dict(np.load(lomo)), merge=merge,
                 test_lomo=(dict(np.load(str(lomo).replace(".npz", f"{public.test_suffix()}.npz")))
                            if public.test_suffix() else None),
                 tag=tag)


def decode_all(tag, merge, jobs):
    """{(j, m): (codes, scaled evidence)} of decoded() over `jobs`. Spawned workers: a forked one
    can deadlock when the parent has already run multithreaded linear algebra."""
    setup(tag, merge)
    with get_context("spawn").Pool(bench.WORKERS, initializer=setup, initargs=(tag, merge)) as pool:
        out = pool.map(decoded, jobs)
    return {(j, m): (c, s) for j, m, c, s in out}


def dfl_data(merge):
    """{j: (merged truth, runs, feats columns)} of the seven DFL matches, degraded with seed j."""
    data = {}
    for j, match_id in enumerate(data_io.DFL_MATCHES):
        _, truth, runs, _ = data_io.dfl(match_id)
        x = public.dfl_frames(match_id, j)[0]
        target = merge if config.CORR_MERGE is None else config.CORR_MERGE
        data[j] = (public.merge_short(truth, target), runs, x[:, :len(feats.NAMES)].astype(np.float64))
    return data


def train(tag, merge):
    """(model, threshold) learned on all seven DFL matches."""
    n = len(data_io.DFL_MATCHES)
    codes = decode_all(tag, merge, [(j, None) for j in range(n)])
    sets = train_sets_without(codes, dfl_data(merge), None)
    return fit(sets), threshold(sets)


def apply(body, codes, scores, x, runs):
    """The corrector stored in an artifact (`body["corrector"]`) applied to decoded codes."""
    model = learn.Logistic.from_json(body["corrector"]["model"])
    return correct(model, body["corrector"]["threshold"], codes, scores,
                   x[:, :len(feats.NAMES)].astype(np.float64), runs)
