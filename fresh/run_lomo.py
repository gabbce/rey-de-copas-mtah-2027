"""Experiment A, step 1: the learned evidence, judged on public data only.

Each DFL match is predicted by a logistic regression trained on the other six (mirrored, degraded),
and its held-out evidence is cached for the decoder search (fresh/run_decoder.py). Prints frame
accuracy per match and, for reference, the evidence decoded with fixed costs.

    uv run python fresh/run_lomo.py [--set NAME] [--seeds 0,10,20]

With several seeds, each fold averages one model per DFL degradation seed; the held-out match keeps
its seed-0 degradation, so every variant is scored on the same frames.
"""

import argparse
import time
from multiprocessing import Pool

import bench
import config
import data_io
import longctx
import numpy as np
import public

from possession.official import X

FEATURE_SETS = {
    "all": None,
    # without the ball's gaps: what the match's own data must teach (fresh/adapt.py)
    "nogap": [n for n in bench.feats.NAMES if not n.startswith("gap_len")
              and n not in ("gap_since", "gap_until", "seen_near")],
    # nogap plus the player-only long-context columns (fresh/longctx.py, Lang et al. 2022)
    "long": [n for n in bench.feats.NAMES if not n.startswith("gap_len")
             and n not in ("gap_since", "gap_until", "seen_near")] + list(longctx.NAMES),
    # long plus team evidence around changes (fresh/teamctx.py)
    "team": [n for n in bench.feats.NAMES if not n.startswith("gap_len")
             and n not in ("gap_since", "gap_until", "seen_near")] + list(longctx.NAMES)
    + list(bench.teamctx.NAMES),
    # long plus its even bends x^2 / (1 + |x|) (bench.BEND_NAMES)
    "bend": [n for n in bench.feats.NAMES if not n.startswith("gap_len")
             and n not in ("gap_since", "gap_until", "seen_near")] + list(longctx.NAMES)
    + list(bench.BEND_NAMES),
    # bend plus the touch detector's columns (fresh/touch.py)
    "touch": [n for n in bench.feats.NAMES if not n.startswith("gap_len")
              and n not in ("gap_since", "gap_until", "seen_near")] + list(longctx.NAMES)
    + list(bench.BEND_NAMES) + list(bench.touch.NAMES),
    # bend plus team shape from the players only (fresh/shape.py)
    "shape": [n for n in bench.feats.NAMES if not n.startswith("gap_len")
              and n not in ("gap_since", "gap_until", "seen_near")] + list(longctx.NAMES)
    + list(bench.BEND_NAMES) + list(bench.shape.NAMES),
    # bend plus projected ball, visibility and control history (fresh/ctrl.py)
    "ctrl": [n for n in bench.feats.NAMES if not n.startswith("gap_len")
             and n not in ("gap_since", "gap_until", "seen_near")] + list(longctx.NAMES)
    + list(bench.BEND_NAMES) + list(bench.ctrl.NAMES),
    # bend plus the set-piece geometry of the players (fresh/setpiece.py)
    "setp": [n for n in bench.feats.NAMES if not n.startswith("gap_len")
             and n not in ("gap_since", "gap_until", "seen_near")] + list(longctx.NAMES)
    + list(bench.BEND_NAMES) + list(bench.setpiece.NAMES),
    # the same columns, the evidence restacked by run_stack.py
    "bendst": [n for n in bench.feats.NAMES if not n.startswith("gap_len")
               and n not in ("gap_since", "gap_until", "seen_near")] + list(longctx.NAMES)
    + list(bench.BEND_NAMES),
    "ctrlst": [n for n in bench.feats.NAMES if not n.startswith("gap_len")
               and n not in ("gap_since", "gap_until", "seen_near")] + list(longctx.NAMES)
    + list(bench.BEND_NAMES) + list(bench.ctrl.NAMES),
    "ball_only": ["seen", "adv", "near", "adv_near", "dmin", "ball_speed", "fast",
                  "adv_near_fast", "ball_out", "line_dist", "press"],
}


def _features(args):
    i, seed = args
    match_id = data_io.DFL_MATCHES[i]
    public.dfl_frames(match_id, seed + i)
    return match_id


def _fold(args):
    held, columns, l2, seeds = args
    model = bench.train_seeds(columns, seeds, exclude=held, l2=l2)
    match_id, x, truth, _ = bench.dfl_set()[held]
    xs = x if columns is None else x[:, columns]
    loglik = model.log_likelihood(xs)
    if public.test_version() != config.DEGRADE:
        with public.degradation(public.test_version()):
            match_id, x, truth, _ = bench.dfl_set()[held]
        xs = x if columns is None else x[:, columns]
        loglik = model.log_likelihood(xs)
    graded = truth != X
    acc = float((model.log_proba(xs).argmax(1)[graded] == truth[graded]).mean())
    return match_id, loglik.astype(np.float32), acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="all", choices=sorted(FEATURE_SETS))
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--seeds", default="0", help="DFL degradation seeds, comma-separated")
    args = parser.parse_args()
    seeds = bench.parse_seeds(args.seeds)
    began = time.time()
    with Pool(bench.WORKERS) as pool:
        pool.map(_features, [(i, s) for s in seeds for i in range(len(data_io.DFL_MATCHES))])
    names = FEATURE_SETS[args.set]
    columns = None if names is None else bench.columns_of(names)
    with Pool(bench.WORKERS) as pool:
        results = pool.map(_fold, [(i, columns, args.l2, seeds)
                                   for i in range(len(data_io.DFL_MATCHES))])
    tag = bench.evidence_tag(args.set, seeds)
    out = data_io.PUBLIC_CACHE / f"lomo-{tag}-{args.l2:g}{public.test_suffix()}.npz"
    np.savez(out, **{match_id: loglik for match_id, loglik, _ in results})
    accs = [acc for _, _, acc in results]
    print(f"evidence {tag} ({'all' if names is None else len(names)} columns), l2 {args.l2:g}: "
          f"held-out frame accuracy mean {np.mean(accs):.3f} "
          f"({' '.join(f'{a:.3f}' for a in accs)}); {time.time() - began:.0f} s")


if __name__ == "__main__":
    main()
