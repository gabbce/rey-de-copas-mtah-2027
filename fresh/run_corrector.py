"""Experiment: does a second stage that corrects the decoder's segments help? (fresh/corrector.py)

Nested like run_nested.py: for each DFL match m, the corrector and its threshold are learned on the
other six, each decoded by the median of the ten best candidates ranked without that match and
without m (durations fitted the same way), and m is scored decoded by the usual nested pick and
then corrected. No match is ever scored by anything that saw it. `--check gap` repeats it on DFL
with the ball hidden like the match's (variant B, run_simgap.py); `--check public` fits it on all
seven and reports the public window and the match's counts.

    uv run python fresh/run_corrector.py --set bend [--check gap|public]
"""

import argparse
import json
from multiprocessing import get_context
from pathlib import Path

import adapt
import bench
import changes
import corrector
import data_io
import feats
import numpy as np
import pipeline
import public
from corrector import (
    STATE,
    correct,
    decode_all,
    dfl_data,
    fit,
    threshold,
    train_sets_without,
)
from run_adapt import median_params, to_decode
from run_lomo import FEATURE_SETS
from run_simgap import hide_by_state, match_targets, top_candidates

from possession.official import FPS, GRADED_RUNS


def _gap_fold(args):
    """Variant B on DFL match i with its ball hidden like the match's (run_simgap.py): the codes,
    the scaled evidence without channels (what the corrector reads), the features, runs, target."""
    i, p_dead, p_live, columns, tag = args
    c = median_params(top_candidates(tag, i))
    params = to_decode(c)
    durations = bench.dfl_durations(exclude=i) if bench.DUR else None
    match_id = data_io.DFL_MATCHES[i]
    tracking, truth, runs, pitch = data_io.dfl(match_id)
    degraded = hide_by_state(tracking, truth, runs, np.random.default_rng(700 + i), p_dead, p_live)
    x, sign = feats.compute(degraded, runs, FPS, pitch)
    x = bench.with_bends(np.hstack([x, bench.dfl_long(match_id),
                                    bench.dfl_team(match_id, None, sign, degraded=degraded)]))
    x = np.hstack([x, bench.dfl_touch(match_id, None, degraded=degraded),
                   bench.dfl_shape(match_id, None, sign, degraded=degraded),
                   bench.dfl_ctrl(match_id, None, degraded=degraded),
                   bench.dfl_setp(match_id, None, degraded=degraded)])
    dfl_ll = bench.train_seeds(columns, (0,), exclude=i).log_likelihood(x[:, columns])
    if tag.split("-")[0].endswith("st"):  # E6: the second stage on top, as in run_stack.py
        import run_stack
        dfl_ll = run_stack.fit_stack(run_stack.first_stage(tag), i).log_likelihood(
            run_stack.rows(dfl_ll))
    seen = np.isfinite(degraded.ball[:, 0])
    q = changes.probabilities(changes.train(exclude=i), x) if bench.CHANGE else None
    evidence, _, _ = adapt.em(dfl_ll * c["scale"], [adapt.gap_channel(seen)], params, c["scale"],
                              runs=runs)
    codes = bench.decode_runs(evidence, runs, params, tables=bench.dur_tables(durations, c),
                              gain=bench.gains(seen, q, c))
    return (i, codes, dfl_ll * c["scale"], x[:, :len(feats.NAMES)].astype(np.float64), runs,
            public.merge_short(truth, 3.0))


def check_gap(codes, data, columns):
    if (data_io.MATCH_CACHE / "gap-targets-v1.json").exists():
        p_dead, p_live = public.gap_targets()  # the same targets for every degradation version
    else:
        c = median_params(top_candidates("nogap", dur=False))
        p_dead, p_live = match_targets(to_decode(c), c["scale"])
    n = len(data_io.DFL_MATCHES)
    # spawn, not fork: the parent has run multithreaded linear algebra by now (match_targets), and
    # a forked child can deadlock on its locks
    with get_context("spawn").Pool(bench.WORKERS) as pool:
        folds = pool.map(_gap_fold, [(i, p_dead, p_live, columns, STATE["tag"]) for i in range(n)])
    before, after = [], []
    for i, b_codes, scores, x, runs, target in folds:
        train_sets = train_sets_without(codes, data, i)
        fixed = correct(fit(train_sets), threshold(train_sets), b_codes, scores, x, runs)
        before.append(bench.spell(target, b_codes))
        after.append(bench.spell(target, fixed))
        print(f"  gap bench match {i}: B {before[-1]:.3f} -> corrected {after[-1]:.3f}", flush=True)
    wins = sum(a > b for a, b in zip(after, before))
    print(f"gap bench B (nested): {np.mean(before):.3f} -> corrected {np.mean(after):.3f}; "
          f"wins {wins} of {n}")


def check_public(codes, data):
    train_sets = train_sets_without(codes, data, None)
    model, best = fit(train_sets), threshold(train_sets)
    body = pipeline.load(pipeline.FINAL) or pipeline.fit(pipeline.FINAL)
    x, _ = bench.match_frames()
    loglik = pipeline._loglik(body, x)
    c = body["members"][0]
    b_codes = bench.decode_runs(
        pipeline._evidence(loglik, c, pipeline._channels(pipeline.FINAL), body["tables"]),
        GRADED_RUNS, to_decode(c), tables=bench.dur_tables(body.get("durations"), c),
        gain=bench.gains(x[:, 0] > 0.5, pipeline.change_evidence(body, x), c))
    fixed = correct(model, best, b_codes, loglik * c["scale"], x.astype(np.float64), GRADED_RUNS)
    print(f"corrector on all seven DFL matches: threshold {best}")
    for name, out in (("B", bench.dead_before_kickoff(b_codes)),
                      ("B + corrector", bench.dead_before_kickoff(fixed))):
        spell, macro, n_pred, n_gt = bench.public_score(out)
        low, high = bench.bootstrap_public(out)
        n_poss, n_dead, live, graded = bench.counts(out)
        print(f"{name}: public spellF1 {spell:.3f} [{low:.3f}, {high:.3f}] macroF1 {macro:.3f}, "
              f"possessions {n_pred}/{n_gt}; match {n_poss} possessions, {n_dead} stoppages, "
              f"live {100 * live / graded:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="bend")
    parser.add_argument("--merge", type=float, default=3.0)
    parser.add_argument("--threshold", type=float,
                        help="a fixed corrector threshold instead of the leave-one-out choice (1.01: off)")
    parser.add_argument("--score-merge", type=float,
                        help="score nested DFL against truth merged at this many seconds (default --merge)")
    parser.add_argument("--raw", action="store_true",
                        help="also score the same decodes against DFL's raw (unmerged) truth")
    parser.add_argument("--refs", default="",
                        help="comma list of merge seconds (0 = raw) to score the corrected decodes against")
    parser.add_argument("--json", type=Path, help="write per-match scores against --refs here")
    parser.add_argument("--check", choices=("gap", "public"),
                        help="instead of the nested DFL score: the gap bench, or the public window")
    args = parser.parse_args()
    tag = bench.evidence_tag(args.set)
    if args.threshold is not None:
        corrector.THRESHOLDS = (args.threshold,)
    n = len(data_io.DFL_MATCHES)
    jobs = [(j, None) for j in range(n)] + [(j, m) for m in range(n) for j in range(n) if j != m]
    codes = decode_all(tag, args.merge, jobs)
    data = dfl_data(args.merge)
    if args.check and public.test_suffix():
        raise SystemExit("test_degradation is for the nested DFL score only")
    if args.check == "gap":
        return check_gap(codes, data, bench.columns_of(FEATURE_SETS[args.set]))
    if args.check == "public":
        return check_public(codes, data)

    refs = [float(r) for r in args.refs.split(",") if r]
    by_ref = {r: [] for r in refs}
    before, after, chosen, poss, raw_b, raw_a = [], [], [], [], [], []
    for m in range(n):
        train_sets = train_sets_without(codes, data, m)
        best = threshold(train_sets)
        c, s = codes[(m, None)]
        target, runs, x = data[m]
        if public.test_suffix():
            with public.degradation(public.test_version()):
                x = public.dfl_frames(data_io.DFL_MATCHES[m], m)[0][:, :len(feats.NAMES)].astype(np.float64)
        if args.score_merge is not None:
            target = public.merge_short(data_io.dfl(data_io.DFL_MATCHES[m])[1], args.score_merge)
        fixed = correct(fit(train_sets), best, c, s, x, runs)
        wrong = np.mean(np.concatenate([t[3] for t in train_sets]))
        before.append(bench.spell(target, c))
        after.append(bench.spell(target, fixed))
        chosen.append(best)
        raw_truth = data_io.dfl(data_io.DFL_MATCHES[m])[1]
        for r in refs:
            by_ref[r].append(bench.spell(public.merge_short(raw_truth, r) if r else raw_truth, fixed))
        if args.raw:
            raw = data_io.dfl(data_io.DFL_MATCHES[m])[1]
            raw_b.append(bench.spell(raw, c))
            raw_a.append(bench.spell(raw, fixed))
            print(f"  match {m} raw truth: {raw_b[-1]:.3f} -> {raw_a[-1]:.3f}; raw possessions "
                  f"{bench.counts(raw, runs)[0]}", flush=True)
        poss.append([bench.counts(k, runs)[0] for k in (target, c, fixed)])
        print(f"  match {m}: wrong segments in training {wrong:.1%}; threshold {best}; "
              f"{before[-1]:.3f} -> {after[-1]:.3f}", flush=True)
    wins = sum(a > b for a, b in zip(after, before))
    ties = sum(a == b for a, b in zip(after, before))
    print(f"nested DFL (merged truth): median10 {np.mean(before):.3f} -> corrected {np.mean(after):.3f}; "
          f"wins {wins}, ties {ties}, losses {n - wins - ties}; thresholds {chosen}")
    if args.raw:
        print(f"nested DFL (raw truth): median10 {np.mean(raw_b):.3f} -> corrected {np.mean(raw_a):.3f}")
    for r in refs:
        print(f"nested DFL, corrected, against truth merged at {r:g} s: {np.mean(by_ref[r]):.4f}")
    if args.json:
        args.json.write_text(json.dumps({"merge": args.merge, "refs": {f"{r:g}": by_ref[r] for r in refs},
                                         "possessions": np.sum(poss, axis=0).tolist()}) + "\n")
    total = np.sum(poss, axis=0)
    print(f"possessions over the seven matches: merged truth {total[0]}, decoded {total[1]}, "
          f"corrected {total[2]}")


if __name__ == "__main__":
    main()
