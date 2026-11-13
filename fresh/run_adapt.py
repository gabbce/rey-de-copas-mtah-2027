"""Experiment B on the match: the DFL evidence (no gap columns) adapted by EM with the match's own
channels, decoded with a decoder chosen on DFL. Prints the public window, the whole-match counts
and the learned tables (aggregate parameters of the model, no frame).

    uv run python fresh/run_adapt.py --pick median10
"""

import argparse
import json

import adapt
import bench
import config
import data_io
import numpy as np
from run_lomo import FEATURE_SETS

from possession.decode import DecodeParams

USE = {"none": (), "gap": ("gap",), "gap_bc": ("gap", "broadcast"),
       "all": ("gap", "broadcast", "phase", "setpiece"), "tags": ("phase", "setpiece"),
       "no_bc": ("gap", "phase", "setpiece")}


def ranked(set_name="nogap", merge=3.0, seed=0, dur=None):
    """Every candidate of a decoder search (fresh/run_decoder.py), best mean DFL spellF1 first."""
    name = f"decoder-{bench.dtag(set_name, dur)}-m{merge:g}-s{seed}-all.npz"
    # a fresh search in fresh/out/ wins; else the tracked copy of the official one (DFL only)
    saved = load_search(name)
    candidates = [json.loads(c) for c in saved["candidates"]]
    return [candidates[i] for i in search_order(saved)]


def load_search(name):
    """A decoder search: a fresh one in fresh/out/ (with its DFL scores) wins; else the tracked
    copy of the official one, which keeps only the candidates and their rankings, no score."""
    path = data_io.OUT / name if (data_io.OUT / name).exists() else data_io.ARTIFACTS / name
    return dict(np.load(path))


def _key(excluded):
    return ",".join(str(j) for j in sorted(excluded))


def search_order(saved, excluded=(), top=None):
    """Candidate indices, best mean DFL spellF1 over the matches not in `excluded` first."""
    if "merged" in saved:
        scores = {"merged": saved["merged"], "raw": saved["raw"],
                  "mean": (saved["merged"] + saved["raw"]) / 2}[config.RANK]
        others = [j for j in range(scores.shape[1]) if j not in excluded]
        order = np.argsort(-scores[:, others].mean(1), kind="stable")
    elif config.RANK != "merged":
        raise SystemExit(f"decoder_rank {config.RANK!r} needs a search with its DFL scores in fresh/out/")
    elif not excluded:
        order = saved["order"]
    else:
        keys = [str(k) for k in saved["excluded"]]
        order = saved["top"][keys.index(_key(excluded))]
    return order if top is None else order[:top]


def strip_search(src, dst, n_matches=7, top=10):
    """Write the tracked copy of a search: candidates, the full ranking, and the top `top` of every
    leave-one and leave-two ranking the nested corrector needs; no score is kept."""
    saved = dict(np.load(src))
    sets = [(j,) for j in range(n_matches)]
    sets += [(j, m) for j in range(n_matches) for m in range(j + 1, n_matches)]
    np.savez(dst, candidates=saved["candidates"], order=search_order(saved),
             excluded=np.array([_key(e) for e in sets]),
             top=np.array([search_order(saved, e, top) for e in sets]))


def pick_params(rule, set_name="nogap", merge=3.0, seed=0):
    """rankN (0-based) or medianK of a decoder search."""
    top = ranked(set_name, merge, seed)
    if rule.startswith("rank"):
        return top[int(rule[4:])]
    return median_params(top[:int(rule.replace("median", ""))])


COSTS = ("team_switch", "to_dead", "from_dead")


def median_params(top):
    """Parameter-wise median of several candidates, the costs taken relative to each one's scale
    (the decoder only sees cost / scale), then put back on the median scale."""
    scale = float(np.median([p["scale"] for p in top]))
    out = {"scale": scale}
    for key in top[0]:
        if key in COSTS:
            out[key] = float(np.median([p[key] / p["scale"] for p in top])) * scale
        elif key in ("dur_weight", "switch_seen", "change_weight"):
            out[key] = float(np.median([p[key] for p in top]))
        elif key != "scale":
            out[key] = int(np.median([p[key] for p in top]))
    return out


def to_decode(c):
    return DecodeParams(team_switch=c["team_switch"], to_dead=c["to_dead"],
                        from_dead=c["from_dead"], min_possession=int(c["min_possession"]),
                        min_dead=int(c["min_dead"]), shift=int(c["shift"]))


def report(name, codes):
    sp, mf, n_pred, n_gt = bench.public_score(codes)
    n_poss, n_dead, live, graded = bench.counts(codes)
    print(f"{name:<28} public spellF1 {sp:.3f} macroF1 {mf:.3f} ({n_pred} vs {n_gt}) | match "
          f"{n_poss} possessions, {n_dead} stoppages, live {100 * live / graded:.0f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pick", default="median10")
    parser.add_argument("--merge", type=float, default=3.0)
    parser.add_argument("--set", default="nogap")
    parser.add_argument("--use", default="all,gap,gap_bc,tags,no_bc")
    parser.add_argument("--alpha", type=float, help="channel temper (default: the model's scale)")
    parser.add_argument("--iterations", type=int, default=8)
    args = parser.parse_args()

    c = pick_params(args.pick, args.set, args.merge)
    params = to_decode(c)
    print(f"decoder ({args.pick} of the {args.set} search, merge {args.merge:g} s): {c}")
    names = FEATURE_SETS[args.set]
    columns = None if names is None else bench.columns_of(names)
    model = bench.train(bench.dfl_set(), columns)
    x, _ = bench.match_frames()
    base = model.log_likelihood(x if columns is None else x[:, columns]) * c["scale"]
    tracking, tags, live = data_io.match()
    seen = np.isfinite(tracking.ball[:, 0])
    report("A (no adaptation)", bench.dead_before_kickoff(
        bench.decode_runs(base, bench.GRADED_RUNS, params)))
    alpha = args.alpha if args.alpha is not None else c["scale"]
    for use in args.use.split(","):
        channels = adapt.channels_for(seen, live, tags, USE[use])
        evidence, tables, _ = adapt.em(base, channels, params, alpha, args.iterations)
        codes = bench.dead_before_kickoff(bench.decode_runs(evidence, bench.GRADED_RUNS, params))
        report(f"B adapted: {use}", codes)
        for channel, table in zip(channels, tables, strict=True):
            ratio = np.log(table[:, 2] / table[:, 0])
            print(f"      {channel.name}: log q(bin|D)/q(bin|H) " + " ".join(f"{r:+.1f}" for r in ratio))


if __name__ == "__main__":
    main()
