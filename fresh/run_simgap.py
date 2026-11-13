"""Does the EM adaptation help when a match's tracker loses the ball the way this match's does?

Label-free test bench. Each DFL match in turn gets its ball hidden in a state-dependent way: over
most of each stoppage with some probability, and in long random gaps during play, at the rates the
EM on our match estimates (share of missing frames that are dead, overall missing share 41%). The
evidence comes from a model trained on the other six matches with gaps placed at random, exactly
as on the match; then experiment A (no adaptation) and B (EM with the gap channel) are decoded with
the same decoder and scored against DFL's merged truth.

    uv run python fresh/run_simgap.py --nested [--seeds 0,10,20] [--with-c]

Variants: A and B decode once with the median of the ten best decoders; Av and Bv let those ten
vote frame by frame (B's EM tables learned once, with the median decoder).
"""

import argparse
import json
from multiprocessing import Pool

import adapt
import bench
import config
import data_io
import feats
import numpy as np
import public
import selftrain
from run_adapt import median_params, to_decode
from run_lomo import FEATURE_SETS

from possession.official import FPS, GRADED_RUNS, A, D, H, X, segments
from possession.tracking import Tracking

SET = config.SET  # fresh/config.json
COLUMNS = bench.columns_of(FEATURE_SETS[SET])


def match_targets(params, scale):
    """(P(missing | dead), P(missing | live)) under the EM posteriors on the match. Always from
    the "nogap" evidence (variant B as declared), so every feature set meets the same frames."""
    columns = bench.columns_of(FEATURE_SETS["nogap"])
    model = bench.train(bench.dfl_set(), columns)
    x, _ = bench.match_frames()
    base = model.log_likelihood(x[:, columns]) * scale
    tracking, tags, live = data_io.match()
    seen = np.isfinite(tracking.ball[:, 0])
    channels = adapt.channels_for(seen, live, tags, ("gap", "broadcast"))
    evidence, _, _ = adapt.em(base, channels, params, scale)
    mask = data_io.runs_mask(len(seen), GRADED_RUNS)
    gamma = adapt.posterior(evidence, params)
    dead = gamma[mask, D]
    missing = ~seen[mask]
    return float((dead * missing).sum() / dead.sum()), float(((1 - dead) * missing).sum()
                                                            / (1 - dead).sum())


def hide_by_state(tracking, truth, runs, rng, p_dead, p_live):
    """Blur, then hide: each stoppage mostly hidden with a probability set so that ~p_dead of dead
    frames are missing, and long random gaps in play covering ~p_live of live frames."""
    ball = public.blur_ball(tracking.ball, rng)
    live_share = p_live
    mean_gap = public.GAP_MEDIAN * np.exp(public.GAP_SIGMA ** 2 / 2) * FPS
    mean_shown = mean_gap * (1 - live_share) / max(live_share, 1e-3)
    hidden = np.zeros(len(ball), dtype=bool)
    for start, stop in runs:
        for a, b, k in segments(truth, [(start, stop)]):
            if k != D:
                continue
            if rng.random() < min(1.0, p_dead / 0.85):
                lo = a + int(rng.uniform(0, 3) * FPS)
                hi = b - int(rng.uniform(0, 1) * FPS)
                hidden[lo:max(lo, hi)] = True
        t = start + int(rng.exponential(mean_shown))
        while t < stop:
            span = int(max(public.MIN_GAP, rng.lognormal(np.log(public.GAP_MEDIAN),
                                                          public.GAP_SIGMA)) * FPS)
            live_part = np.isin(truth[t:t + span], (H, A))
            hidden[t:t + span] |= live_part
            t += span + max(1, int(rng.exponential(mean_shown)))
    ball[hidden] = np.nan
    return Tracking(ball, tracking.players, tracking.team)


def top_candidates(tag, exclude=None, k=10, dur=None):
    """The k best candidates of the decoder search on `tag`'s evidence, ranked on every DFL match
    but `exclude` (all seven when None)."""
    from run_adapt import load_search, search_order
    saved = load_search(f"decoder-{bench.dtag(tag, dur)}-m{config.MERGE:g}-s{config.DECODER_SEED}-all.npz")
    candidates = [json.loads(c) for c in saved["candidates"]]
    excluded = () if exclude is None else (exclude,)
    return [candidates[j] for j in search_order(saved, excluded, k)]


def _decode_all(loglik, members, runs, channels=(), tables=None, durations=None, seen=None,
                q=None):
    """Each member decodes its own scaled evidence (channels at the same scale); one member is
    returned as is, several vote."""
    decoded = []
    for c in members:
        evidence = loglik * c["scale"]
        if channels:
            evidence = evidence + adapt._channel_scores(channels, tables, c["scale"])
        decoded.append(bench.decode_runs(evidence, runs, to_decode(c),
                                         tables=bench.dur_tables(durations, c),
                                         gain=None if seen is None else bench.gains(seen, q, c)))
    return decoded[0] if len(decoded) == 1 else bench.vote(decoded, runs)


def _fold(args):
    i, p_dead, p_live, seeds, nested, with_c = args
    top = top_candidates(bench.evidence_tag(SET, seeds), i if nested else None)
    c = median_params(top)
    params = to_decode(c)
    durations = bench.dfl_durations(exclude=i) if bench.DUR else None
    match_id = data_io.DFL_MATCHES[i]
    tracking, truth, runs, pitch = data_io.dfl(match_id)
    rng = np.random.default_rng(700 + i)
    degraded = hide_by_state(tracking, truth, runs, rng, p_dead, p_live)
    x, sign = feats.compute(degraded, runs, FPS, pitch)
    x = bench.with_bends(np.hstack([x, bench.dfl_long(match_id),
                                    bench.dfl_team(match_id, None, sign, degraded=degraded)]))
    x = np.hstack([x, bench.dfl_touch(match_id, None, degraded=degraded),
                   bench.dfl_shape(match_id, None, sign, degraded=degraded),
                   bench.dfl_ctrl(match_id, None, degraded=degraded),
                   bench.dfl_setp(match_id, None, degraded=degraded)])
    model = bench.train_seeds(COLUMNS, seeds, exclude=i)
    dfl_ll = model.log_likelihood(x[:, COLUMNS])
    target = public.merge_short(truth, 3.0)
    seen = np.isfinite(degraded.ball[:, 0])
    q = None
    if bench.CHANGE:
        import changes
        q = changes.probabilities(changes.train(exclude=i), x)
    graded = truth != X
    out = {"miss_dead": float((~seen)[truth == D].mean()),
           "miss_live": float((~seen)[np.isin(truth, (H, A))].mean())}
    channels = [adapt.gap_channel(seen)]
    evidence, tables, _ = adapt.em(dfl_ll * c["scale"], channels, params, c["scale"], runs=runs)
    decoded = [("A", _decode_all(dfl_ll, [c], runs, durations=durations, seen=seen, q=q)),
               ("Av", _decode_all(dfl_ll, top, runs, durations=durations, seen=seen, q=q)),
               ("B", bench.decode_runs(evidence, runs, params,
                                       tables=bench.dur_tables(durations, c),
                                       gain=bench.gains(seen, q, c))),
               ("Bv", _decode_all(dfl_ll, top, runs, channels, tables, durations, seen, q))]
    if with_c:
        gamma = adapt.posterior(evidence, params, runs)
        self_ll = selftrain.out_of_block(x, gamma, runs, COLUMNS)
        for name, w in (("C50", 0.5), ("C0", 0.0)):
            mixed = (w * dfl_ll + (1 - w) * self_ll) * c["scale"]
            evidence_c, _, _ = adapt.em(mixed, channels, params, c["scale"], runs=runs)
            decoded.append((name, bench.decode_runs(evidence_c, runs, params)))
    for name, codes in decoded:
        out[name] = (bench.spell(target, codes), bench.spell(truth, codes),
                     float((codes[graded] == D).mean()))
    true_tab = np.zeros((channels[0].n_bins, 2))
    for col, states in enumerate(((D,), (H, A))):
        sel = graded & np.isin(truth, states)
        true_tab[:, col] = np.bincount(channels[0].codes[sel], minlength=channels[0].n_bins)
    true_tab = (true_tab + 1) / (true_tab + 1).sum(0)
    out["true_ratio"] = np.log(true_tab[:, 0] / true_tab[:, 1]).round(1).tolist()
    out["em_ratio"] = np.log(tables[0][:, D] / tables[0][:, H]).round(1).tolist()
    out["true_dead"] = float((truth[graded] == D).mean())
    return match_id, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nested", action="store_true",
                        help="pick each match's decoders from a search ranked without it")
    parser.add_argument("--seeds", default="0", help="DFL degradation seeds of the evidence")
    parser.add_argument("--with-c", action="store_true", help="also the self-trained variants")
    parser.add_argument("--targets", action="store_true",
                        help="only write the match's gap targets cache (public.gap_targets)")
    args = parser.parse_args()
    if args.targets:
        if config.DEGRADE != 1:
            raise SystemExit("the gap targets are estimated with version-1 evidence")
        c = median_params(top_candidates("nogap", dur=False))
        targets = match_targets(to_decode(c), c["scale"])
        path = data_io.MATCH_CACHE / "gap-targets-v1.json"
        path.write_text(json.dumps(list(targets)) + "\n")
        print(f"wrote {path}")
        return
    seeds = bench.parse_seeds(args.seeds)
    # the simulated gaps always follow variant B's estimate, so every variant meets the same frames
    c = median_params(top_candidates("nogap", dur=False))
    p_dead, p_live = match_targets(to_decode(c), c["scale"])
    print(f"targets from the match's EM posteriors: missing on {p_dead:.2f} of dead frames, "
          f"{p_live:.2f} of live frames; evidence {bench.evidence_tag(SET, seeds)}, "
          f"{'nested' if args.nested else 'decoders ranked on all seven'}")
    with Pool(bench.WORKERS) as pool:
        results = pool.map(_fold, [(i, p_dead, p_live, seeds, args.nested, args.with_c)
                                   for i in range(len(data_io.DFL_MATCHES))])
    names = [k for k in ("A", "Av", "B", "Bv", "C50", "C0") if k in results[0][1]]
    rows = {k: [] for k in names}
    for match_id, out in results:
        print(f"{match_id}: simulated missing dead {out['miss_dead']:.2f} live "
              f"{out['miss_live']:.2f}, true dead {out['true_dead']:.2f} | "
              + " ".join(f"{k} {out[k][0]:.3f}" for k in names))
        for k, v in rows.items():
            v.append(out[k])
    for k, v in rows.items():
        v = np.array(v)
        print(f"{k}: mean merged spellF1 {v[:, 0].mean():.3f}, raw {v[:, 1].mean():.3f}, "
              f"dead share {v[:, 2].mean():.2f}")
    for name in names[1:]:
        diff = np.array(rows[name])[:, 0] - np.array(rows["A"])[:, 0]
        print(f"{name} - A merged: {diff.mean():+.3f} (wins {int((diff > 0).sum())} of {len(diff)})")
    if "Bv" in rows:
        diff = np.array(rows["Bv"])[:, 0] - np.array(rows["B"])[:, 0]
        print(f"Bv - B merged: {diff.mean():+.3f} (wins {int((diff > 0).sum())} of {len(diff)})")


if __name__ == "__main__":
    main()
