"""Write the fresh pipeline's full-match submission and check it. Aggregates only.

    uv run python fresh/predict.py                 # variant B from fresh/model-B.json
    uv run python fresh/predict.py --refit         # learn it again first (label-free)
    uv run python fresh/predict.py --variant A --out /tmp/a.csv

The CSV goes to fresh/out/ by default, which a local .gitignore keeps untracked. Prints the official
validator's verdict, the whole-match possession count and the file's SHA-256 (the public window's
score only with --report-public, as a report: it decides nothing), so a re-run can be compared byte for byte.
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

import bench
import data_io
import pipeline

from possession.submission import validate, write_submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default=pipeline.FINAL, choices=pipeline.VARIANTS)
    parser.add_argument("--refit", action="store_true", help="learn the artifacts again")
    parser.add_argument("--out", type=Path, help="CSV path (default fresh/out/fresh-<variant>.csv)")
    parser.add_argument("--report-public", action="store_true",
                        help="also print the public window's score: a report only, never a choice")
    args = parser.parse_args()
    began = time.time()
    if args.refit or pipeline.load(args.variant) is None:
        body = pipeline.fit(args.variant)
        pipeline.save(body, args.variant)
        print(f"fitted and saved {pipeline.artifact_path(args.variant).name}")
    codes = pipeline.predict_match(args.variant)
    name = pipeline.artifact_path(args.variant).stem.replace("model-", "fresh-")
    out = write_submission(args.out or data_io.OUT / f"{name}.csv", codes)
    reason = validate(out)
    if reason:
        sys.exit(f"REJECTED by the official validator: {reason}")
    n_poss, n_dead, live, graded = bench.counts(codes)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"variant {args.variant}: wrote {out.name}, valid")
    if args.report_public:
        spell, macro, n_pred, n_gt = bench.public_score(codes)
        low, high = bench.bootstrap_public(codes)
        print(f"public window (report only): spellF1 {spell:.4f} [{low:.3f}, {high:.3f}]  "
              f"macroF1 {macro:.4f}  possessions {n_pred} predicted / {n_gt} true")
    print(f"whole match: {n_poss} possessions, {n_dead} stoppages, live {100 * live / graded:.1f}% "
          f"of {graded:.1f} graded minutes")
    print(f"sha256 {digest}  ({time.time() - began:.0f} s)")


if __name__ == "__main__":
    main()
