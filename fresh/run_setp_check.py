"""E9 report (docs/r2-rule.md), deciding nothing: do the set-piece readings mean the same on the
match as on DFL? Prints per-column means only, on the match's graded frames and on each DFL
match's live and dead frames, so a reading the match's tracker never produces (players beyond
the lines, say) shows as a column near zero on the match. Aggregates only.

    uv run python fresh/run_setp_check.py
"""

import bench
import data_io
import numpy as np
import setpiece

from possession.official import GRADED_RUNS


def main():
    now = [j for j, n in enumerate(setpiece.NAMES) if n.endswith("_now")]
    x = bench.match_setp()
    mask = data_io.runs_mask(len(x), GRADED_RUNS)
    match_mean = x[mask][:, now].mean(0)
    live, dead = [], []
    for match_id in data_io.DFL_MATCHES:
        _, truth, runs, _ = data_io.dfl(match_id)
        d = bench.dfl_setp(match_id)
        inside = data_io.runs_mask(len(truth), runs)
        live.append(d[inside & (truth < 2)][:, now].mean(0))
        dead.append(d[inside & (truth == 2)][:, now].mean(0))
    live, dead = np.mean(live, 0), np.mean(dead, 0)
    print(f"{'reading':<16}{'match':>8}{'DFL live':>10}{'DFL dead':>10}")
    for k, j in enumerate(now):
        print(f"{setpiece.NAMES[j]:<16}{match_mean[k]:8.3f}{live[k]:10.3f}{dead[k]:10.3f}")


if __name__ == "__main__":
    main()
