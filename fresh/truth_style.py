"""Which DFL truth definition matches the public annotation (pre-registered in README)."""

import data_io
import numpy as np
import public
from scipy.stats import ks_2samp

from possession.official import FPS, A, H, segments


def segs(truth):
    out = []  # (label, frames, interior)
    for a, b in public._runs(truth):
        ss = segments(truth, [(a, b)])
        for i, (s, e, k) in enumerate(ss):
            out.append((k, e - s, 0 < i < len(ss) - 1))
    return out


def stats(sg):
    k = np.array([x[0] for x in sg])
    n = np.array([x[1] for x in sg])
    inn = np.array([x[2] for x in sg])
    ha = np.isin(k, (H, A))
    dur = n[ha & inn] / FPS
    minutes = n.sum() / FPS / 60
    return {
        "dur": dur,
        "median": float(np.median(dur)),
        "short": float((dur < 3).mean()),
        "ppm": ha.sum() / minutes,
        "alive": n[ha].sum() / n.sum(),
        "n": int(ha.sum()),
    }


def dist(p, v):
    return ks_2samp(p["dur"], v["dur"]).statistic + abs(p["ppm"] - v["ppm"]) / v["ppm"]


truths = [data_io.dfl(m)[1] for m in data_io.DFL_MATCHES]
variants = {}
for name, sec in (("raw", 0), ("1s", 1), ("2s", 2), ("3s", 3)):
    sg = []
    for t in truths:
        sg += segs(public.merge_short(t, sec) if sec else t)
    variants[name] = stats(sg)
pub_sg = segs(data_io.load_public_gt())
pub = stats(pub_sg)
print(
    f"public: n={pub['n']} median={pub['median']:.2f} short={pub['short']:.2f} ppm={pub['ppm']:.2f} alive={pub['alive']:.3f} ndur={len(pub['dur'])}"
)
for name, v in variants.items():
    print(
        f"{name}: n={v['n']} median={v['median']:.2f} short={v['short']:.2f} ppm={v['ppm']:.2f} alive={v['alive']:.3f} "
        f"KS={ks_2samp(pub['dur'], v['dur']).statistic:.3f} dist={dist(pub, v):.3f}"
    )
rng = np.random.default_rng(0)
names = list(variants)
wins = dict.fromkeys(names, 0)
ds = {n: [] for n in names}
for _ in range(2000):
    b = [pub_sg[i] for i in rng.integers(0, len(pub_sg), len(pub_sg))]
    bs = stats(b)
    if len(bs["dur"]) == 0:
        continue
    d = {n: dist(bs, variants[n]) for n in names}
    wins[min(d, key=d.get)] += 1
    for n in names:
        ds[n].append(d[n])
print("bootstrap argmin share:", {n: round(w / 2000, 3) for n, w in wins.items()})
print("dist 90% CI:", {n: tuple(np.round(np.percentile(ds[n], [5, 95]), 3)) for n in names})
