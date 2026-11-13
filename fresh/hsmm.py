"""Posterior state probabilities under the decoder's own path model, for EM on the match.

The decoder (possession.decode.viterbi) scores a path by its per-frame evidence minus a cost per
change of state, and forbids any segment shorter than its state's minimum length except the first
and the last, which the stretch's edges may cut. The same scores define a distribution over paths,
p(path) proportional to exp(score); `posteriors` returns each frame's marginal under it, by a
forward-backward over segments in O(n x states) -- the interior of a minimum-length segment is a
plain sum of evidence, read off cumulative sums instead of a chain of sub-states.
"""

import math

import numpy as np

NEG = -math.inf


def _lae(a, b):
    if a < b:
        a, b = b, a
    if b == NEG:
        return a
    return a + math.log1p(math.exp(b - a))


def posteriors(evidence, cost, lengths):
    """(marginals (n, k), log normaliser) for (n, k) evidence, a (k, k) [from, to] cost matrix
    (diagonal ignored) and per-state minimum lengths."""
    e = np.asarray(evidence, dtype=np.float64)
    n, k = e.shape
    cum = np.vstack([np.zeros(k), np.cumsum(e, axis=0)])
    el = e.tolist()
    cl = cum.tolist()
    cost = np.asarray(cost, dtype=np.float64).tolist()
    lengths = [max(1, int(v)) for v in lengths]
    states = range(k)

    def seg(y, a, b):  # sum of evidence of state y over frames a..b inclusive
        return cl[b + 1][y] - cl[a][y]

    ready = [[NEG] * k for _ in range(n)]
    entry = [[NEG] * k for _ in range(n)]
    ready[0] = el[0][:]
    for t in range(1, n):
        prev = ready[t - 1]
        for y in states:
            v = NEG
            for z in states:
                if z != y:
                    v = _lae(v, prev[z] - cost[z][y])
            entry[t][y] = v
        for y in states:
            stay = prev[y] + el[t][y]
            u = t - lengths[y] + 1
            fresh = entry[u][y] + seg(y, u, t) if u >= 1 else NEG
            ready[t][y] = _lae(stay, fresh)
    log_z = NEG
    for y in states:
        log_z = _lae(log_z, ready[n - 1][y])
        for u in range(max(1, n - lengths[y] + 1), n):
            log_z = _lae(log_z, entry[u][y] + seg(y, u, n - 1))

    back = [[0.0] * k for _ in range(n)]
    cont = [[NEG] * k for _ in range(n)]  # continuation after entering y at u
    for u in range(n - 1, 0, -1):
        for y in states:
            r = u + lengths[y] - 1
            cont[u][y] = seg(y, u, r) + back[r][y] if r <= n - 1 else seg(y, u, n - 1)
        if u >= 1:
            t = u - 1
            for y in states:
                v = el[u][y] + back[u][y]
                for z in states:
                    if z != y:
                        v = _lae(v, cont[u][z] - cost[y][z])
                back[t][y] = v

    ready_a = np.array(ready)
    back_a = np.array(back)
    entry_a = np.array(entry)
    cont_a = np.array(cont)
    with np.errstate(invalid="ignore"):
        p_ready = np.exp(ready_a + back_a - log_z)
        p_entry = np.exp(entry_a + cont_a - log_z)
    p_entry[0] = 0.0
    p_entry = np.nan_to_num(p_entry)
    p_ready = np.nan_to_num(p_ready)
    csum = np.vstack([np.zeros(k), np.cumsum(p_entry, axis=0)])
    occupancy = p_ready.copy()
    idx = np.arange(n)
    for y in states:
        low = np.maximum(1, idx - lengths[y] + 2)
        occupancy[:, y] += csum[idx + 1, y] - csum[low, y]
    return occupancy, log_z
