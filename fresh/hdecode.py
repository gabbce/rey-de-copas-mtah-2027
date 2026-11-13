"""A duration-aware Viterbi: the decoder of possession.decode plus a learned log-probability of
each segment's length (an explicit-duration HSMM, Yu 2010). Idea 2, Oct 22.

The base decoder already gives every state a chain of sub-states (a segment's age) and a minimum
length. Here the chain runs to DMAX frames, a segment may end at any age from its minimum on, and
ending it at age d adds weight * log p(d), with p a discretised log-normal fitted to the lengths of
DFL's merged-truth segments (H and A pooled, D apart). Segments at least DMAX long share the tail
mass. The first and the last segment of a stretch, cut by its edges, get no duration term, as they
get no minimum in the base decoder. Scores are integers on the base decoder's grid, so the output
is identical on any machine.
"""

import numpy as np

from possession.decode import NEVER, _quantise, transition_costs
from possession.official import FPS, A, D, H, segments

DMAX = {H: 30 * FPS, A: 30 * FPS, D: 60 * FPS}  # frames: chain length per state


def fit(truths, runs_list):
    """Log-normal (mu, sigma) of interior segment lengths in frames: {"live": .., "dead": ..}."""
    lengths = {"live": [], "dead": []}
    for truth, runs in zip(truths, runs_list, strict=True):
        for start, stop in runs:
            segs = segments(truth, [(start, stop)])
            for a, b, k in segs[1:-1]:
                lengths["dead" if k == D else "live"].append(b - a)
    out = {}
    for key, values in lengths.items():
        logs = np.log(np.asarray(values, dtype=np.float64))
        out[key] = [float(logs.mean()), float(logs.std())]
    return out


def bonus_tables(durations, weight, min_lengths):
    """Per state, (DMAX,) integer bonus for ending a segment at age j + 1 (last entry: any age at
    least DMAX); NEVER below the state's minimum length."""
    tables = []
    for s in (0, 1, 2):
        mu, sigma = durations["dead" if s == D else "live"]
        dmax = DMAX[s]
        d = np.arange(1, dmax + 1, dtype=np.float64)
        logpdf = -np.log(d) - (np.log(d) - mu) ** 2 / (2 * sigma ** 2)
        grid = np.arange(1, 50 * dmax, dtype=np.float64)
        logall = -np.log(grid) - (np.log(grid) - mu) ** 2 / (2 * sigma ** 2)
        top = logall.max()
        norm = np.log(np.exp(logall - top).sum()) + top
        logp = logpdf - norm
        tail = np.log(np.exp(logall[dmax - 1:] - top).sum()) + top - norm
        logp[-1] = tail
        table = _quantise(weight * logp)
        table[:max(0, min_lengths[s] - 1)] = NEVER
        tables.append(table)
    return tables


def viterbi(scores, params, tables, gain=None):
    """Best H/A/D path through (n, 3) scores under `params` (costs, shift unused here) and the
    duration `tables` of bonus_tables, as int8 codes."""
    n = len(scores)
    if not n:
        return np.zeros(0, dtype=np.int8)
    cost = transition_costs(params)
    scores = _quantise(scores)
    gain = _quantise(gain) if gain is not None and np.any(gain) else None
    states = range(3)
    # chains[s][j]: best score with the current segment in s and j + 1 frames old (last: >= DMAX)
    # first[s]: the stretch's first segment, which has no duration term and no minimum
    chains = [np.full(len(tables[s]), NEVER, dtype=np.int64) for s in states]
    first = np.array([scores[0, s] for s in states], dtype=np.int64)
    in_first = np.ones(3, dtype=bool)
    entered_from = np.zeros((n, 3), dtype=np.int8)
    exit_link = np.full((n, 3), -1, dtype=np.int32)  # link the previous segment ended at; -1 first
    stayed = np.zeros((n, 3), dtype=bool)
    for t in range(1, n):
        ready = np.empty(3, dtype=np.int64)
        link = np.empty(3, dtype=np.int64)
        for s in states:
            total = chains[s] + tables[s]
            j = int(np.argmax(total))
            ready[s], link[s] = total[j], j
            if in_first[s] and first[s] >= ready[s]:
                ready[s], link[s] = first[s], -1
        options = ready[:, None] - cost
        if gain is not None:  # a hand-over into H or A made cheaper (or dearer) at this frame
            options[A, H] += gain[t, H]
            options[H, A] += gain[t, A]
        np.fill_diagonal(options, NEVER)
        src = np.argmax(options, axis=0)
        entered_from[t] = src
        entry = options[src, np.arange(3)]
        exit_link[t] = link[src]
        for s in states:
            chain = chains[s]
            grown = np.empty_like(chain)
            stayed[t, s] = chain[-1] >= chain[-2]
            grown[0] = entry[s]
            grown[1:-1] = chain[:-2]
            grown[-1] = max(chain[-1], chain[-2])
            chains[s] = grown + scores[t, s]
        first = first + scores[t]
    # the last segment: no duration term, no minimum
    ends = []
    for s in states:
        j = int(np.argmax(chains[s]))
        ends.append((int(chains[s][j]), s, j))
        if in_first[s]:
            ends.append((int(first[s]), s, -1))
    _, state, link = max(ends, key=lambda e: (e[0], -e[1], e[2]))
    path = np.empty(n, dtype=np.int8)
    path[-1] = state
    last = {s: len(tables[s]) - 1 for s in states}
    for t in range(n - 1, 0, -1):
        if link == -1:
            pass  # the first segment runs back to the start
        elif link == last[state] and stayed[t, state]:
            pass
        elif link == 0:
            prev = int(entered_from[t, state])
            link = int(exit_link[t, state])
            state = prev
        else:
            link -= 1
        path[t - 1] = state
    return path
