"""Turn per-frame evidence into whole possessions with a three-state Viterbi.

A change of state costs a fixed penalty, so it only happens once the evidence for the new state
has outweighed it. That is what keeps a possession from flickering -- the most expensive mistake
on spellF1 -- and it also means a stretch with no evidence (the ball unseen) simply carries the
current state, the way an abstention does on the board.

Each state also has a minimum length. A segment that starts and ends inside the decoded stretch
can never be shorter than its state's minimum -- the Viterbi runs over a chain of sub-states per
state and may only leave a chain from its last link -- so a brief burst of contrary evidence only
becomes a possession if it outweighs a whole minimum-length segment, not just its own frames. Only
the first and last segments, which the stretch's edges may cut, are exempt.

The search runs on integers. Scores are rounded to a grid of QUANTUM first, so the last-bit float
differences between machines, libm builds or thread counts cannot flip a near-tie: the rules
require the organisers' re-run to reproduce the submitted file exactly.
"""

from dataclasses import dataclass

import numpy as np

from possession.official import A, D, H, X

QUANTUM = 1e-4  # nats; far below any difference that matters, far above float noise


@dataclass(frozen=True)
class DecodeParams:
    team_switch: float = 12.0  # nats to hand the ball straight to the other team
    to_dead: float = 8.0  # nats to call a stoppage
    from_dead: float = 8.0  # nats to restart play
    # Minimum lengths in frames. These defaults are only a starting point; the submitted
    # decoder stores its own lengths, chosen on external matches.
    min_possession: int = 10
    min_dead: int = 50
    shift: int = 0  # frames every decoded change is moved later (earlier if negative)


DEFAULT_DECODE = DecodeParams()


def _quantise(values):
    return np.round(np.asarray(values, dtype=np.float64) / QUANTUM).astype(np.int64)


def transition_costs(params):
    """3x3 integer costs in QUANTUM units, [from, to], indexed by the official H, A, D codes."""
    cost = np.zeros((3, 3))
    cost[H, A] = cost[A, H] = params.team_switch
    cost[H, D] = cost[A, D] = params.to_dead
    cost[D, H] = cost[D, A] = params.from_dead
    return _quantise(cost)


NEVER = np.iinfo(np.int64).min // 4  # an unreachable score that cannot overflow when added to


def _min_lengths(params):
    """Chain length per state, indexed by the official H, A, D codes."""
    lengths = [1, 1, 1]
    lengths[H] = lengths[A] = max(1, int(params.min_possession))
    lengths[D] = max(1, int(params.min_dead))
    return lengths


def viterbi(scores, params=DEFAULT_DECODE, gain=None):
    """Best H/A/D path through (n, 3) scores, as int8 codes, with every segment that starts and
    ends inside the stretch at least its state's minimum length.

    `gain`, optional (n, 3): nats off the cost of a team-to-team hand-over into H or A at each
    frame, so a change of possession can be cheap exactly where the new side touched the ball.
    """
    n = len(scores)
    if not n:
        return np.zeros(0, dtype=np.int8)
    cost = transition_costs(params)
    np.fill_diagonal(cost, -NEVER)  # staying is the chain's business, not a transition
    scores = _quantise(scores)
    gain = _quantise(gain) if gain is not None and np.any(gain) else None
    lengths = _min_lengths(params)

    # chains[s][j]: best score of a path whose current segment is state s and j+1 frames old,
    # the last link also standing for "old enough to leave". The first segment may have begun
    # before the stretch, so it starts already old enough.
    chains = [np.full(length, NEVER, dtype=np.int64) for length in lengths]
    for s in range(3):
        chains[s][-1] = scores[0, s]
    entered_from = np.zeros((n, 3), dtype=np.int8)  # state a segment starting at t came from
    stayed = np.zeros((n, 3), dtype=bool)  # last link at t continued the same segment

    for t in range(1, n):
        ready = np.array([chain[-1] for chain in chains])
        options = ready[:, None] - cost  # [from, to]
        if gain is not None:
            options[A, H] += gain[t, H]
            options[H, A] += gain[t, A]
        entered_from[t] = np.argmax(options, axis=0)
        entry = options[entered_from[t], np.arange(3)]
        for s in range(3):
            chain = chains[s]
            grown = np.empty_like(chain)
            if len(chain) == 1:
                stayed[t, s] = chain[0] >= entry[s]
                grown[0] = max(chain[0], entry[s])
            else:
                stayed[t, s] = chain[-1] >= chain[-2]
                grown[0] = entry[s]
                grown[1:-1] = chain[:-2]
                grown[-1] = max(chain[-1], chain[-2])
            chains[s] = grown + scores[t, s]

    ends = [(int(chain.max()), s, int(np.argmax(chain))) for s, chain in enumerate(chains)]
    _, state, link = max(ends, key=lambda e: (e[0], -e[1]))  # ties: lowest state code
    path = np.empty(n, dtype=np.int8)
    path[-1] = state
    for t in range(n - 1, 0, -1):
        last = lengths[state] - 1
        if link == last and stayed[t, state]:
            pass  # the same segment, already old enough, one frame earlier
        elif link == 0:
            state = int(entered_from[t, state])
            link = lengths[state] - 1
        else:
            link -= 1
        path[t - 1] = state
    return path


def shift_changes(codes, runs, lag):
    """Every label change inside each run moved `lag` frames later (earlier if negative).

    For a systematic timing offset between our evidence and the reference's notion of control:
    the metric lab shows a uniform second of lag costs a quarter of spellF1.
    """
    out = codes.copy()
    for start, stop in runs:
        window = codes[start:stop]
        k = min(abs(lag), len(window))
        if lag > 0:
            out[start + k:stop] = window[:len(window) - k]
            out[start:start + k] = window[0]
        elif lag < 0:
            out[start:stop - k] = window[k:]
            out[stop - k:stop] = window[-1]
    return out


def decode(scores, runs, params=DEFAULT_DECODE, gain=None):
    """Full-length labels: Viterbi inside each graded run, X everywhere else."""
    out = np.full(len(scores), X, dtype=np.int8)
    for start, stop in runs:
        out[start:stop] = viterbi(scores[start:stop], params,
                                  None if gain is None else gain[start:stop])
    return shift_changes(out, runs, params.shift) if params.shift else out
