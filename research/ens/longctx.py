"""Long-context family for H40: retrieval and multi-hop over pair facts scattered among distractors.

A sequence of length L holds n_pairs facts "GR a b" (a -> b, a and b value tokens; every source a is unique, so
the facts form a function), interleaved at random positions with filler key tokens. The query at the end asks
for the value reached from a start value after k hops:

  [BOS] ... GR a b ... filler ... GR c d ... [Q] V(start) [HOP] V(k) [SLOT] [NEUTRAL] [ANS]  -> answer

The chain start -> ... is guaranteed to have k hops inside the facts; the other facts are distractors. With
k = 1 this is plain long-range retrieval. Rows are generated from a fixed seed; train and held rows come from
different seeds, so no sequence repeats across splits.

  field position = the NEUTRAL slot (a bus write lands there), answer position = the final ANS token.
"""
from __future__ import annotations

import random

from research.ens import vocab as V

N_VAL = V.N_VALUES


def make_row(rng: random.Random, L: int, k: int, n_pairs: int):
    """One row: (tokens list of length L, field_idx, ans_idx, answer value index, chain)."""
    if n_pairs > N_VAL or n_pairs < k:
        raise ValueError("need k <= n_pairs <= number of value tokens")
    sources = rng.sample(range(N_VAL), n_pairs)
    mapping = {}
    for s in sources:
        mapping[s] = rng.randrange(N_VAL)
    # force a k-hop chain from a start whose path stays inside the facts
    start = sources[0]
    chain = [start]
    cur = start
    free = [s for s in sources if s != start]
    for hop in range(k):
        if hop < k - 1:
            nxt = free.pop(rng.randrange(len(free)))       # the next hop is itself a source
            mapping[cur] = nxt
        else:
            # last hop: a value not already on the chain, so the answer never equals the start or a node
            nxt = mapping[cur] if mapping[cur] not in chain else rng.choice([x for x in range(N_VAL) if x not in chain])
            mapping[cur] = nxt
        chain.append(mapping[cur])
        cur = mapping[cur]
    answer = chain[-1]
    # the hop count is a key token after HOP, not a value token, so it cannot be confused with the start value
    tail = [V.Q, V.val(start), V.tok("HOP"), V.key(k), V.SLOT, V.NEUTRAL, V.ANS]
    body_len = L - 1 - len(tail)
    facts = [[V.tok("GR"), V.val(a), V.val(b)] for a, b in mapping.items()]
    if 3 * len(facts) > body_len:
        raise ValueError("sequence too short for the facts")
    n_fill = body_len - 3 * len(facts)
    # place facts at random slots among filler tokens (keys K0..K15 as noise)
    units = facts + [[V.key(rng.randrange(V.N_KEYS))] for _ in range(n_fill)]
    rng.shuffle(units)
    body = [t for u in units for t in u]
    ids = [V.BOS] + body + tail
    assert len(ids) == L
    return ids, L - 2, L - 1, answer, chain


def make_rows(n: int, *, L: int = 512, k: int = 1, n_pairs: int = 48, seed: int = 0):
    rng = random.Random(f"ens.longctx.v0:{L}:{k}:{n_pairs}:{seed}")
    return [make_row(rng, L, k, n_pairs) for _ in range(n)]


def answer_from_tokens(ids, k: int):
    """Re-derive the answer from the token sequence alone (oracle used by the tests)."""
    m = {}
    for i in range(len(ids) - 2):
        if ids[i] == V.tok("GR"):
            m[ids[i + 1] - V.V0] = ids[i + 2] - V.V0
    q = ids.index(V.Q)
    cur = ids[q + 1] - V.V0
    for _ in range(k):
        cur = m[cur]
    return cur
