"""AGENTS.md near-tie classification of one first-divergence position.

Inputs are the two arms' log-probability rows at the divergent position
(any 1-D float sequences over the same vocabulary; full rows, or a dense row
rebuilt from a window). The caller guarantees the shared prefix (the first
divergence of the two streams, or a same-state single step) and runs the
functional oracle on the continuations where one applies; this module
checks the margin clauses:

* top-two clause: the same unordered top two in both arms, both
  selected-versus-alternate margins at most the 0.5-nat ceiling;
* exact-tie clause: one arm's margin is exactly zero; the other arm selected
  one of that arm's tied tokens, with its own margin at most the ceiling;
* multi-way clause: each arm's selected token lies within the ceiling of the
  other arm's top log-probability. A token missing from a window leaves the
  row unclassified unless the window proves it is further than the ceiling
  below the top (which fails the row); with full rows nothing is missing.
"""

from __future__ import annotations

import math

CEILING = 0.5


def _top(row, k):
    return sorted(range(len(row)), key=lambda t: (-row[t], t))[:k]


def window(row, k=8):
    """``[(token, logp)]`` of the ``k`` most probable tokens."""
    return [(t, float(row[t])) for t in _top(row, k)]


def _exact_tie(tied_row, tied_margin, other_selected, other_margin, ceiling):
    if tied_margin != 0.0 or other_margin > ceiling:
        return False
    top = max(tied_row)
    return tied_row[other_selected] == top


def classify(row_a, row_b, *, ceiling=CEILING, k=8):
    """Verdict and evidence for one divergent position of arms ``a``/``b``."""
    a, b = _top(row_a, 2), _top(row_b, 2)
    margins = (float(row_a[a[0]] - row_a[a[1]]), float(row_b[b[0]] - row_b[b[1]]))
    out = {"a_top2": a, "b_top2": b, "margins": margins,
           "a_window": window(row_a, k), "b_window": window(row_b, k),
           # each arm's log-probability of the other arm's selected token
           "a_logp_of_b_selected": float(row_a[b[0]]),
           "b_logp_of_a_selected": float(row_b[a[0]])}
    if a[0] == b[0]:
        out["verdict"] = "same_selection"
    elif set(a) == set(b) and max(margins) <= ceiling:
        out["verdict"] = "near_tie_equivalent"
        out["clause"] = "top_two"
    elif (_exact_tie(row_a, margins[0], b[0], margins[1], ceiling)
          or _exact_tie(row_b, margins[1], a[0], margins[0], ceiling)):
        out["verdict"] = "near_tie_equivalent"
        out["clause"] = "exact_tie"
    elif (row_a[a[0]] - row_a[b[0]] <= ceiling and row_b[b[0]] - row_b[a[0]] <= ceiling):
        out["verdict"] = "near_tie_equivalent"
        out["clause"] = "multi_way"
    else:
        out["verdict"] = "unclassified"
    return out


def dense(window_pairs, vocab, floor=-math.inf):
    """A dense row from ``[(token, logp)]`` (tokens outside it at ``floor``)."""
    row = [floor] * vocab
    for token, logp in window_pairs:
        row[int(token)] = float(logp)
    return row
