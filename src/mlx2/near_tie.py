"""Pure first-divergence numerical equivalence rules (no model state)."""
from __future__ import annotations

import math


def near_tie_equivalent(left_selected, left, right_selected, right, *, ceiling=0.5):
    """Accept shared top-two, exact ties, or bounded multi-way cross gaps.

    Callers own prefix, physical-width, trace-binding and functional gates.
    Every mapping contains log probabilities from the same divergence position.
    Missing cross tokens never imply equivalence.
    """
    def valid(values, selected):
        return (isinstance(values, dict) and len(values) >= 2 and selected in values
                and all(type(k) is int and type(v) in (int, float) and math.isfinite(v)
                        for k, v in values.items())
                and max(values.values()) - values[selected] <= 1e-6)
    if not valid(left, left_selected) or not valid(right, right_selected):
        return False
    left_two = sorted(left, key=left.get, reverse=True)[:2]
    right_two = sorted(right, key=right.get, reverse=True)[:2]
    lm = left[left_selected] - next(left[t] for t in left_two if t != left_selected)
    rm = right[right_selected] - next(right[t] for t in right_two if t != right_selected)
    if -1e-6 <= lm <= ceiling and -1e-6 <= rm <= ceiling:
        if set(left_two) == set(right_two):
            return True
        if lm == 0.0 and right_selected in left_two:
            return True
        if rm == 0.0 and left_selected in right_two:
            return True
    return (right_selected in left and left_selected in right
            and 0 <= max(left.values()) - left[right_selected] <= ceiling
            and 0 <= max(right.values()) - right[left_selected] <= ceiling)
