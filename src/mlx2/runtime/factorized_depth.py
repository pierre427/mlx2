"""Bounded online depth predictions from conditional acceptance and round cost.

Each owner belongs to one loaded runtime. Exact physical widths have separate
tables; no shape-only persistence can leak costs between checkpoints. Rejected
prefix tails are censored, not counted as further independent rejections.
"""

import math
from collections import OrderedDict


class FactorizedDepthCost:
    def __init__(self, max_depth, *, alpha, min_samples, stale_rounds):
        self.max_depth = max_depth
        self.alpha = alpha
        self.min_samples = min_samples
        self.stale_rounds = stale_rounds
        self.widths = OrderedDict()
        self.samples = 0

    def _state(self, width):
        if width not in self.widths:
            if len(self.widths) >= 32:
                self.widths.popitem(last=False)
            self.widths[width] = {"cost": {}, "landing": {}}
        self.widths.move_to_end(width)
        return self.widths[width]

    def _update(self, table, key, value, boundary):
        old = table.get(key)
        if old is None or boundary - old[2] >= self.stale_rounds:
            table[key] = (float(value), 1, boundary)
        else:
            table[key] = (self.alpha * value + (1 - self.alpha) * old[0],
                          min(old[1] + 1, (1 << 63) - 1), boundary)

    def observe(self, *, width, depth, depths, accepted, seconds, boundary):
        # The caller excludes copied, terminal and padded/mixed cohorts.
        # Validate the complete record before mutating any learned state.
        if (len(depths) != width or len(accepted) != width
                or any(isinstance(d, bool) or not isinstance(d, int) or d != depth for d in depths)
                or any(isinstance(a, bool) or not isinstance(a, int) or not 0 <= a <= depth for a in accepted)
                or not 0 <= depth <= self.max_depth
                or not math.isfinite(seconds) or seconds <= 0):
            raise ValueError("invalid factorized depth observation")
        state = self._state(width)
        self._update(state["cost"], width * (depth + 1), seconds, boundary)
        for position in range(1, depth + 1):
            reached = sum(a >= position - 1 for a in accepted)
            if reached:
                landed = sum(a >= position for a in accepted)
                self._update(state["landing"], position, landed / reached, boundary)
        self.samples = min(self.samples + 1, (1 << 63) - 1)

    def predictions(self, *, width, cap, boundary):
        state = self.widths.get(width)
        if state is None:
            return {}

        def fresh(table):
            return {key: value for key, (value, count, last) in table.items()
                    if count >= self.min_samples and boundary - last < self.stale_rounds}

        costs = fresh(state["cost"])
        landing = fresh(state["landing"])
        if not costs:
            return {}
        # Use a monotone envelope to avoid treating a noisy larger forward as
        # arbitrarily cheaper. Only interpolate inside observed geometry.
        rows = sorted(costs)
        running = 0.0
        for row in rows:
            running = max(running, costs[row])
            costs[row] = running
        predictions = {}
        expected, reach = 1.0, 1.0
        for depth in range(min(cap, self.max_depth) + 1):
            if depth:
                if depth not in landing:
                    break
                reach *= landing[depth]
                expected += reach
            target = width * (depth + 1)
            if target < rows[0] or target > rows[-1]:
                continue
            if target in costs:
                seconds = costs[target]
            else:
                high_index = next(i for i, row in enumerate(rows) if row > target)
                low, high = rows[high_index - 1], rows[high_index]
                seconds = costs[low] + (costs[high] - costs[low]) * (target - low) / (high - low)
            predictions[depth] = width * expected / seconds
        return predictions

    def diagnostics(self):
        return {"samples": self.samples, "widths": list(self.widths),
                "cost_contract": "whole-round-seconds; interpolate observed rows only",
                "acceptance_contract": "conditional-prefix; rejected tails censored"}
