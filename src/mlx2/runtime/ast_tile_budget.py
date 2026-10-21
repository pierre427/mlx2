"""Measured tree-tile size: proposal nodes per round by tokens per second.

Original mlx2 implementation (the idea follows Kapitsa's KFlash budget,
n = argmax E(n) - rate * cost(n), and TensorFold's plan_lanes).  A best-first
tree lists parents before children, so its first ``n`` nodes are the best
``n``-node tree, and the committed path's node ranks increase along the
path.  One round verified with ``N`` nodes therefore tells what every
smaller ``n`` would have committed: ``1 + #{path nodes ranked below n}``.
Cost is measured per verified size and fitted as ``a + b * rows``, which
captures both flat lane verifies (dense 4-bit) and steep ones (MoE expert
loads, long contexts).

Per lane width the controller runs the knob maximum to learn acceptance,
probes one smaller size for the cost slope, then serves the size with the
best expected tokens per second, refreshing at the maximum every
``refresh`` rounds.  State is generator-local timing, never shared.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CANDIDATES = (3, 5, 7, 9, 11, 15, 23, 31)


@dataclass
class _Width:
    rounds: int = 0
    tokens: dict = field(default_factory=dict)      # n -> summed counterfactual tokens
    samples: dict = field(default_factory=dict)     # n -> rounds observed with size >= n
    cost: dict = field(default_factory=dict)        # verified size -> EWMA seconds
    cost_samples: dict = field(default_factory=dict)
    chosen: int | None = None


class TileBudgetController:
    """Choose proposal nodes per lane from counterfactual acceptance and cost."""

    def __init__(self, *, warmup=6, probe=3, refresh=48, alpha=0.25, margin=1.03):
        self.warmup, self.probe, self.refresh = int(warmup), int(probe), int(refresh)
        self.alpha, self.margin = float(alpha), float(margin)
        self._widths: dict[int, _Width] = {}
        self.decisions: dict[int, int] = {}

    def _state(self, width):
        return self._widths.setdefault(int(width), _Width())

    def choose(self, width, maximum):
        """Proposal nodes for one lane of a round at ``width`` lanes."""

        maximum = int(maximum)
        state = self._state(width)
        sizes = [n for n in CANDIDATES if n < maximum] + [maximum]
        if state.rounds < self.warmup or state.rounds % self.refresh == 0:
            return maximum
        probe = sizes[max(0, len(sizes) // 2 - 1)]
        if len(state.cost) < 2 and state.cost_samples.get(probe, 0) < self.probe:
            return probe
        fit = self._fit(state)
        if fit is None:
            return maximum
        a, b = fit
        best, best_rate = maximum, None
        for n in sizes:
            seen = state.samples.get(n, 0)
            if not seen:
                continue
            expected = state.tokens[n] / seen
            rate = expected / max(a + b * (n + 1), 1e-6)
            if best_rate is None or rate > best_rate * (self.margin if n < best else 1.0):
                best, best_rate = n, rate
        state.chosen = best
        self.decisions[int(width)] = best
        return best

    @staticmethod
    def _fit(state):
        points = [(n + 1, seconds) for n, seconds in state.cost.items()]
        if len(points) < 2:
            return None
        mean_x = sum(x for x, _ in points) / len(points)
        mean_y = sum(y for _, y in points) / len(points)
        var = sum((x - mean_x) ** 2 for x, _ in points)
        if var <= 0:
            return None
        slope = max(0.0, sum((x - mean_x) * (y - mean_y) for x, y in points) / var)
        return mean_y - slope * mean_x, slope

    def observe(self, width, nodes, path_ranks, seconds):
        """Record one lane's round: verified ``nodes``, committed node ranks."""

        state = self._state(width)
        state.rounds += 1
        ranks = sorted(int(rank) for rank in path_ranks)
        for n in sorted({n for n in CANDIDATES if n <= nodes} | {int(nodes)}):
            state.tokens[n] = state.tokens.get(n, 0) + 1 + sum(1 for rank in ranks if rank < n)
            state.samples[n] = state.samples.get(n, 0) + 1
        if seconds is not None and seconds > 0:
            previous = state.cost.get(nodes)
            state.cost[nodes] = (
                seconds if previous is None else (1 - self.alpha) * previous + self.alpha * seconds
            )
            state.cost_samples[nodes] = state.cost_samples.get(nodes, 0) + 1

    def receipt(self):
        return {
            str(width): {
                "rounds": state.rounds,
                "chosen": state.chosen,
                "cost_ms": {str(n): round(s * 1e3, 3) for n, s in sorted(state.cost.items())},
                "tokens_per_round": {
                    str(n): round(state.tokens[n] / state.samples[n], 3)
                    for n in sorted(state.samples)
                },
            }
            for width, state in sorted(self._widths.items())
        }


__all__ = ["CANDIDATES", "TileBudgetController"]
