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

Cost estimate (``cost_mode``).  ``"ewma"`` (the default) keeps one
exponentially weighted mean per verified size.  It is noise-sensitive: the
first sample at a new verify shape pays Metal pipeline compilation, a size
the chooser stops using (the cost probe, or the maximum between refreshes)
is updated so rarely that one such sample stays in its mean for hundreds of
rounds, and an inflated small-size point turns the least-squares slope
negative, which the fit clamps to a flat cost (always the maximum) while an
inflated maximum makes small tiles look fast.  ``"robust"`` (opt-in) drops
the first ``cost_skip`` samples at every size, estimates each size's cost as
the median of its last ``cost_window`` kept samples, and fits only sizes with
at least ``min_cost_samples`` kept samples.  The serving route pairs it with
the round-period cost scope (round start to the next round's start), so the
per-round host work after the commit counts against small tiles too.  A round
whose period is not clean (interleaved cohorts, a prefill slice) reports its
start-through-commit seconds instead.  The two scopes keep separate windows
and the fit uses one scope only: the period whenever it covers two sizes,
else the commit scope, so a cheaper commit-scope probe never meets
period-scope maxima.  The cost probe continues (bounded) until the fit's
scope covers it.

Robust mode also waits out the process warm-up: on Qwen3.8-27B + DFlash2
(2026-11-12 trace) every size ran 110-135 ms per round for the first ~20
rounds against ~97 ms steady, and the one-shot counterfactual probe landed
exactly there, so the probed size kept a warm-up cost for the whole run.
Cost samples before ``cost_warmup_rounds`` are dropped, the probe starts
after them, and the probe size is re-sampled once per refresh epoch so its
window tracks the steady state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

CANDIDATES = (3, 5, 7, 9, 11, 15, 23, 31)
# "model" (opt-in, deterministic controller mode, 2026-11-23): round cost is the fixed ``a + b * rows`` of the
# ``cost_model`` given at construction (rows = lanes x (nodes + 1)), never a measured time, and no cost probe runs:
# the choice depends only on the counterfactual acceptance history, so the same token history gives the same sizes.
COST_MODES = ("ewma", "robust", "model")
COST_SCOPES = ("period", "commit")


@dataclass
class _Width:
    rounds: int = 0
    tokens: dict = field(default_factory=dict)      # n -> summed counterfactual tokens
    samples: dict = field(default_factory=dict)     # n -> rounds observed with size >= n
    cost: dict = field(default_factory=dict)        # verified size -> EWMA seconds
    cost_samples: dict = field(default_factory=dict)
    chosen: int | None = None
    maximum: int | None = None                      # the knob's node maximum
    verified: dict = field(default_factory=dict)    # verified nodes -> lane rounds
    pending_choice: tuple | None = None             # immutable within one observed round
    probe_epoch: int = -1                          # measured-yield refresh already probed
    window: dict = field(default_factory=dict)      # robust: (scope, size) -> recent kept seconds
    skipped: dict = field(default_factory=dict)     # robust: verified size -> warm-up samples dropped
    warm_dropped: int = 0                           # robust: samples before cost_warmup_rounds


class TileBudgetController:
    """Choose proposal nodes per lane from counterfactual acceptance and cost."""

    def __init__(self, *, warmup=6, probe=3, refresh=48, alpha=0.25, margin=1.08,
                 counterfactual=True, cost_mode="ewma", cost_skip=2, cost_window=32,
                 min_cost_samples=2, cost_warmup_rounds=24, trace=None, cost_model=None):
        self.warmup, self.probe, self.refresh = int(warmup), int(probe), int(refresh)
        self.alpha, self.margin = float(alpha), float(margin)
        self.counterfactual = bool(counterfactual)
        if cost_mode not in COST_MODES:
            raise ValueError(f"cost_mode must be one of {COST_MODES}")
        self.cost_mode = cost_mode
        if (cost_mode == "model") != (cost_model is not None):
            raise ValueError('cost_model (a_seconds, b_seconds_per_row) goes with cost_mode "model" only')
        if cost_model is not None:
            a, b = (float(v) for v in cost_model)
            if not (a > 0 and b >= 0):
                raise ValueError("cost_model needs a > 0 seconds and b >= 0 seconds per row")
            cost_model = (a, b)
        self.cost_model = cost_model
        self.cost_skip, self.cost_window = int(cost_skip), int(cost_window)
        self.min_cost_samples = int(min_cost_samples)
        self.cost_warmup_rounds = int(cost_warmup_rounds)
        if self.cost_skip < 0 or self.cost_warmup_rounds < 0 or self.cost_window < 1 or self.min_cost_samples < 1:
            raise ValueError(
                "cost_skip >= 0, cost_warmup_rounds >= 0, cost_window >= 1 "
                "and min_cost_samples >= 1"
            )
        # Diagnostic sink: called with (width, round, verified nodes, cost ms,
        # chosen before this sample) for every positive cost sample.
        self.trace = trace
        self._widths: dict[int, _Width] = {}
        self.decisions: dict[int, int] = {}

    def _state(self, width):
        return self._widths.setdefault(int(width), _Width())

    def reset(self):
        """Forget every learned acceptance and cost (deterministic mode: a request on an idle engine)."""
        self._widths = {}
        self.decisions = {}

    def _model_fit(self, width):
        a, b = self.cost_model
        return a, b * int(width)                       # seconds = a + (b * lanes) * (nodes + 1)

    def choose(self, width, maximum):
        """Proposal nodes for one lane of a round at ``width`` lanes."""

        state = self._state(width)
        key = (state.rounds, int(maximum))
        if state.pending_choice is None or state.pending_choice[:2] != key:
            state.pending_choice = (*key, self._choose(width, maximum))
        return state.pending_choice[2]

    def _choose(self, width, maximum):

        maximum = int(maximum)
        state = self._state(width)
        state.maximum = maximum
        sizes = [n for n in CANDIDATES if n < maximum] + [maximum]
        if state.rounds < self.warmup or state.rounds % self.refresh == 0:
            return maximum
        robust = self.cost_mode == "robust"
        if robust and state.rounds < self.cost_warmup_rounds:
            return maximum                  # costs are not sampled yet
        if self.cost_mode == "model":
            # Costs are known (no cost probe, no measured fit). Without the prefix counterfactual this also skips
            # the yield probes, so only the maximum gathers samples and the choice stays at the maximum.
            pass
        elif not self.counterfactual:
            # Rebuilt grafts cannot infer smaller-budget yields from a large
            # tree. Measure each candidate before comparing its expected rate.
            epoch = state.rounds // self.refresh
            for size in sizes:
                seen = state.samples.get(size, 0)
                if seen < self.probe or (
                    # Robust: until its cost has enough kept samples (bounded).
                    robust and seen < 4 * self.probe
                    and state.cost_samples.get(size, 0) < self.min_cost_samples
                ):
                    state.probe_epoch = epoch
                    return size
            # Keep learning when copy opportunities change. One rotating
            # smaller-budget probe per refresh epoch also works when cohort
            # observations advance the round count by more than one.
            if sizes[:-1] and epoch > state.probe_epoch:
                state.probe_epoch = epoch
                return sizes[:-1][epoch % (len(sizes) - 1)]
        else:
            probe = sizes[max(0, len(sizes) // 2 - 1)]
            taken = state.cost_samples.get(probe, 0)
            if probe != maximum and (
                taken < self.probe
                # Robust: keep probing (bounded) until the fit's scope also
                # covers the probe, else the fit stays one-sized and flat.
                or (robust and taken < 4 * self.probe
                    and probe not in self._costs(state))
            ):
                return probe
            if robust and probe != maximum:
                # One fresh probe sample per later refresh epoch.
                epoch = state.rounds // self.refresh
                if state.probe_epoch < 0:
                    state.probe_epoch = epoch       # the initial burst's epoch
                elif epoch > state.probe_epoch:
                    state.probe_epoch = epoch
                    return probe
        fit = self._model_fit(width) if self.cost_mode == "model" else self._fit(state)
        if fit is None:
            return maximum
        a, b = fit
        rates = {}
        for n in sizes:
            seen = state.samples.get(n, 0)
            if not seen:
                continue
            expected = state.tokens[n] / seen
            rates[n] = expected / max(a + b * (n + 1), 1e-6)
        if maximum not in rates:
            return maximum
        # The reduction margin is relative to the configured maximum once,
        # not compounded at every intermediate candidate. Among candidates
        # clearing it, use the best expected rate (prefer larger on ties).
        best, best_rate = maximum, rates[maximum]
        threshold = best_rate * self.margin
        for n in reversed(sizes):
            rate = rates.get(n, 0.0)
            if rate > threshold and rate > best_rate:
                best, best_rate = n, rate
        state.chosen = best
        self.decisions[int(width)] = best
        return best

    @staticmethod
    def _median(values):
        ordered = sorted(values)
        middle = len(ordered) // 2
        return ordered[middle] if len(ordered) % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])

    def _scope_costs(self, state, scope):
        """Robust: size -> median seconds over sizes with enough kept samples."""

        return {
            n: self._median(values) for (kind, n), values in state.window.items()
            if kind == scope and len(values) >= self.min_cost_samples
        }

    def _fit_scope(self, state):
        """Robust: the one cost scope the fit uses (None before any sample)."""

        covered = {scope: len(self._scope_costs(state, scope)) for scope in COST_SCOPES}
        for scope in COST_SCOPES:          # the period first, whenever it can fit
            if covered[scope] >= 2:
                return scope
        best = max(COST_SCOPES, key=lambda scope: covered[scope])  # period wins ties
        return best if covered[best] else None

    def _costs(self, state):
        if self.cost_mode == "ewma":
            return state.cost
        if self.cost_mode == "model":
            return {}
        scope = self._fit_scope(state)
        return {} if scope is None else self._scope_costs(state, scope)

    def _fit(self, state):
        points = [(n + 1, seconds) for n, seconds in self._costs(state).items()]
        if len(points) < 2:
            return None
        mean_x = sum(x for x, _ in points) / len(points)
        mean_y = sum(y for _, y in points) / len(points)
        var = sum((x - mean_x) ** 2 for x, _ in points)
        if var <= 0:
            return None
        slope = max(0.0, sum((x - mean_x) * (y - mean_y) for x, y in points) / var)
        return mean_y - slope * mean_x, slope

    def cost_curve(self, width):
        """``(a, b)`` of the fitted round seconds ``a + b * rows``, or None."""

        if self.cost_mode == "model":
            return self._model_fit(width)
        state = self._widths.get(int(width))
        return None if state is None else self._fit(state)

    def observe(self, width, nodes, path_ranks, seconds):
        """Record one lane's round: verified ``nodes``, committed node ranks.

        ``seconds=None`` records acceptance only (a lane's first rounds also
        pay its prompt ingest into the draft plane, which is not tile cost).
        """

        state = self._state(width)
        state.rounds += 1
        state.pending_choice = None
        state.verified[int(nodes)] = state.verified.get(int(nodes), 0) + 1
        ranks = sorted(int(rank) for rank in path_ranks)
        sizes = ({n for n in CANDIDATES if n <= nodes} if self.counterfactual else set()) | {int(nodes)}
        for n in sorted(sizes):
            state.tokens[n] = state.tokens.get(n, 0) + 1 + sum(1 for rank in ranks if rank < n)
            state.samples[n] = state.samples.get(n, 0) + 1
        self.observe_cost(width, nodes, seconds)

    def observe_cost(self, width, nodes, seconds, scope="commit"):
        """Complete a deferred round cost without counting acceptance twice.

        A prelaunched lattice keeps its chosen budget until the next accepted
        observation, even when commit-time cost completes after prelaunch.
        ``scope`` (robust mode) names what ``seconds`` spans: ``"commit"``
        (round start through commit) or ``"period"`` (start to next start).
        """
        if scope not in COST_SCOPES:
            raise ValueError(f"scope must be one of {COST_SCOPES}")
        state = self._state(width)
        if self.cost_mode == "model" or seconds is None or not seconds > 0:
            return                              # model mode: measured time never enters the state
        nodes = int(nodes)
        if self.trace is not None:
            self.trace((int(width), state.rounds, nodes, round(seconds * 1e3, 3), state.chosen))
        if self.cost_mode == "robust":
            if state.rounds < self.cost_warmup_rounds:
                state.warm_dropped += 1     # process warm-up, every size
                return
            # The first samples at a size pay its first-use compilation and
            # warm-up; the median of recent samples ignores isolated stalls.
            if state.skipped.get(nodes, 0) < self.cost_skip:
                state.skipped[nodes] = state.skipped.get(nodes, 0) + 1
                return
            window = state.window.setdefault((scope, nodes), [])
            window.append(float(seconds))
            del window[: -self.cost_window]
        else:
            previous = state.cost.get(nodes)
            state.cost[nodes] = (
                seconds if previous is None else (1 - self.alpha) * previous + self.alpha * seconds
            )
        state.cost_samples[nodes] = state.cost_samples.get(nodes, 0) + 1

    def snapshot(self):
        """An independent copy of the learned state, for round rollback."""

        return (
            {width: self._copy(state) for width, state in self._widths.items()},
            dict(self.decisions),
        )

    @staticmethod
    def _copy(state):
        return replace(
            state, tokens=dict(state.tokens), samples=dict(state.samples),
            cost=dict(state.cost), cost_samples=dict(state.cost_samples),
            verified=dict(state.verified),
            window={n: list(values) for n, values in state.window.items()},
            skipped=dict(state.skipped),
        )

    def restore(self, snapshot):
        """Return to a ``snapshot()``; the snapshot itself stays reusable."""

        widths, decisions = snapshot
        self._widths = {width: self._copy(state) for width, state in widths.items()}
        self.decisions = dict(decisions)

    def _fit_receipt(self, state, width=1):
        fit = self._model_fit(width) if self.cost_mode == "model" else self._fit(state)
        if fit is None:
            return None
        return {"intercept_ms": round(fit[0] * 1e3, 4), "slope_ms_per_row": round(fit[1] * 1e3, 4)}

    def receipt(self):
        """Per lane width: plain numbers and short strings (metrics-ready).

        ``tile_rows`` is the knob's tile (maximum nodes + anchor: 16 or 8 at
        B1), ``mode`` says the size is chosen adaptively within it, and
        ``verified_nodes`` counts lane rounds by the node count verified.
        """
        receipt = {
            str(width): {
                "mode": "adaptive",
                "cost_mode": self.cost_mode,
                "yield_observation": "prefix_counterfactual" if self.counterfactual else "verified_budget_only",
                "maximum_nodes": state.maximum,
                "tile_rows": None if state.maximum is None else state.maximum + 1,
                "verified_nodes": {str(n): c for n, c in sorted(state.verified.items())},
                "fit": self._fit_receipt(state, width),
                "rounds": state.rounds,
                "chosen": state.chosen,
                "cost_ms": {str(n): round(s * 1e3, 3) for n, s in sorted(self._costs(state).items())},
                "cost_samples": {str(n): c for n, c in sorted(state.cost_samples.items())},
                "tokens_per_round": {
                    str(n): round(state.tokens[n] / state.samples[n], 3)
                    for n in sorted(state.samples)
                },
            }
            for width, state in sorted(self._widths.items())
        }
        if self.cost_mode == "robust":
            for width, state in self._widths.items():
                entry = receipt[str(width)]
                entry["cost_skipped"] = {str(n): c for n, c in sorted(state.skipped.items())}
                entry["cost_warmup_dropped"] = state.warm_dropped
                entry["fit_scope"] = self._fit_scope(state)
                entry["cost_samples_by_scope"] = {
                    scope: {
                        str(n): len(values)
                        for (kind, n), values in sorted(state.window.items(), key=lambda item: item[0][1])
                        if kind == scope
                    }
                    for scope in COST_SCOPES
                }
        return receipt


__all__ = ["CANDIDATES", "COST_MODES", "TileBudgetController"]
