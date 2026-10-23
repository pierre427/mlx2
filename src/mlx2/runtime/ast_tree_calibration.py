"""Per-depth drafter calibration for best-first tree tiles.

Original mlx2 implementation of the idea in Kapitsa's KFlash budget
(``src/model/kflash_budget.cpp`` ``KfBudget::fit``/``add_record`` and
``kflash.h`` ``kf_lcal``; Apache-2.0, provenance in
``provenance/kapitsa-ast-tree-calibration.json``).

A drafter assigns each tree node a probability ``q`` given its parent.  How
often the target accepts such a node depends on its depth: drafters tend to
be overconfident deeper in the tree.  This module learns, per depth ``d``,

    p_accept = sigmoid(alpha_d + beta_d * logit(q))

from the acceptance walk's own records (every node whose parent lies on the
committed path is a trial; it succeeded when it is on the committed path
too).  ``alpha = 0, beta = 1`` is the identity (``p = q``) and the prior the
fit is regularized toward.  The fit is an L2-regularized logistic regression
solved by a few Newton steps every ``refit_every`` observed trees over a
ring window of recent records per depth.

The calibration only decides which nodes a tree proposes and how many rows
it verifies.  The target's verification law is untouched, so the committed
tokens follow the target exactly as with any other tree.  Greedy and
sampled requests accept by different events and keep separate tables.
"""

from __future__ import annotations

import math

SCHEMA = "mlx2.ast-tree-calibration.v1"
MODES = ("order", "order_size")
MAX_DEPTH = 32
# ``kf_lcal`` clamps q below one so logit(q) stays finite.
_Q_MAX = 0.999999


def logit_q(log_q):
    """``logit(q)`` from ``log q`` (q clamped below one)."""

    log_q = min(float(log_q), 0.0)
    q = min(math.exp(log_q), _Q_MAX)
    return log_q - math.log1p(-q)


def calibrated_log_p(log_q, alpha, beta):
    """``log sigmoid(alpha + beta * logit(q))``, numerically stable."""

    z = float(alpha) + float(beta) * logit_q(log_q)
    # log sigmoid(z) = -softplus(-z)
    return -(math.log1p(math.exp(-z)) if z > -30.0 else -z)


def fit_logistic(xs, ys, alpha=0.0, beta=1.0, *, iters=4, lam=1.0):
    """L2-regularized logistic fit of ``y`` on ``(1, x)`` toward (0, 1).

    Newton steps from ``(alpha, beta)``; the prior weighs about as much as a
    few records.  ``beta`` stays positive (siblings keep the drafter's own
    ranking) and both are clamped to a sane range.
    """

    import numpy as np

    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    a, b = float(alpha), float(beta)
    for _ in range(int(iters)):
        p = 1.0 / (1.0 + np.exp(-np.clip(a + b * x, -30.0, 30.0)))
        w = p * (1.0 - p)
        r = p - y
        ga = lam * a + float(r.sum())
        gb = lam * (b - 1.0) + float((r * x).sum())
        haa = lam + float(w.sum())
        hab = float((w * x).sum())
        hbb = lam + float((w * x * x).sum())
        det = haa * hbb - hab * hab
        if not det > 1e-12:
            break
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        a = min(8.0, max(-8.0, a - da))
        b = min(4.0, max(0.1, b - db))
        if abs(da) + abs(db) < 1e-4:
            break
    return a, b


def tree_records(parents, depths, log_qs, commit_rows, *, accepted_rows=None):
    """Calibration trials of one verified tree.

    ``parents`` index the tree's nodes (-1: the anchor), ``commit_rows`` are
    verify rows (row 0 the anchor, row ``i + 1`` node ``i``).  Every node
    whose parent row was evaluated is a trial; its outcome is its own
    acceptance. ``accepted_rows`` may additionally name a confirmed terminal
    child whose input state was never committed or evaluated. Its descendants
    are censored rather than recorded as failed trials.
    """

    evaluated = {int(row) - 1 for row in commit_rows if int(row) > 0}
    accepted = evaluated if accepted_rows is None else {
        int(row) - 1 for row in accepted_rows if int(row) > 0
    }
    records = []
    for node, parent in enumerate(parents):
        parent = int(parent)
        if parent >= 0 and parent not in evaluated:
            continue
        records.append((int(depths[node]), float(log_qs[node]), node in accepted))
    return records


def expected_tokens(path_log_ps, nodes):
    """``E(n) = 1 + sum of the first n nodes' calibrated path probabilities``."""

    return 1.0 + sum(math.exp(value) for value in list(path_log_ps)[: int(nodes)])


def live_nodes(path_log_ps, cost, *, rate=None, minimum=1):
    """Kapitsa's live-row rule over a best-first tree's prefixes.

    ``cost(n)`` is the measured seconds of a round that verifies ``n``
    nodes (plus the anchor).  With a long-run ``rate`` (tokens per second)
    the rule maximizes ``E(n) - rate * cost(n)``: a row is kept when its
    expected tokens pay for the time it costs at the rate the route already
    achieves.  Without a rate it maximizes ``E(n) / cost(n)``.  A best-first
    tree's prefixes are parent-closed, so every ``n`` is a valid tree.
    """

    values = list(path_log_ps)
    if not values:
        return 0
    best_n, best_value = None, None
    running = 1.0
    for n in range(1, len(values) + 1):
        running += math.exp(values[n - 1])
        if n < int(minimum):
            continue
        seconds = float(cost(n))
        if not seconds > 0:
            continue
        value = running - rate * seconds if rate else running / seconds
        if best_value is None or value > best_value + 1e-12:
            best_n, best_value = n, value
    return len(values) if best_n is None else best_n


class _Table:
    __slots__ = ("alpha", "beta", "xs", "ys", "heads", "count", "since_fit", "trials", "hits")

    def __init__(self, depths):
        self.alpha = [0.0] * (depths + 1)
        self.beta = [1.0] * (depths + 1)
        self.xs = [[] for _ in range(depths + 1)]
        self.ys = [[] for _ in range(depths + 1)]
        self.heads = [0] * (depths + 1)
        self.count = 0
        self.since_fit = 0
        self.trials = [0] * (depths + 1)
        self.hits = [0] * (depths + 1)

    def copy(self):
        other = _Table.__new__(_Table)
        other.alpha, other.beta = list(self.alpha), list(self.beta)
        other.xs = [list(row) for row in self.xs]
        other.ys = [list(row) for row in self.ys]
        other.heads = list(self.heads)
        other.count, other.since_fit = self.count, self.since_fit
        other.trials, other.hits = list(self.trials), list(self.hits)
        return other


class DepthCalibration:
    """Per-depth acceptance calibration learned online from verified trees."""

    def __init__(self, mode="order", *, window=512, min_records=32, refit_every=16,
                 iters=4, max_depth=MAX_DEPTH, rate_decay=63.0 / 64.0):
        if mode not in MODES:
            raise ValueError(f"ast_tree_calibration must be one of {MODES}")
        self.mode = mode
        self.window, self.min_records = int(window), int(min_records)
        self.refit_every, self.iters = int(refit_every), int(iters)
        self.max_depth = int(max_depth)
        self.rate_decay = float(rate_decay)
        self._tables = {}
        self.trees = 0
        self.fits = 0
        self.predicted = 0.0
        self.realized = 0.0
        self.rate_tokens = 0.0
        self.rate_seconds = 0.0
        self.truncated_rounds = 0
        self.truncated_nodes = 0

    def _table(self, key):
        table = self._tables.get(key)
        if table is None:
            table = self._tables[key] = _Table(self.max_depth)
        return table

    def params(self, key="greedy"):
        """An immutable ``(alpha, beta)`` pair of per-depth tuples (index d)."""

        table = self._table(key)
        return tuple(table.alpha), tuple(table.beta)

    @property
    def sizes(self):
        return self.mode == "order_size"

    def rate(self):
        """Recent committed tokens per second (decayed sums), or None."""

        return self.rate_tokens / self.rate_seconds if self.rate_seconds > 0 else None

    def observe(self, key, records, *, predicted=None, realized=None):
        """Add one tree's ``(depth, log_q, hit)`` trials; refit when due."""

        table = self._table(key)
        for depth, log_q, hit in records:
            depth = int(depth)
            if not 1 <= depth <= self.max_depth:
                continue
            x, y = logit_q(log_q), 1.0 if hit else 0.0
            xs, ys = table.xs[depth], table.ys[depth]
            if len(xs) < self.window:
                xs.append(x)
                ys.append(y)
            else:
                head = table.heads[depth]
                xs[head], ys[head] = x, y
                table.heads[depth] = (head + 1) % self.window
            table.trials[depth] += 1
            table.hits[depth] += int(bool(hit))
        table.count += 1
        self.trees += 1
        if predicted is not None and realized is not None:
            self.predicted += float(predicted)
            self.realized += float(realized)
        table.since_fit += 1
        if table.since_fit >= self.refit_every:
            table.since_fit = 0
            for depth in range(1, self.max_depth + 1):
                if len(table.xs[depth]) >= self.min_records:
                    table.alpha[depth], table.beta[depth] = fit_logistic(
                        table.xs[depth], table.ys[depth],
                        table.alpha[depth], table.beta[depth], iters=self.iters,
                    )
            self.fits += 1

    def observe_rate(self, tokens, seconds):
        """Feed one round's committed tokens and wall seconds to the rate."""

        if seconds is None or not seconds > 0:
            return
        self.rate_tokens = self.rate_tokens * self.rate_decay + float(tokens)
        self.rate_seconds = self.rate_seconds * self.rate_decay + float(seconds)

    def snapshot(self):
        return (
            {key: table.copy() for key, table in self._tables.items()},
            (self.trees, self.fits, self.predicted, self.realized, self.rate_tokens,
             self.rate_seconds, self.truncated_rounds, self.truncated_nodes),
        )

    def restore(self, snapshot):
        tables, scalars = snapshot
        self._tables = {key: table.copy() for key, table in tables.items()}
        (self.trees, self.fits, self.predicted, self.realized, self.rate_tokens,
         self.rate_seconds, self.truncated_rounds, self.truncated_nodes) = scalars

    def receipt(self):
        tables = {}
        for key, table in sorted(self._tables.items()):
            depths = [d for d in range(1, self.max_depth + 1) if table.trials[d]]
            tables[key] = {
                "trees": table.count,
                "alpha": {str(d): round(table.alpha[d], 4) for d in depths},
                "beta": {str(d): round(table.beta[d], 4) for d in depths},
                "trials": {str(d): table.trials[d] for d in depths},
                "hit_rate": {str(d): round(table.hits[d] / table.trials[d], 4) for d in depths},
            }
        rate = self.rate()
        return {
            "schema": SCHEMA,
            "enabled": True,
            "mode": self.mode,
            "law": "sigmoid(alpha_d+beta_d*logit(q)); l2 newton toward (0,1)",
            "window": self.window,
            "min_records": self.min_records,
            "refit_every": self.refit_every,
            "trees": self.trees,
            "fits": self.fits,
            "predicted_tokens_per_tree": (
                round(self.predicted / self.trees, 4) if self.trees else None
            ),
            "realized_tokens_per_tree": (
                round(self.realized / self.trees, 4) if self.trees else None
            ),
            "rate_tokens_per_second": None if rate is None else round(rate, 3),
            "truncated_rounds": self.truncated_rounds,
            "truncated_nodes": self.truncated_nodes,
            "tables": tables,
        }


def parse_mode(value):
    """Policy value -> mode or None (off).  ``true`` means ``"order"``."""

    if value is None or value is False:
        return None
    if value is True:
        return "order"
    if value in MODES:
        return value
    raise ValueError(f"ast_tree_calibration must be false, true, or one of {MODES}")


__all__ = [
    "MODES",
    "SCHEMA",
    "DepthCalibration",
    "calibrated_log_p",
    "expected_tokens",
    "fit_logistic",
    "live_nodes",
    "logit_q",
    "parse_mode",
    "tree_records",
]
