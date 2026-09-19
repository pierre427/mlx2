"""State-aware release of a run-on reasoning channel.

A reasoning model can circle a formed answer without ever emitting its
thinking-close token: the close token stays a high-ranked alternative but never
wins probability mass.  A static bias cannot fix that without truncating hard
problems, so the guard is state-aware:

* **budget** (JUICE): a reasoning-token budget; at ``soft_ratio`` of it the
  release begins, at the budget the channel is closed outright;
* **run-on alarm** (tau): a CUSUM over content-blind repetition evidence that
  releases early when the reasoning is looping;
* **release** (the actuator): a bias on the close token ramped by
  ``ramp_nats`` per step, so the model winds down over a few tokens rather than
  being cut mid-thought.

The guard is a pure function of the generated ids, so speculative verify rows
and rollbacks need no bookkeeping.  It is off unless a request asks for a
``thinking_budget``; nothing about it is model-specific beyond the close ids the
adapter declares.
"""

from __future__ import annotations


class ThinkingGuard:
    def __init__(self, prompt_length, close_ids, *, budget, soft_ratio=0.8, ramp_nats=2.0,
                 tau=2.5, ngram=6, direction=None, alpha=0.0, hammer=0.0):
        self.prompt_length = int(prompt_length)
        self.close_ids = tuple(int(token) for token in close_ids)
        if len(self.close_ids) != 1:
            raise ValueError("the thinking guard needs a single-token close marker")
        self.budget = None if budget is None else int(budget)
        self.soft = (
            None
            if self.budget is None
            else max(1, int(self.budget * float(soft_ratio)))
        )
        self.ramp_nats = float(ramp_nats)
        self.tau = float(tau)
        self.ngram = int(ngram)
        self._ids = []
        self._seen = {}
        self._cusum = 0.0
        self._tripped_at = None
        self.trip_reason = None
        self.released_at = None
        self.forced = False
        self.think_tokens = 0
        # alpha actuator: an adapter-calibrated commit direction
        # {"layer": L, "vector": rms_L * v_hat}.  While the reasoning channel is
        # open the generator adds alpha * vector to that layer's residual for
        # this lane (hammer * vector once the alarm or soft budget has tripped).
        # It is a pull toward the model's own close decision, not a cut.
        self._direction = direction if (direction and alpha > 0) else None
        self.alpha, self.hammer = float(alpha), float(hammer)
        self._open = True
        self.steered_steps = 0

    def residual_steer(self, input_token):
        """``(layer, vector)`` for the next decode step of this lane, or None."""
        if self._direction is None or not self._open:
            return None
        if int(input_token) == self.close_ids[0]:
            self._open = False
            return None
        strength = self.hammer if (self.hammer and self._tripped_at is not None) else self.alpha
        self.steered_steps += 1
        return self._direction["layer"], self._direction["vector"] * strength

    def _reset(self):
        self._ids, self._seen, self._cusum = [], {}, 0.0
        self._tripped_at = self.trip_reason = None

    def _advance(self, token):
        """CUSUM run-on alarm: +1 for a recurring n-gram, -0.25 for a novel one."""
        self._ids.append(token)
        position = len(self._ids)
        if position >= self.ngram:
            gram = tuple(self._ids[-self.ngram:])
            repeated = gram in self._seen
            self._seen[gram] = position
            self._cusum = max(0.0, self._cusum + (1.0 if repeated else -0.25))
        if self._tripped_at is None:
            if self._cusum >= self.tau * self.ngram:
                self._tripped_at, self.trip_reason = position, "run_on"
            elif self.soft is not None and position >= self.soft:
                self._tripped_at, self.trip_reason = position, "budget_soft"

    def _track(self, ids):
        common = 0
        for ours, theirs in zip(self._ids, ids):
            if ours != theirs:
                break
            common += 1
        if common < len(self._ids):  # rollback or divergence: re-derive
            self._reset()
            common = 0
        for token in ids[common:]:
            self._advance(token)

    def __call__(self, tokens, logits):
        import mlx.core as mx

        ids = [int(item) for item in tokens.tolist()][self.prompt_length:]
        close = self.close_ids[0]
        if close in ids:
            self._open = False
            if self.released_at is None:
                self.released_at = ids.index(close)
                self.think_tokens = self.released_at
            return logits
        self._track(ids)
        self.think_tokens = len(ids)
        if self._tripped_at is None or close >= logits.shape[-1]:
            return logits
        if self.budget is not None and len(ids) >= self.budget:
            self.forced = True
            keep = mx.arange(logits.shape[-1]) == close
            return mx.where(keep, logits, mx.array(-float("inf"), dtype=logits.dtype))
        bias = self.ramp_nats * (len(ids) - self._tripped_at + 1)
        boost = mx.where(mx.arange(logits.shape[-1]) == close,
                         mx.array(bias, dtype=logits.dtype), mx.array(0.0, dtype=logits.dtype))
        return logits + boost

    def receipt(self):
        return {
            "schema": "mlx2.thinking-guard.v1",
            "budget": self.budget,
            "soft_budget": self.soft,
            "think_tokens": self.think_tokens,
            "tripped": self.trip_reason,
            "tripped_at": self._tripped_at,
            "released_at": self.released_at,
            "forced_close": self.forced,
            "steering": (
                {"layer": self._direction["layer"], "alpha": self.alpha, "hammer": self.hammer,
                 "steered_steps": self.steered_steps, "source": self._direction.get("source")}
                if self._direction is not None else None
            ),
        }


# JUICE: the reasoning budget scales with the requested effort around a
# server-configured medium anchor (same ratios as the lab's Puzzle server:
# low 0.375x, medium 1x, high 4x).  An explicit request budget always wins.
EFFORT_SCALE = {"minimal": 0.125, "low": 0.375, "medium": 1.0, "high": 4.0,
                "xhigh": 6.0, "max": 8.0, "ultra": 8.0}
MAX_THINKING_BUDGET = 8192


def resolve_thinking_budget(request, anchor):
    """Reasoning-token budget for a request, or 0 for no guard."""
    explicit = request.get("thinking_budget")
    if explicit is not None:
        return int(explicit)  # an explicit 0 turns the guard off for this request
    if not anchor:
        return 0
    effort = str(request.get("reasoning_effort") or "high").lower()
    scale = EFFORT_SCALE.get(effort, EFFORT_SCALE["high"])
    return max(64, min(MAX_THINKING_BUDGET, int(anchor * scale)))
