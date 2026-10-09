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

The close marker may be a token sequence (GPT-OSS harmony closes analysis with
``<|end|><|start|>assistant<|channel|>final<|message|>``).  The release then
boosts the next marker token after whatever prefix of it the output already
ends with, and a forced close emits the rest of the marker token by token, as
``ThinkingBudgetProcessor`` does in history mode.

**Soft landing** (JUICE, the lab's Puzzle server): an adapter may declare a
short plain-text wrap-up nudge.  At the soft budget the guard writes it into
the reasoning, token by token, so the model concludes on its own before the
hard close.  Like the marker, which nudge token comes next is read from the
ids, and a budget too small to hold nudge and marker skips it.

The guard is a pure function of the generated ids, so speculative verify rows
and rollbacks need no bookkeeping from the caller.  Internally it is
incremental: each step reads only the tail of the token context and a rollback
undoes the alarm state position by position, so a step costs O(new tokens +
``rewrite_window``) rather than O(generated tokens).  The one contract that
buys: a caller only ever rewrites the last ``rewrite_window`` generated tokens
(speculative verify rows and rollbacks rewrite at most the draft depth).  It is off unless a request asks for a
``thinking_budget``; nothing about it is model-specific beyond the close ids the
adapter declares.

**Self-addressed reasoning** (Muse): a model that reasons in messages it
addresses to itself ends reasoning by opening a message to anybody else -- the
user's answer, a tool call, or, skipping reasoning, a first message to either.
Only the first writes the release marker, so such an adapter also declares its
message separator and self header (``reasoning_message``) and reasoning closes,
unforced, where the first other message begins (``reasoning_closed_at``).
Headers share their first tokens, so the recipient is the model's to choose:
the release stays out of every header the model writes, and a budget that
falls in one waits for it to end (``release_boundary``) -- forcing from there
when the message is to itself, nothing when it is to anybody else.
"""

from __future__ import annotations


def reasoning_closed_at(generated, separator, header, release=(), start=0):
    """``(close, decided)`` where self-addressed reasoning ended, or None.

    The reply's first message opens at index 0 and every later one after
    ``separator``; reasoning ends where the first message whose header departs
    from ``header`` begins (``close``: 0 or its separator), as the id at
    ``decided`` shows.  A message whose header is still a prefix of ``header``
    is undecided, and so is the ``release`` marker while it is being written:
    it closes once complete, so a budget that forces it finishes it.  A pure
    function of the ids; ``start`` skips decisions before it, which the caller
    has already examined and found open.
    """
    width, size = len(separator), len(header)

    def decide(at, opened):
        departed = None
        for offset, token in enumerate(generated[opened : opened + size]):
            if token != header[offset]:
                departed = opened + offset
                break
        if departed is None:
            return None
        written = generated[at : at + len(release)]
        for offset, token in enumerate(written):
            if token != release[offset]:
                return max(departed, at + offset)
        if len(written) < len(release):
            return None
        return max(departed, at + len(release) - 1)

    decided = decide(0, 0)
    if decided is not None:
        return 0, decided
    first = separator[0]
    index = max(0, start - max(width + size, len(release)) + 1)
    while True:
        try:
            index = generated.index(first, index)
        except ValueError:
            return None
        if tuple(generated[index : index + width]) == separator:
            decided = decide(index, index + width)
            if decided is not None:
                return index, decided
        index += 1


def open_header(generated, separator, header, length, window):
    """Where the header the first ``length`` ids end inside opened, or None.

    A message opens at 0 or right after ``separator``, and its header runs to
    the first ``header[-1]`` (the message start that ends every header).
    Until then its recipient is not decided: headers share their first ids
    (`` to``, and `` to=user`` also opens a tool named ``user_x``), so only
    the model can tell whom the message is for.  ``window`` bounds the look
    back; a longer header counts as none.
    """
    width, end = len(separator), header[-1]
    for opened in range(length, max(0, length - window) - 1, -1):
        if opened == 0 or (
            opened >= width and tuple(generated[opened - width : opened]) == separator
        ):
            return opened
        if generated[opened - 1] == end:
            return None
    return None


def release_boundary(generated, separator, header, budget, release):
    """The first generated index a reasoning budget may force, or None.

    The budget itself, unless the model is writing a message header there
    (``open_header``): the recipient is the model's choice.  Addressed to
    itself, the message is reasoning and the budget applies where its header
    ends; addressed to anybody else, reasoning ended unforced and nothing is
    forced (None, also while the header is still being written).  Until
    reasoning closes, an open header is a prefix of ``header`` or the
    ``release`` being written (``reasoning_closed_at``), which bounds the look
    back.
    """
    if len(generated) < budget:
        return budget  # nothing is forced before the budget anyway
    opened = open_header(
        generated, separator, header, budget, len(header) + len(release)
    )
    if opened is None:
        return budget
    try:
        end = generated.index(header[-1], opened) + 1
    except ValueError:
        return None
    return end if tuple(generated[opened:end]) == header else None


class ThinkingGuard:
    # P5: a pure function of the generated ids (see the module docstring).
    history_pure = True

    def __init__(self, prompt_length, close_ids, *, budget, soft_ratio=0.8, ramp_nats=2.0,
                 tau=2.5, ngram=6, direction=None, alpha=0.0, hammer=0.0,
                 rewrite_window=256, nudge_ids=(), reasoning_message=None):
        self.prompt_length = int(prompt_length)
        self.rewrite_window = max(1, int(rewrite_window))
        self.close_ids = tuple(int(token) for token in close_ids)
        if not self.close_ids:
            raise ValueError("the thinking guard needs a close marker")
        # (separator, self header) of self-addressed reasoning, or None.
        self.reasoning_message = (
            tuple(tuple(int(token) for token in part) for part in reasoning_message)
            if reasoning_message
            else None
        )
        if self.reasoning_message is not None and not all(self.reasoning_message):
            raise ValueError("self-addressed reasoning needs a separator and a header")
        self.nudge_ids = tuple(int(token) for token in nudge_ids)
        self.budget = None if budget is None else int(budget)
        self.soft = (
            None
            if self.budget is None
            else max(1, int(self.budget * float(soft_ratio)))
        )
        self.ramp_nats = float(ramp_nats)
        self.tau = float(tau)
        self.ngram = int(ngram)
        self._ids = []  # alarm input: generated ids before any close marker
        self._seen = {}
        self._cusum = 0.0
        self._tripped_at = None
        self._undo = []  # per _ids position: alarm state before advancing it
        self._generated = []  # every generated id seen, close marker included
        self._close_at = None  # index of the first close marker in _generated
        self._close_decided = None  # index of the id that decided _close_at
        self.trip_reason = None
        self.released_at = None
        self.forced = False
        self._forced_at = None
        self.think_tokens = 0
        # alpha actuator: an adapter-calibrated commit direction
        # {"layer": L, "vector": rms_L * v_hat}.  While the reasoning channel is
        # open the generator adds alpha * vector to that layer's residual for
        # this lane (hammer * vector once the alarm or soft budget has tripped).
        # It is a pull toward the model's own close decision, not a cut.
        self._direction = direction if (direction and alpha > 0) else None
        if self._direction is not None and len(self.close_ids) != 1:
            raise ValueError("residual steering needs a single-token close marker")
        self.alpha, self.hammer = float(alpha), float(hammer)
        self._open = True
        self.steered_steps = 0

    def residual_steer(self, input_token):
        """``(layer, vector)`` for the next decode step of this lane, or None."""
        request = self._steer_request(input_token)
        if request is not None:
            self.steered_steps += 1
        return request

    def _steer_request(self, input_token):
        if self._direction is None or not self._open:
            return None
        if int(input_token) == self.close_ids[0]:
            self._open = False
            return None
        strength = self.hammer if (self.hammer and self._tripped_at is not None) else self.alpha
        return self._direction["layer"], self._direction["vector"] * strength

    def residual_steer_block(self, context, inputs):
        """``residual_steer`` for every position of a speculative verify block.

        ``context`` holds the token ids before ``inputs[0]``.  Position ``j``
        gets exactly what the ordinary decode step feeding ``inputs[j]`` gets:
        the guard as that step's previous logits call left it (synced through
        ``inputs[j - 1]``), then asked about ``inputs[j]`` itself.  So rows
        before the first close token are steered (the anchor included), the
        close and every later row are not, and a hammer switches in at the
        position the alarm trips.  Nothing is counted here: a verify block can
        be rejected, so the caller reports the positions it committed through
        ``count_steered``.
        """
        if self._direction is None:
            return [None] * len(inputs)
        import numpy as np

        tokens = np.concatenate(
            [
                np.asarray(context, dtype=np.int64).reshape(-1),
                np.asarray(inputs, dtype=np.int64).reshape(-1),
            ]
        )
        base = tokens.size - len(inputs)
        requests = []
        for index, token in enumerate(inputs):
            self._observe(tokens[: base + index])
            requests.append(self._steer_request(token))
        return requests

    def count_steered(self, count):
        """Record ``count`` committed steered decode positions."""
        self.steered_steps += int(count)

    def _advance(self, token):
        """CUSUM run-on alarm: +1 for a recurring n-gram, -0.25 for a novel one."""
        undo = (self._cusum, self._tripped_at, self.trip_reason, None, None)
        self._ids.append(token)
        position = len(self._ids)
        if position >= self.ngram:
            gram = tuple(self._ids[-self.ngram:])
            previous = self._seen.get(gram)
            undo = undo[:3] + (gram, previous)
            self._seen[gram] = position
            self._cusum = max(0.0, self._cusum + (1.0 if previous is not None else -0.25))
        self._undo.append(undo)
        if self._tripped_at is None:
            if self._cusum >= self.tau * self.ngram:
                self._tripped_at, self.trip_reason = position, "run_on"
            elif self.soft is not None and position >= self.soft:
                self._tripped_at, self.trip_reason = position, "budget_soft"

    def _truncate(self, length):
        """Undo the alarm back to ``length`` ids, one position at a time."""
        while len(self._ids) > length:
            self._ids.pop()
            self._cusum, self._tripped_at, self.trip_reason, gram, previous = (
                self._undo.pop()
            )
            if gram is not None:
                if previous is None:
                    del self._seen[gram]
                else:
                    self._seen[gram] = previous

    def _sync(self, tokens):
        """Mirror the generated ids, reading only the rewritable tail."""
        length = max(0, tokens.size - self.prompt_length)
        known = self._generated
        start = max(0, min(len(known), length) - self.rewrite_window)
        begin = self.prompt_length + start
        tail = [int(item) for item in tokens[begin : self.prompt_length + length].tolist()]
        common = start
        for ours, theirs in zip(known[start:], tail):
            if ours != theirs:
                break
            common += 1
        del known[common:]
        known.extend(tail[common - start :])
        if self._close_at is not None and self._close_decided >= common:
            self._close_at = None
        if self._close_at is None:
            self._close_at, self._close_decided = self._first_close(known, common)
        if self._forced_at is not None and common < self._forced_at:
            self._forced_at = None
            self.forced = False
        self._truncate(min(len(self._ids), common))
        return length

    def _first_close(self, known, start):
        """``(index, decided)`` of the first close, or ``(None, None)``.

        The marker, or the end of self-addressed reasoning, whichever begins
        first.  Ids before ``start`` were already examined with reasoning
        open, so only a close that the ids from ``start`` on decide is new.
        """
        width = len(self.close_ids)
        close = self._find_close(known, max(0, start - width + 1))
        found = (None, None) if close is None else (close, close + width - 1)
        if self.reasoning_message is not None:
            ended = reasoning_closed_at(
                known, *self.reasoning_message, release=self.close_ids, start=start
            )
            if ended is not None and (found[0] is None or ended < found):
                found = ended
        return found

    def _find_close(self, known, start):
        """Index of the first complete close marker at or after ``start``."""
        marker, width = self.close_ids, len(self.close_ids)
        first = marker[0]
        for index in range(start, len(known) - width + 1):
            if known[index] == first and tuple(known[index : index + width]) == marker:
                return index
        return None

    def _marker_offset(self, length):
        """The longest proper close-marker prefix the first ``length`` ids end with.

        Both the release and a forced close continue the marker from here, so
        which marker token comes next is a function of the ids alone: a
        rollback or a stale speculative row needs no forcing state.
        """
        known, marker = self._generated, self.close_ids
        for width in range(min(len(marker) - 1, length), 0, -1):
            if tuple(known[length - width : length]) == marker[:width]:
                return width
        return 0

    def _nudge_start(self):
        """Where the soft-landing nudge goes, or None when it does not fit."""
        if not self.nudge_ids or self.budget is None or self.soft is None:
            return None
        if self.soft + len(self.nudge_ids) + len(self.close_ids) > self.budget:
            return None
        return self.soft

    def _nudge_target(self, length):
        """The nudge token the step after ``length`` ids writes, or None."""
        start = self._nudge_start()
        if start is None or not start <= length < start + len(self.nudge_ids):
            return None
        if tuple(self._generated[start:length]) != self.nudge_ids[: length - start]:
            return None  # a stale branch: the nudge is not being written here
        return self.nudge_ids[length - start]

    def _observe(self, tokens):
        """Sync the guard state to ``tokens``; the state half of ``__call__``."""
        length = self._sync(tokens)
        self._open = self._close_at is None
        self.released_at = self._close_at
        if self._close_at is not None:
            self.think_tokens = self._close_at
            return length
        for token in self._generated[len(self._ids):]:
            self._advance(token)
        self.think_tokens = length
        return length

    def _release_boundary(self):
        """The first generated index the budget may force (``release_boundary``)."""
        if self.reasoning_message is None or self.budget is None:
            return self.budget
        return release_boundary(
            self._generated, *self.reasoning_message, self.budget, self.close_ids
        )

    def _chooses(self, length):
        """Whether the step after ``length`` ids is the model's recipient choice.

        Self-addressed reasoning only.  Before the budget the release and the
        nudge stay out of every header (all are the model's); at the budget a
        header the model opened by then decides first (``release_boundary``),
        while one the forced release opened is finished by it.
        """
        if self.reasoning_message is None:
            return False
        if self.budget is not None and length >= self.budget:
            boundary = self._release_boundary()
            return boundary is None or length < boundary
        window = len(self.reasoning_message[1]) + len(self.close_ids)
        return open_header(self._generated, *self.reasoning_message, length, window) is not None

    def _forces(self, length):
        """Whether the step after ``length`` ids forces the close; latches it."""
        if self._close_at is not None or self._tripped_at is None:
            return False
        if self.budget is None or length < self.budget:
            return False
        self.forced = True
        if self._forced_at is None:
            # A marker prefix the model already wrote continues, not restarts.
            self._forced_at = length - self._marker_offset(length)
        return True

    def __call__(self, tokens, logits):
        import mlx.core as mx

        length = self._observe(tokens)
        if self._close_at is not None or self._tripped_at is None or self._chooses(length):
            return logits
        nudge = self._nudge_target(length)
        if nudge is not None and nudge < logits.shape[-1]:
            keep = mx.arange(logits.shape[-1]) == nudge
            return mx.where(keep, logits, mx.array(-float("inf"), dtype=logits.dtype))
        if self._forces(length):
            close = self.close_ids[self._marker_offset(length)]
            if close >= logits.shape[-1]:
                return logits
            keep = mx.arange(logits.shape[-1]) == close
            return mx.where(keep, logits, mx.array(-float("inf"), dtype=logits.dtype))
        close = self.close_ids[self._marker_offset(length)]
        if close >= logits.shape[-1]:
            return logits
        bias = self.ramp_nats * (length - self._tripped_at + 1)
        boost = mx.where(mx.arange(logits.shape[-1]) == close,
                         mx.array(bias, dtype=logits.dtype), mx.array(0.0, dtype=logits.dtype))
        return logits + boost

    def probe(self, tokens, logits):
        """``__call__`` for a provisional draft row, leaving the guard as it was.

        Draft probing otherwise deep-copies the guard, whose history, n-gram
        table and undo log grow with the generation: O(generated) per drafted
        position.  The call below changes the ids only from the sync window
        on, so the guard restores that tail, undoes the alarm back to the
        first position the call changed and replays the ids that were there.
        """
        generated, ids = self._generated, self._ids
        ids_length = len(ids)
        length = max(0, int(tokens.size) - self.prompt_length)
        start = max(0, min(len(generated), length) - self.rewrite_window)
        tail = generated[start:]
        scalars = (
            self._close_at, self._close_decided, self.released_at, self.forced,
            self._forced_at, self.think_tokens, self._open,
        )
        try:
            return self(tokens, logits)
        finally:
            del generated[start:]
            generated.extend(tail)
            (
                self._close_at, self._close_decided, self.released_at, self.forced,
                self._forced_at, self.think_tokens, self._open,
            ) = scalars
            # Alarm positions before ``start`` were never truncated.
            index = min(start, ids_length)
            limit = min(len(ids), ids_length)
            while index < limit and ids[index] == generated[index]:
                index += 1
            self._truncate(index)
            for token in generated[index:ids_length]:
                self._advance(token)

    @property
    def resyncs_after_rollback(self):
        """Whether a rollback needs no snapshot of this guard.

        Everything but the steering count is a function of the ids the next
        call passes, so a recovery snapshot can share the live guard.  A
        steering guard counts committed positions, which the ids do not
        determine, and is still copied.
        """
        return self._direction is None

    def settle(self, generated):
        """Re-sync the receipt state to the committed generated ids.

        Speculative routes call the guard on verify rows that are never
        committed (drafts past a stop token, rejected rows) and prompt
        lookup never calls it for the final token.  Ordinary decode's last
        call sees exactly the committed ids, so settling to them makes every
        route's receipt describe the committed stream.

        That last call is a lookahead, though: it evaluates the row after the
        final committed token, which is never sampled.  So whether a close
        was forced is read from the committed ids themselves, not from the
        rows the guard evaluated.
        """
        import numpy as np

        tokens = np.concatenate(
            [
                np.zeros(self.prompt_length, dtype=np.int64),
                np.asarray(list(generated), dtype=np.int64),
            ]
        )
        self._observe(tokens)
        close_at = self._close_at
        if close_at is not None:
            # Ordinary decode observed every id before the close, one step at
            # a time; a speculative route may have skipped some of them.
            for token in self._generated[len(self._ids) : close_at]:
                self._advance(token)
        # A forcing row admits only the next marker token, so a forced close
        # ends at or past the budget while a natural one ends before it: a
        # marker whose last token is at or past the budget, with the alarm
        # tripped by then, is exactly a forced one.  Self-addressed reasoning
        # that ended in another message was never forced, and a recipient
        # the model was choosing at the budget moves it (``release_boundary``).
        width = len(self.close_ids)
        boundary = self._release_boundary()
        self.forced = (
            boundary is not None
            and close_at is not None
            and close_at + width - 1 >= boundary
            and self._tripped_at is not None
            and tuple(self._generated[close_at : close_at + width]) == self.close_ids
        )
        self._forced_at = close_at if self.forced else None

    def dormant(self, tokens):
        """P5: True once reasoning has closed (logits pass through).

        Side-effect free; before the marker the alarm may bias at any step,
        so the guard is never reported dormant there.
        """
        generated = tokens[self.prompt_length :]
        if hasattr(generated, "tolist"):
            generated = generated.tolist()
        generated = [int(item) for item in generated]
        return self._find_close_in(generated) or (
            self.reasoning_message is not None
            and reasoning_closed_at(
                generated, *self.reasoning_message, release=self.close_ids
            )
            is not None
        )

    def _find_close_in(self, generated):
        marker, width = self.close_ids, len(self.close_ids)
        return any(
            tuple(generated[index : index + width]) == marker
            for index in range(len(generated) - width + 1)
        )

    def _nudged(self):
        """Whether the committed ids hold the whole soft-landing nudge."""
        start = self._nudge_start()
        if start is None:
            return False
        end = start + len(self.nudge_ids)
        return (tuple(self._generated[start:end]) == self.nudge_ids
                and (self._close_at is None or self._close_at >= end))

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
            "nudged": self._nudged(),
            "steering": (
                {"layer": self._direction["layer"], "alpha": self.alpha, "hammer": self.hammer,
                 "steered_steps": self.steered_steps, "source": self._direction.get("source")}
                if self._direction is not None else None
            ),
        }


def verify_block_steer(processors, context, inputs):
    """Per-position residual steering for one lane's speculative verify block.

    Returns ``(requests, commit)``.  ``requests[j]`` is the ``(layer, vector)``
    the ordinary route applies to the decode step that feeds ``inputs[j]``, or
    None; ``context`` holds the token ids before ``inputs[0]``.  As in the
    ordinary route, the last processor that asks wins a position.
    ``commit(consumed)`` counts the steered positions among the ``consumed``
    inputs the round committed.  A processor without ``residual_steer_block``
    is asked position by position, as the ordinary route asks it every step.
    """
    requests = [None] * len(inputs)
    counted = []
    for processor in processors or ():
        block = getattr(processor, "residual_steer_block", None)
        if block is not None:
            rows = block(context, inputs)
            counted.append((processor, rows))
        else:
            ask = getattr(processor, "residual_steer", None)
            if ask is None:
                continue
            rows = [ask(token) for token in inputs]
        for index, request in enumerate(rows):
            if request is not None:
                requests[index] = request

    def commit(consumed):
        for processor, rows in counted:
            processor.count_steered(
                sum(request is not None for request in rows[: int(consumed)])
            )

    return requests, commit


def stack_block_steer(lane_requests, width):
    """``(layer, [B, width, D])`` from per-lane position requests, or None.

    One steering layer per forward, as in the ordinary route: the first
    request fixes it and a request for another layer is dropped.  Unsteered
    positions and padding get zero rows, so one forward serves them all.
    """
    import mlx.core as mx

    layer, sample, kept = None, None, []
    for requests in lane_requests:
        row = []
        for request in requests:
            if request is not None and (layer is None or request[0] == layer):
                layer, sample = request[0], request[1]
                row.append(request[1])
            else:
                row.append(None)
        kept.append(row)
    if layer is None:
        return None
    zero = mx.zeros(sample.shape, dtype=sample.dtype)
    return layer, mx.stack(
        [
            mx.stack(
                [
                    row[index] if index < len(row) and row[index] is not None else zero
                    for index in range(width)
                ]
            )
            for row in kept
        ]
    )


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
