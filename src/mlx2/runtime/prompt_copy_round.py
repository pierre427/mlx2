"""Prompt copy as a round replacement, with probation and backoff (opt-in).

Adapted from MTPLX (https://github.com/youssofal/MTPLX, Apache-2.0, revision
3890be556160af48887d5bfbf5304c4ad5b9417f): ``mtplx/models/inkling/copy_draft.py``
(``PromptCopy.propose``/``observe``: probation cap, acceptance EMA,
suspension with exponential backoff) and ``mtplx/context_copy.py``
(``NgramIndex``, ``block_for_ext``, the confidence ladder).  Re-implemented
for mlx2's external draft/verify engine; see provenance/mtplx-device-round.json.
The cost gate of ``mtplx/copy_gate.py`` is not ported (it is off by default
there too); its veto would sit in ``PromptCopyRuntime.propose``.

How it differs from mlx2's existing copy composition
(``proposal_composition``, ``ComposedDraftModel`` + ``prompt_lookup``):
composition scores prompt-lookup rows against the drafter's chain every
round, with n-grams 3-6 over a 256-token lookback and spans capped at
``num_draft`` (7 unless ``prompt_lookup_max_span``), and always pays the
drafter.  This mode is either/or per round, as MTPLX does it: when the
prompt continues the current suffix by an n-gram of ``ngram_min`` (default
6) or more, the round verifies a copy of up to ``max_tokens`` (default 24)
prompt tokens and runs no drafter; otherwise the drafter runs as usual.  A
lane whose copies stop landing (acceptance EMA below ``ema_floor`` after
three copy rounds) is suspended for ``backoff_min`` tokens, doubling to
``backoff_max``, so content without verbatim reuse (GSM8K) pays at most a
few probation rounds.  Copies are deterministic point masses verified by
the exact target rule, so the output law is unchanged.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import asdict, dataclass
from types import SimpleNamespace

POLICY_KEY = "prompt_copy_round"
# Block length by backward match extension beyond ``ngram_min``
# (context_copy._BLOCK_LADDER): a weak match risks only a short verify.
LADDER = (8, 12, 16, 24, 32)

COUNTERS = (
    "external_prompt_copy_probes",
    "external_prompt_copy_rounds",
    "external_prompt_copy_lanes",
    "external_prompt_copy_proposed_tokens",
    "external_prompt_copy_accepted_tokens",
    "external_prompt_copy_suspensions",
)
DECLINE_REASONS = "external_prompt_copy_decline_reasons"


@dataclass(frozen=True)
class PromptCopyPolicy:
    ngram_min: int = 6
    ngram_max: int = 10
    max_tokens: int = 24
    probation_tokens: int = 8
    min_tokens: int = 1
    ema_floor: float = 0.35
    backoff_min: int = 64
    backoff_max: int = 4096
    max_candidates: int = 32

    @classmethod
    def from_value(cls, value):
        if value is None or value is False:
            return None
        if value is True:
            return cls()
        if not isinstance(value, dict):
            raise ValueError(f"{POLICY_KEY} must be true or an object")  # noqa: TRY004
        value = dict(value)
        enabled = value.pop("enabled", True)
        if type(enabled) is not bool:
            raise ValueError(f"{POLICY_KEY}.enabled must be a boolean")
        fields = set(cls.__dataclass_fields__)
        unknown = set(value) - fields
        if unknown:
            raise ValueError(f"unknown {POLICY_KEY} keys: {sorted(unknown)}")
        for key, item in value.items():
            if key == "ema_floor":
                if isinstance(item, bool) or not isinstance(item, (int, float)) or not 0 <= item <= 1:
                    raise ValueError(f"{POLICY_KEY}.ema_floor must be in [0, 1]")
            elif type(item) is not int or item < 1:
                raise ValueError(f"{POLICY_KEY}.{key} must be a positive integer")
        policy = cls(**value)
        if policy.ngram_min < 2 or policy.ngram_max < policy.ngram_min:
            raise ValueError(f"{POLICY_KEY} needs 2 <= ngram_min <= ngram_max")
        if policy.min_tokens > policy.max_tokens or policy.probation_tokens > policy.max_tokens:
            raise ValueError(f"{POLICY_KEY} token bounds must not exceed max_tokens")
        if policy.backoff_max < policy.backoff_min:
            raise ValueError(f"{POLICY_KEY}.backoff_max must be at least backoff_min")
        return policy if enabled else None

    def as_dict(self):
        return asdict(self)


def block_for_ext(ext, cap):
    """``context_copy.block_for_ext``: ladder length by extension, capped."""
    index = max(0, min(int(ext), len(LADDER) - 1))
    return min(LADDER[index], max(4, int(cap)))


class NgramIndex:
    """``context_copy.NgramIndex`` (exact matcher): grams of ``ngram_min``
    prompt tokens -> the positions right after them."""

    def __init__(self, ngram_min, ngram_max, max_candidates=32):
        self.ng_min, self.ng_max, self.max_candidates = ngram_min, ngram_max, max_candidates
        self.grams = {}
        self.indexed = 0

    def sync(self, corpus):
        for end in range(max(self.indexed + 1, self.ng_min), len(corpus) + 1):
            self.grams.setdefault(tuple(corpus[end - self.ng_min:end]), []).append(end)
        self.indexed = len(corpus)

    def find(self, context, length, *, max_pos):
        """Best ``(continuation position, extension)`` or ``(None, -1)``.

        ``context(i)`` reads the current sequence (prompt + output so far,
        the anchor last) and ``length`` is its length; the longest backward
        extension wins, the most recent position breaks ties."""
        if length < self.ng_min + 1:
            return None, -1
        key = tuple(context(i) for i in range(length - self.ng_min, length))
        candidates = self.grams.get(key)
        if not candidates:
            return None, -1
        best_pos, best_ext = None, -1
        max_ext = self.ng_max - self.ng_min
        corpus = self._corpus
        end = bisect_left(candidates, min(length, max_pos))
        for pos in reversed(candidates[max(0, end - self.max_candidates):end]):
            ext = 0
            while (ext < max_ext and pos - self.ng_min - 1 - ext >= 0
                   and corpus[pos - self.ng_min - 1 - ext]
                   == context(length - self.ng_min - 1 - ext)):
                ext += 1
            if ext > best_ext:
                best_pos, best_ext = pos, ext
                if ext == max_ext:
                    break
        return best_pos, best_ext


@dataclass
class LaneCopyState:
    """Per-lane copy gate, on the lane so a round restore rewinds it."""

    ema: float = 0.5
    seen: int = 0
    suspend_until: int = 0
    backoff: int = 64
    suspensions: int = 0
    probes: int = 0
    rounds: int = 0
    proposed: int = 0
    accepted: int = 0


def _bump(stats, key, amount=1):
    stats[key] = stats.get(key, 0) + amount


class PromptCopyRuntime:
    def __init__(self, policy):
        self.policy = policy
        # uid -> (prompt length, index).  The index covers the immutable
        # prompt only, so it is never part of a round snapshot.
        self._indexes = {}

    def install_counters(self, stats):
        stats.update(dict.fromkeys(COUNTERS, 0))
        stats[DECLINE_REASONS] = {}

    def count(self, owner, reason, lanes=1):
        reasons = owner.scheduler_stats.setdefault(DECLINE_REASONS, {})
        reasons[reason] = reasons.get(reason, 0) + lanes

    def _state(self, lane):
        state = getattr(lane, "prompt_copy_state", None)
        if state is None:
            state = lane.prompt_copy_state = LaneCopyState(backoff=self.policy.backoff_min)
        return state

    def lane_receipt(self, lane):
        state = getattr(lane, "prompt_copy_state", None)
        return None if state is None else asdict(state)

    def _index(self, owner, lane):
        record = self._indexes.get(lane.uid)
        if record is None:
            live = getattr(owner, "lanes", None)
            if live is not None:
                for uid in [uid for uid in self._indexes if uid not in live]:
                    self._indexes.pop(uid, None)
            # history + [anchor] == prompt + generated tokens (prefill leaves
            # the last prompt token as the anchor with nothing generated).
            prompt_length = len(lane.history) + 1 - int(lane.generated)
            corpus = [*lane.history, int(lane.anchor)][:prompt_length]
            index = NgramIndex(self.policy.ngram_min, self.policy.ngram_max,
                               self.policy.max_candidates)
            index.sync(corpus)
            index._corpus = corpus
            record = self._indexes[lane.uid] = (prompt_length, index)
        return record

    def propose(self, owner, lane, *, cap=None):
        """This lane's copy outcome for the round, with no bookkeeping.

        Returns ``SimpleNamespace(tokens, decline, limits, probed)``:
        ``tokens`` is the copy or None.  Nothing is counted here; the runner
        calls ``book`` only for a round it serves, so a copy the round then
        discards is never counted as a probe or a limit.  ``cap`` (the
        lane-exact executor width, or None) bounds the copy length."""
        policy = self.policy
        state = self._state(lane)
        outcome = SimpleNamespace(tokens=None, decline=None, limits=[], probed=False)
        remaining = int(lane.maximum - lane.generated - 1)
        if remaining < 1:
            outcome.decline = "requested_output_budget"
            return outcome
        if lane.generated < state.suspend_until:
            outcome.decline = "acceptance_backoff"
            return outcome
        prompt_length, index = self._index(owner, lane)
        if prompt_length <= policy.ngram_min:
            outcome.decline = "no_prompt_continuation"
            return outcome
        outcome.probed = True
        history, anchor = lane.history, int(lane.anchor)
        length = len(history) + 1

        def context(i):
            return anchor if i == length - 1 else history[i]

        pos, ext = index.find(context, length, max_pos=prompt_length)
        if pos is None:
            outcome.decline = "no_prompt_match"
            return outcome
        ladder_cap = (policy.max_tokens if state.seen >= 2 and state.ema >= 0.5
                      else policy.probation_tokens)
        count = min(block_for_ext(ext, ladder_cap), ladder_cap)
        bounds = [
            ("verify_geometry", int(owner.num_draft)),
            ("requested_output_budget", remaining),
            ("end_of_prompt", prompt_length - pos),
        ]
        if cap is not None:
            bounds.append(("lane_exact_width", int(cap)))
        for reason, bound in bounds:
            if bound < count:
                count = bound
                outcome.limits.append(reason)
        if count < max(1, policy.min_tokens):
            outcome.decline = "below_min_tokens"
            return outcome
        outcome.tokens = [int(t) for t in index._corpus[pos: pos + count]]
        return outcome

    def book(self, owner, lane, outcome):
        """Count one served round's copy outcome (probe, limits, decline)."""
        if outcome.probed:
            self._state(lane).probes += 1
            _bump(owner.scheduler_stats, "external_prompt_copy_probes")
        for reason in outcome.limits:
            self.count(owner, "limit_" + reason)
        if outcome.decline is not None:
            self.count(owner, outcome.decline)

    def observe(self, owner, lane, offered, landed, *, emitted=None):
        """``PromptCopy.observe``: EMA of the landed share, suspension with
        exponential backoff (MTPLX 2.12.3 rule).  ``emitted`` is the lane's
        output length after this round (default: ``lane.generated``)."""
        policy = self.policy
        state = self._state(lane)
        state.rounds += 1
        state.proposed += offered
        state.accepted += landed
        stats = owner.scheduler_stats
        _bump(stats, "external_prompt_copy_rounds")
        _bump(stats, "external_prompt_copy_lanes")
        _bump(stats, "external_prompt_copy_proposed_tokens", offered)
        _bump(stats, "external_prompt_copy_accepted_tokens", landed)
        share = landed / offered if offered else 0.0
        state.ema = 0.7 * state.ema + 0.3 * share
        state.seen += 1
        if share >= 0.5:
            state.backoff = policy.backoff_min
        if state.seen >= 3 and state.ema < policy.ema_floor:
            state.suspend_until = (
                int(lane.generated if emitted is None else emitted) + state.backoff
            )
            state.backoff = min(state.backoff * 2, policy.backoff_max)
            state.ema, state.seen = 0.5, 0
            state.suspensions += 1
            _bump(stats, "external_prompt_copy_suspensions")


__all__ = ["LADDER", "LaneCopyState", "NgramIndex", "PromptCopyPolicy", "PromptCopyRuntime",
           "block_for_ext"]
