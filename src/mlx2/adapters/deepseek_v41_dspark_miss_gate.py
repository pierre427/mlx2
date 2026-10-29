# SPDX-License-Identifier: MIT
"""Opt-in DSpark speculation admission priced by measured expert-miss cost.

``deepseek_dspark_admission: expert_miss_cost_v1`` (embedded DSpark only).
On the Source checkpoint a verify of k+1 rows routes to more experts than one
ordinary token, and each missing expert is an SSD read: DSpark at 95-98%
acceptance still lost to ordinary decode in every 2026-10-28 arm. The gate
decides per round, from this request's own measurements, whether a
speculative round is expected to beat plain decode:

    draft_s + verify_nonread_s + predicted_miss_bytes / read_bandwidth
        <= useful_tokens / plain_tokens_per_second

* ``plain_tokens_per_second``: EWMA of this request's plain rounds (a round
  with zero proposals: the draft plane only appends context and the target
  commits one token through the same verify owner).
* ``draft_s``, ``verify_nonread_s``, ``useful_tokens``: EWMAs of this
  request's speculative rounds; verify time excludes the expert-read wait.
* ``predicted_miss_bytes``: EWMA of the expert bytes a speculative round read
  (no route predictor; the previous rounds' measured misses).
* ``read_bandwidth``: this request's measured expert bytes over read wait,
  falling back to the provider's lifetime measurement.

Like the ds4 policy it is adapted from, the first rounds measure plain decode,
then one speculative round prices speculation. A rejected round runs plain;
``HANDOFF_REJECTIONS`` consecutive priced rejections hand the request to the
ordinary target route (one-way) through the scheduler's recoverable
``DraftUnavailable`` path (counted in ``draft_fallbacks``). The exception
carries ``ordinary_reason = REASON``; the draft receipt's
``admission_observations`` records the handoff and the full decision terms.
(Copying that reason into the response's ``ordinary_fallback_reason`` needs a
one-line change in the generic scheduler, whose source hash binds other
families' AST identities; deliberately not made here.) Every committed token
stays the target's greedy choice, so output is exact; only the route changes.
Unqualified, default off.
Provenance: provenance/deepseek-v41-dspark-miss-cost-gate.json.
"""
import math
import time

KEY = 'deepseek_dspark_admission'
OFF = 'off'
POLICY = 'expert_miss_cost_v1'
POLICIES = (OFF, POLICY)
REASON = 'dspark_expert_miss_cost_gate'
PLAIN_PROBE_ROUNDS = 2
HANDOFF_REJECTIONS = 2
EWMA = 0.5
MIN_BANDWIDTH_EVIDENCE_BYTES = 64 << 20


def validate(value):
    if value not in POLICIES:
        raise ValueError('deepseek_dspark_admission must be off or expert_miss_cost_v1')
    return value


def _ewma(previous, value):
    return value if previous is None else EWMA * value + (1 - EWMA) * previous


def decide(*, plain_tokens_per_second, propose_seconds, verify_nonread_seconds, predicted_miss_bytes,
           useful_tokens, read_bytes, read_seconds, fallback_read_bytes=0, fallback_read_seconds=0.0):
    """Pure admission law. Returns a decision dict; raises on invalid evidence."""
    values = (plain_tokens_per_second, propose_seconds, verify_nonread_seconds, predicted_miss_bytes,
              useful_tokens, read_bytes, read_seconds, fallback_read_bytes, fallback_read_seconds)
    if any(v is not None and (type(v) not in (int, float) or not math.isfinite(v) or v < 0) for v in values):
        raise ValueError('DSpark miss-cost evidence must be finite and nonnegative')
    if not plain_tokens_per_second:
        return dict(action='plain', admit=False, priced=False, reason='observe_plain_rate')
    if propose_seconds is None or not useful_tokens:
        return dict(action='speculate', admit=True, priced=False, reason='price_speculation')
    bandwidth, source = None, None
    if read_bytes >= MIN_BANDWIDTH_EVIDENCE_BYTES and read_seconds > 0:
        bandwidth, source = read_bytes / read_seconds, 'request'
    elif fallback_read_bytes >= MIN_BANDWIDTH_EVIDENCE_BYTES and fallback_read_seconds > 0:
        bandwidth, source = fallback_read_bytes / fallback_read_seconds, 'provider_lifetime'
    if predicted_miss_bytes > 0 and bandwidth is None:
        return dict(action='speculate', admit=True, priced=False, reason='read_bandwidth_unknown')
    miss_seconds = predicted_miss_bytes / bandwidth if predicted_miss_bytes > 0 else 0.0
    expected = propose_seconds + verify_nonread_seconds + miss_seconds
    plain = useful_tokens / plain_tokens_per_second
    admit = expected <= plain
    return dict(action='speculate' if admit else 'plain', admit=admit, priced=True,
                reason='expected_round_within_plain_time' if admit else 'expected_round_exceeds_plain_time',
                expected_seconds=expected, plain_seconds=plain, miss_seconds=miss_seconds,
                bytes_per_second=bandwidth, bandwidth_source=source,
                predicted_miss_bytes=predicted_miss_bytes, useful_tokens=useful_tokens,
                plain_tokens_per_second=plain_tokens_per_second)


class MissCostGate:
    """Request-local evidence; ``stats`` is a callable returning provider counters."""

    def __init__(self, stats):
        self._stats = stats
        self.plain_rate = None
        self.propose_s = self.verify_nonread_s = self.miss_bytes = self.useful = None
        self.plain_rounds = self.spec_rounds = self.rejections = 0
        self.read_bytes = 0
        self.read_ns = 0
        self.previous = None       # (kind, draft_seconds, end_ns, counters)
        self.last = None
        self.handoff = False

    def _counters(self):
        values = self._stats() or {}
        return int(values.get('bytes_read', 0)), int(values.get('demand_wait_ns', 0))

    def observe(self, committed_tokens):
        """Close the previous round at the start of the next draft call."""
        if self.previous is None:
            return
        kind, draft_s, end_ns, (bytes_before, wait_before) = self.previous
        self.previous = None
        now = time.perf_counter_ns()
        read, wait = self._counters()
        delta_bytes, delta_wait = max(0, read - bytes_before), max(0, wait - wait_before)
        wall = max(0, now - end_ns) / 1e9
        self.read_bytes += delta_bytes
        self.read_ns += delta_wait
        if committed_tokens <= 0:
            return
        if kind == 'plain':
            self.plain_rounds += 1
            seconds = draft_s + wall
            if seconds > 0:
                self.plain_rate = _ewma(self.plain_rate, committed_tokens / seconds)
        else:
            self.spec_rounds += 1
            self.propose_s = _ewma(self.propose_s, draft_s)
            self.verify_nonread_s = _ewma(self.verify_nonread_s, max(0.0, wall - delta_wait / 1e9))
            self.miss_bytes = _ewma(self.miss_bytes, float(delta_bytes))
            self.useful = _ewma(self.useful, float(committed_tokens))

    def decision(self):
        if self.plain_rounds < PLAIN_PROBE_ROUNDS:
            result = dict(action='plain', admit=False, priced=False, reason='observe_plain_rate')
        else:
            lifetime_bytes, lifetime_wait = self._counters()
            result = decide(plain_tokens_per_second=self.plain_rate or 0.0, propose_seconds=self.propose_s,
                            verify_nonread_seconds=self.verify_nonread_s or 0.0,
                            predicted_miss_bytes=self.miss_bytes or 0.0, useful_tokens=self.useful or 0.0,
                            read_bytes=self.read_bytes, read_seconds=self.read_ns / 1e9,
                            fallback_read_bytes=lifetime_bytes, fallback_read_seconds=lifetime_wait / 1e9)
        if result['priced'] and not result['admit']:
            self.rejections += 1
        elif result['priced']:
            self.rejections = 0
        result['consecutive_rejections'] = self.rejections
        result['handoff'] = self.rejections >= HANDOFF_REJECTIONS
        self.handoff = result['handoff']
        self.last = result
        return result

    def begin(self, kind, draft_seconds):
        self.previous = (kind, draft_seconds, time.perf_counter_ns(), self._counters())

    def receipt(self):
        return dict(policy=POLICY, plain_rounds=self.plain_rounds, speculative_rounds=self.spec_rounds,
                    plain_tokens_per_second=self.plain_rate, propose_seconds=self.propose_s,
                    verify_nonread_seconds=self.verify_nonread_s, predicted_miss_bytes=self.miss_bytes,
                    useful_tokens=self.useful, request_read_bytes=self.read_bytes,
                    request_read_seconds=self.read_ns / 1e9, consecutive_rejections=self.rejections,
                    handoff=self.handoff, last_decision=self.last, qualification='unqualified')
