# SPDX-License-Identifier: LicenseRef-Pierre-Lamy-Calculator-NC-ND-1.0
# Copyright (c) 2026 Pierre Lamy
# Scope and earlier grants: provenance/LICENSE-calculator-scope.json
# Terms: provenance/LICENSE-calculator.txt
"""Research-only completion guards; reuse mlx2 alarms without a thinking-channel close.

The evaluator disables thinking, so forcing </think> would be the wrong actuator.
A detected loop/spiral instead requests one bounded, calculator-enabled final-answer
continuation. No candidate answer or ground truth is supplied to that continuation.
"""
from mlx2.spiral_conclude import SpiralDetector
from mlx2.thinking_guard import ThinkingGuard

NUDGE = '\n\nI will stop rechecking and give the final numeric answer now.\n#### '
CONCLUDE_TOKENS = 512
RUN_ON_MIN_TOKENS = 1024


class CompletionDefenses:
    def __init__(self, *, min_tokens=RUN_ON_MIN_TOKENS, tau=2.5, ngram=6, spiral=None):
        # Only the existing pure-CPU alarm is used; EOS/</think> logits are not biased.
        self.alarm = ThinkingGuard(0, (0,), budget=None, tau=tau, ngram=ngram)
        self.spiral = spiral or SpiralDetector()
        self.min_tokens = min_tokens
        self.tokens = 0
        self.reason = None
        self._text = ''

    def feed(self, token, text=None):
        self.tokens += 1
        self.alarm._advance(int(token))
        if text is not None:
            # Decode complete token prefixes, never isolated byte-fallback pieces.
            stable = text.rstrip('\ufffd')
            if not stable.startswith(self._text):
                raise ValueError('defense tokenizer prefix changed')
            delta = stable[len(self._text):]
            self._text = stable
            if self.spiral.feed(delta, self.tokens):
                self.reason = 'spiral'
        if (self.reason is None and self.tokens >= self.min_tokens
                and self.alarm._cusum >= self.alarm.tau * self.alarm.ngram):
            self.reason = 'run_on'
        return self.reason

    def receipt(self):
        return dict(enabled=True,scope='completion_text',reason=self.reason,tokens_observed=self.tokens,
                    run_on=dict(min_tokens=self.min_tokens,tau=self.alarm.tau,ngram=self.alarm.ngram,
                                current_cusum=self.alarm._cusum,first_alarm_token=self.alarm._tripped_at),
                    spiral=self.spiral.receipt())


def phase_budgets(cap, nudge_tokens):
    """Count injected nudge tokens inside the caller's total output-token bound."""
    reserve = min(CONCLUDE_TOKENS, cap // 4)
    initial = cap - nudge_tokens - reserve
    if initial <= 0:
        raise ValueError('defended token budget cannot hold nudge and continuation')
    return initial, reserve
