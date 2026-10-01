"""Greedy MTP verification reference using the existing adaptive depth policy.

Sequential target verification deliberately preserves ordinary token/KV math.
This mechanics runner does not claim accelerated verification or serving use.
"""

import time

import mlx.core as mx

from mlx2.runtime.adaptive_policy import CohortAdaptiveMTPDepth


class GreedyMTPReference:
    def __init__(self, model, *, policy=None, proposal_fn=None):
        if not model.config.mtp:
            raise ValueError("MTP head is absent")
        self.model = model
        self.policy = (
            policy
            if policy is not None
            else CohortAdaptiveMTPDepth(max_depth=1, adaptive_single_lane=True)
        )
        if self.policy.max_depth != 1 or not self.policy.adaptive_single_lane:
            raise ValueError("reference supports explicit single-lane depth one only")
        self.proposal_fn = proposal_fn

    def _proposal(self, first, cache):
        if self.proposal_fn is not None:
            value = self.proposal_fn(self.model, first, cache.fork())
        else:
            boundary = mx.mean(cache.boundary, axis=-2)
            hidden = self.model.mtp_head(
                boundary, self.model.embedding(mx.array([[first]]))
            )
            logits = self.model.embedding.as_linear(self.model.norm(hidden))
            value = int(mx.argmax(logits[0, -1]).item())
        if type(value) is not int or not 0 <= value < self.model.config.vocab_size:
            raise ValueError("invalid draft token")
        return value

    def generate(self, tokens, *, max_tokens=8):
        tokens = list(tokens)
        if (
            self.model.training
            or not tokens
            or any(
                type(t) is not int or not 0 <= t < self.model.config.vocab_size
                for t in tokens
            )
        ):
            raise ValueError("greedy reference requires eval and a valid prompt")
        if (
            type(max_tokens) is not int
            or max_tokens < 1
            or len(tokens) + max_tokens > self.model.config.max_context
        ):
            raise ValueError("invalid generation budget or context length")
        logits, cache = self.model.prefill(mx.array([tokens]))
        output, rounds = [], []
        while len(output) < max_tokens:
            start = time.perf_counter()
            remaining = max_tokens - len(output)
            depth = self.policy.select(admitted_cap=1 if remaining >= 2 else 0, width=1)
            first = int(mx.argmax(logits[0, -1]).item())
            proposed = self._proposal(first, cache) if depth else None
            # The only mutation is a target-verified ordinary token. A draft is
            # never placed into KV before its equality with the target is known.
            logits = self.model.decode(mx.array([[first]]), cache)
            output.append(first)
            accepted = proposed is not None and proposed == int(
                mx.argmax(logits[0, -1]).item()
            )
            committed = 1
            if accepted:
                logits = self.model.decode(mx.array([[proposed]]), cache)
                output.append(proposed)
                committed += 1
            elapsed = time.perf_counter() - start
            self.policy.observe(
                int(depth),
                int(accepted),
                width=1,
                committed=committed,
                elapsed_seconds=elapsed,
            )
            # Policy callbacks may replace weights or the memory snapshot even
            # after the final decode. Never publish verification from that owner.
            self.model._require_inference_owner(cache)
            rounds.append(
                {
                    "depth": depth,
                    "proposed": proposed,
                    "accepted": bool(accepted),
                    "committed": committed,
                    "cache_length": cache.length,
                    "elapsed_seconds": elapsed,
                }
            )
        return (
            output,
            cache,
            {
                "schema": "mlx2.hysparse2-greedy-mtp-reference.v1",
                "rounds": rounds,
                "controller_counters": dict(self.policy.counters),
                "target_verified": True,
                "verification": "sequential-ordinary-reference",
                "proposal": "injected-test"
                if self.proposal_fn is not None
                else "boundary-only-mtp",
                "mtp_history_cache": False,
                "accelerated_verification": False,
                "serving_route_qualified": False,
            },
        )
