# SPDX-License-Identifier: MIT
"""Measured, best-effort admission for bounded demand-streamed routes.

The route owns reclamation and its working-set strategy. Admission retains
truthful estimates, but an advisory shortfall does not prohibit attempting a
single bounded request. Actual allocator failures remain errors.
"""
import math

from .elastic_memory import ElasticHeadroom


def demand_stream_enabled(stream):
    return bool(getattr(stream, "adaptive_residency_enabled", False)
                and getattr(stream, "memory_admission_policy", None)
                == "attempt_after_reclaim")


def recommended_prefill_rows(owner, maximum, *, context_tokens=0):
    """Resolve a route's live chunk cap without widening the caller's cap."""
    if type(maximum) is not int or maximum < 1:
        raise ValueError("prefill maximum must be a positive integer")
    recommend = getattr(owner, "recommended_prefill_rows", None)
    if not callable(recommend):
        return maximum
    rows = recommend(maximum, context_tokens=int(context_tokens))
    if type(rows) is not int or not 1 <= rows <= maximum:
        raise ValueError("route prefill recommendation must be within its maximum")
    return rows


class DemandStreamCacheBudget:
    """Retain the adapter's cache math and expose its live chunk recommendation."""
    def __init__(self, budget, recommend):
        self.budget, self.recommended_prefill_rows = budget, recommend

    def __getattr__(self, name):
        return getattr(self.budget, name)


class DemandStreamHeadroom(ElasticHeadroom):
    def __init__(self, measure, stream):
        if not demand_stream_enabled(stream):
            raise ValueError("route did not declare demand-stream admission")
        self.availability_known = False
        def observe():
            try:
                available = measure()
            except Exception:
                available = None
            self.availability_known = type(available) is int and available >= 0
            return available if self.availability_known else 0
        super().__init__(observe, stream.unfilled_reserve_bytes,
                         stream.rebalance_headroom)
        self.stream = stream

    def allow_pressure_attempt(self, required_bytes):
        if (isinstance(required_bytes, bool)
                or not isinstance(required_bytes, (int, float))
                or not math.isfinite(required_bytes) or required_bytes < 0):
            raise ValueError("pressure requirement must be finite and nonnegative")
        available = self.measure()
        if type(available) is not int or available < 0:
            raise ValueError("measured headroom must be nonnegative bytes")
        receipt = {
            "schema": "mlx2.demand-stream-admission.v1",
            "policy": "attempt_after_reclaim",
            "required_bytes": math.ceil(required_bytes),
            "available_bytes": available,
            "availability_known": self.availability_known,
            "advisory_shortfall_bytes": max(0, math.ceil(required_bytes)-available),
            "allocator_result": "pending",
            "qualification": "unqualified",
        }
        self.stream.last_pressure_admission = receipt
        return True


def allow_pressure_attempt(headroom, required_bytes):
    attempt = getattr(headroom, "allow_pressure_attempt", None)
    return bool(attempt(required_bytes)) if callable(attempt) else False
