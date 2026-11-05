"""Opt-in research receipts; provenance: request-counter-provenance-20261104T0531Z.json.

Use only with serial B1/inflight1. Direct frontend counters are independent of
the worker's periodic status snapshots. No timing or GPU-completion claim.
"""

import threading


class RequestCounters:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.pending = {}
        self.lock = threading.Lock()

    def begin(self, key):
        with self.lock:
            if key in self.pending:
                raise ValueError("duplicate request counter identity")
            overlap = bool(self.pending)
            for row in self.pending.values():
                row["overlap"] = True
            self.pending[key] = {
                "before": self.snapshot(),
                "overlap": overlap,
            }

    def finish(self, key):
        with self.lock:
            row = self.pending.pop(key)
            after = self.snapshot()
            before = row["before"]
            deltas = {}
            reset = False
            for group in set(before) | set(after):
                a, b = before.get(group, {}), after.get(group, {})
                delta = {k: b.get(k, 0) - a.get(k, 0) for k in set(a) | set(b)}
                reset |= any(v < 0 for v in delta.values())
                deltas[group] = {k: v for k, v in delta.items() if v}
            valid = not row["overlap"] and not reset
            return {
                "request_id": key,
                "valid": valid,
                "overlap": row["overlap"],
                "counter_reset": reset,
                "counts": deltas if valid else None,
                "before": before,
                "after": after,
                "scope": "Direct process frontend counters from Job construction to terminal handling; serial B1/inflight1 required. Not backend completion or performance.",
            }


def install(budget, loop_math, serving):
    """Install once in a fresh research server before any Job is constructed."""
    if getattr(serving.Job, "_request_route_counters_installed", False):
        raise RuntimeError("request route counters already installed")
    capture = RequestCounters(
        lambda: {
            "normal": dict(budget.COUNTS),
            "extra": dict(loop_math.LOOP_COUNTS),
        }
    )
    original_init = serving.Job.__init__
    original_finish = serving.ServingEngine._finish

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        capture.begin(self.id)
        self._request_route_counter_pending = True

    def finish(self, job, event):
        if getattr(job, "_request_route_counter_pending", False):
            receipt = capture.finish(job.id)
            job._request_route_counter_pending = False
            if event.get("receipt") is not None:
                event = {
                    **event,
                    "receipt": {
                        **event["receipt"],
                        "probe_request_route_counters": receipt,
                    },
                }
        return original_finish(self, job, event)

    serving.Job.__init__ = init
    serving.Job._request_route_counters_installed = True
    serving.ServingEngine._finish = finish
    return capture
