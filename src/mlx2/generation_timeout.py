"""Bounded HTTP engine-event waits, independent of client and body deadlines."""

import math

DEFAULT_GENERATION_EVENT_TIMEOUT_SECONDS = 1800.0
MAX_GENERATION_EVENT_TIMEOUT_SECONDS = 28800.0


def validate_generation_event_timeout(value):
    """Keep the legacy 30-minute default; explicitly permit long prefills.

    This bounds each wait for an engine event, not total generation time.
    SSE keepalive bytes do not reset the engine-event deadline.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            "generation event timeout must be a finite number within 1..28800 seconds"
        )
    if (
        not math.isfinite(value)
        or not 1 <= value <= MAX_GENERATION_EVENT_TIMEOUT_SECONDS
    ):
        raise ValueError(
            "generation event timeout must be a finite number within 1..28800 seconds"
        )
    return float(value)
