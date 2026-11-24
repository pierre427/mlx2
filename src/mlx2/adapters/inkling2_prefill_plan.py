"""``inkling2_prefill_wide_chunk``: an opt-in 2,048-row Inkling 2 prefill chunk.

Original mlx2 code (2026-11-23). The idea (a wider chunk gated on GPU headroom
and capped at a prompt depth while the attention builds rows x keys mask
planes) comes from the MTPLX sweep (docs/research/2026-11-23-mtplx-inkling2-
sweep); no MTPLX code was copied. See provenance/inkling2-mtplx-prefill.json.

The width schedule is a deterministic function of the absolute depth a chunk
starts at, fixed when the process loads; it never reads load or memory at run
time and keeps no memo:

* a chunk starting below ``CROSSOVER`` (18,432 keys) runs up to the step
  (2,048) and never past the crossover: ``min(step, CROSSOVER - depth)``;
* a chunk starting at or past it runs ``NARROW_ROWS`` (512);

so every chunk of the composed attention stays within the keys MTPLX found
the 2,048-row mask planes affordable at. A cold read and a read resumed from
a restored prefix take the same chunk at the same depth, and a follow-up turn
recomputes the earlier turn's prefix on exactly the chunks that turn used.

Hooks (generic, no model-name branch in the scheduler):

* ``prefill_step_default()`` = 2,048 when the route engages (512 otherwise);
* the model declares ``_mlx2_prefill_rows_at(step, depth)``
  (``generate._depth_bounded_step`` / ``_planned_prefill_rows``), so the idle
  plan, execution and the APCv2 checkpoint grid all see the same widths;
* the adapter declares ``prefill_schedule_identity(step)``
  (``serving.prefill_step_namespace_term``): the schedule (widths, crossover)
  is part of the APCv2 namespace, so the wide route never shares a namespace
  with the 512 route or with a plain ``--prefill-step 2048``.

Admission charges the largest chunk the schedule can run for the request
(``Inkling2CacheBudget.prefill_transient_bytes``: the wide chunk ending at
most at the crossover, or a narrow chunk at the full depth); a request that
does not fit waits or is refused by ordinary admission and is never narrowed.
The executed-chunk counter is bound to the resolved step
(``Inkling2Adapter.bind_prefill_step``) and counts lanes per forward.

Load-time decline: free GPU memory (recommended working set minus MLX active
memory) below ``WIDE_HEADROOM_BYTES`` (12 GiB), or no GPU:
the route does not engage, the step stays 512 and the namespace is the 512
route's (``receipt()["decline"]``).
"""

from __future__ import annotations

WIDE_ROWS = 2048
NARROW_ROWS = 512
CROSSOVER = 18_432
WIDE_HEADROOM_BYTES = 12 << 30
SCHEMA = "mlx2.inkling2-prefill-schedule.v1"


def gpu_headroom_bytes():
    """Bytes the GPU may still allocate before the recommended limit, or None."""
    try:
        import mlx.core as mx
    except ImportError:
        return None
    if not mx.metal.is_available():
        return None
    try:
        working_set = mx.device_info().get("max_recommended_working_set_size")
        active = mx.get_active_memory()
    except (AttributeError, RuntimeError):
        return None
    if not working_set:
        return None
    return int(working_set) - int(active)


def rows_at(step, depth, *, crossover=CROSSOVER, narrow=NARROW_ROWS):
    """The chunk width for a chunk starting at absolute ``depth``."""
    step, depth = int(step), int(depth)
    if depth < crossover:
        return max(1, min(step, crossover - depth))
    return max(1, min(step, narrow))


class WidePrefillSchedule:
    """The load-time route: engaged or declined (with the reason)."""

    def __init__(self, *, headroom_bytes, crossover=CROSSOVER, narrow=NARROW_ROWS,
                 wide=WIDE_ROWS):
        self.crossover, self.narrow, self.wide = int(crossover), int(narrow), int(wide)
        self.load_headroom = headroom_bytes
        if headroom_bytes is None:
            self.decline = "no_gpu"
        elif headroom_bytes < WIDE_HEADROOM_BYTES:
            self.decline = "headroom"
        else:
            self.decline = None
        self.counts = {"rows_at": 0}

    @property
    def engaged(self):
        return self.decline is None

    def rows_at(self, *, step, depth):
        self.counts["rows_at"] += 1
        return rows_at(step, depth, crossover=self.crossover, narrow=self.narrow)

    def identity(self, step):
        return {"schema": SCHEMA, "max_rows": int(step), "narrow_rows": self.narrow,
                "crossover": self.crossover}

    def receipt(self):
        return {"selected": True, "engaged": self.engaged, "decline": self.decline,
                "wide_rows": self.wide, "narrow_rows": self.narrow,
                "crossover": self.crossover, "load_headroom_bytes": self.load_headroom,
                "headroom_floor_bytes": WIDE_HEADROOM_BYTES, "lookups": dict(self.counts)}
