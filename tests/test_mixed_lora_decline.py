"""A declined mixed round prefills the prompt slice with its LoRA rows bound
(sweep 2026-10-02 V2).  With concurrent multi-LoRA every mixed round is
declined (decode lanes bind LoRA rows); the fallback then ran the slice's
own forward without ``bind_lora_rows``, so a LoRA request prefilled as base.
"""

import mlx.core as mx

from mlx2.runtime import generate as G
from mlx2.runtime.multi_lora import bind_lora_rows, clear_lora_rows


class _Manager:
    def __init__(self):
        self.bound = None

    def bind_rows(self, uids):
        self.bound = list(uids)

    def clear_rows(self):
        self.bound = None


class _Model:
    def __init__(self):
        self._mlx2_multi_lora = _Manager()
        self.prefill_rows_seen = []

    def __call__(self, tokens, cache=None, **kw):
        self.prefill_rows_seen.append(self._mlx2_multi_lora.bound)

    def mixed_forward(self, segments):
        return (None, None)

    def logits(self, hidden):
        return hidden


class _Gen:
    """Mirrors GenerationBatch._step's gate: with LoRA rows bound the step
    leaves ``_mixed_segment`` set (declined); without, it consumes it."""

    def __init__(self, model):
        self.model, self._mixed_segment, self.uids = model, None, [7]

    def __len__(self):
        return 1

    def next(self):
        rows = bind_lora_rows(self.model, self.uids)
        try:
            if self._mixed_segment is not None and rows is None:
                self._mixed_segment = None
        finally:
            clear_lora_rows(rows)
        return []


class _Prompt:
    uids = [42]

    def prompt(self, chunks, *, forward_fn=None):
        forward_fn(mx.array(chunks), None)


class _Fair:
    def stall_bound(self, n):
        return n

    def floor_slice(self, chunk, limit, *, contended):
        return chunk

    def observe_prefill(self, *a, **k):
        pass


def test_declined_mixed_round_binds_the_prompt_lora_rows():
    model = _Model()
    bg = object.__new__(G.BatchGenerator)
    bg.model = model
    bg._generation_batch = _Gen(model)
    bg._prompt_batch = _Prompt()
    bg._currently_processing = [[[list(range(256))], 0, 257, None, 0]]
    bg.prefill_step_size = 256
    bg.prefill_step_autoscale = False
    bg.prefill_depth_budget = None
    bg.scheduler_stats = {"prefill_rounds": 0}
    bg._prompt_tokens_counter = bg._gen_tokens_counter = bg._steps_counter = 0
    bg._prompt_time_counter = 0.0
    bg._last_decode_completed_s = None
    bg.adaptive_prefill = False
    bg._fairness = lambda: _Fair()
    bg._next_interior_checkpoint = lambda uid, covered: None
    bg._record_prefill_chunk = lambda uid, n: None
    bg._capture_plain_interior_checkpoints = lambda: None
    bg._sync_decode_fairness_stats = lambda: None
    bg._next_mixed()
    assert bg.scheduler_stats.get("mixed_declined_rounds") == 1
    assert model.prefill_rows_seen == [[42]]
    assert model._mlx2_multi_lora.bound is None  # cleared after the forward
