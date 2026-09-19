"""CPU regressions distilled from the 2026-09-17 peer-PR sweep."""

import mlx.core as mx
import pytest

from mlx2.adapters.qwen38_memory import Qwen38CacheBudget
from mlx2.runtime.apc_v2 import APCKey, APCv2, MTPAPCSidecar
from mlx2.runtime.cache_planes import PLEResidencyHints
from mlx2.runtime.generate import PromptProcessingBatch, _crossed_counter_interval
from mlx2.runtime.models.cache import ArraysCache, KVCache


def _qwen38_config():
    return {
        "num_hidden_layers": 64,
        "full_attention_interval": 4,
        "mtp_num_hidden_layers": 1,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "linear_num_value_heads": 48,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    }


@pytest.mark.parametrize(
    "previous,current,expected",
    [
        (0, 511, False),
        (511, 512, True),
        (510, 514, True),
        (512, 1023, False),
        (513, 1024, True),
    ],
)
def test_allocator_reclaim_detects_crossed_counter_boundaries(
    previous, current, expected
):
    assert _crossed_counter_interval(previous, current, 512) is expected


def test_allocator_reclaim_rejects_non_monotone_counters():
    with pytest.raises(ValueError, match="monotone"):
        _crossed_counter_interval(10, 9, 512)
    with pytest.raises(ValueError, match="positive"):
        _crossed_counter_interval(0, 1, 0)


def test_qwen38_speculative_scratch_is_charged_only_for_running_lanes():
    budget = Qwen38CacheBudget.from_config(_qwen38_config(), mtp=True)
    context = 8192
    resident = budget.project_resident(context)
    running = budget.project(context)
    assert running - resident == budget.speculative_scratch_bytes
    assert budget.project_pool(context, resident_lanes=12, running_lanes=3) == (
        12 * resident + 3 * budget.speculative_scratch_bytes
    )
    assert (
        budget.project_pool(context, resident_lanes=12, running_lanes=0)
        == 12 * resident
    )
    with pytest.raises(ValueError, match="cannot exceed"):
        budget.project_pool(context, resident_lanes=2, running_lanes=3)


class _CheckpointRecorder:
    def __init__(self):
        self.calls = []
        self.state = []

    def state_checkpoint(self, positions, force=False):
        self.calls.append((tuple(positions), bool(force)))


class _NoopModel:
    def __call__(self, _tokens, *, cache):
        return None


def test_short_final_prefill_chunk_forces_exact_logical_checkpoint(monkeypatch):
    recorder = _CheckpointRecorder()
    batch = object.__new__(PromptProcessingBatch)
    batch.uids = [7]
    batch.tokens = [[]]
    batch.prompt_cache = [recorder]
    batch.prefill_step_size = 4
    batch.model = _NoopModel()
    monkeypatch.setattr(mx, "eval", lambda *args: None)
    monkeypatch.setattr(mx, "clear_cache", lambda: None)

    batch.prompt([[1, 2, 3, 4, 5]])

    assert recorder.calls[-1] == ((5,), True)
    assert ((4,), False) in recorder.calls
    assert all(position[0] <= 5 for position, _force in recorder.calls)


def test_sparse_global_layer_ids_use_compact_explicit_slots():
    hints = PLEResidencyHints(
        policy_version="v1",
        resident_layer_ids=(2, 17, 63),
        backing_identity="weights-v1",
    )
    assert hints.compact_slot_by_layer_id == {2: 0, 17: 1, 63: 2}
    assert len(hints.compact_slot_by_layer_id) == len(hints.resident_layer_ids)
    with pytest.raises(ValueError, match="unique"):
        PLEResidencyHints("v1", (2, 2), "weights-v1")


def _hybrid_cache(length, seed):
    recurrent = ArraysCache(1)
    recurrent[0] = mx.full((1, 4), seed, dtype=mx.float32)
    recurrent.lengths = mx.array([length], dtype=mx.int32)
    recurrent._host_lengths = (recurrent.lengths, [length])
    recurrent.state_checkpoint([length], force=True)
    attention = KVCache()
    values = mx.arange(seed, seed + length, dtype=mx.float32).reshape(1, 1, length, 1)
    attention.update_and_fetch(values, values)
    mx.eval(recurrent.state, attention.state)
    return [recurrent, attention]


def test_apcv2_retires_target_recurrence_attention_and_draft_as_one_entry():
    apc = APCv2(max_size=2, layout_name="peer-pr-hybrid-v1")
    key = APCKey("hybrid", revision="v1")
    tokens = [1, 2, 3, 4]
    target = _hybrid_cache(len(tokens), 10)
    draft = [KVCache()]
    values = mx.ones((1, 1, len(tokens) - 1, 1), dtype=mx.float32)
    draft[0].update_and_fetch(values, values)
    apc.store(
        key,
        tokens,
        target,
        sidecar=MTPAPCSidecar(
            (draft, mx.ones((1, 1, 4), dtype=mx.float32)),
            covered_tokens=len(tokens),
        ),
    )

    hit = apc.lookup(key, tokens + [5])
    assert hit.hit_kind == "mtp_sidecar"
    assert hit.sidecar is not None
    hit.cache.close()
    assert apc.evict_oldest_unleased()
    miss = apc.lookup(key, tokens + [5])
    assert not miss.hit
    assert miss.sidecar is None
    apc.clear(release_memory=False)
