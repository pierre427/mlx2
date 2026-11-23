"""CPU checks for the ``MLX2_KV_PLANNED_GROWTH`` kill switch.

``1`` (default) sizes prompt K/V growth for the lane's planned span; ``0``
restores the stepped policy (one 256-row-quantum reallocation per prompt
chunk wider than a step, no scheduler plan, no planned-capacity trims).  The
two arms must give bitwise-identical outputs, and the arm must be visible in
the scheduler receipt.
"""

import os
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import pytest

from mlx2.runtime import prefill_lookahead as PL
from mlx2.runtime.generate import BatchGenerator, PromptProcessingBatch
from mlx2.runtime.models import hy_v4_flash as H
from mlx2.runtime.models.cache import BatchKVCache, KVCache

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 512  # wider than the 256-row step: stepped growth reallocates per chunk
PROMPT = [(7 * i + 3) % 120 + 2 for i in range(6 * CHUNK + 37)]


@pytest.mark.parametrize("raw,expected", [
    (None, True), ("1", True), ("0", False), ("off", False), ("false", False), (" 1 ", True),
])
def test_switch_is_read_once_at_import(raw, expected):
    env = {k: v for k, v in os.environ.items() if k != "MLX2_KV_PLANNED_GROWTH"}
    if raw is not None:
        env["MLX2_KV_PLANNED_GROWTH"] = raw
    env["PYTHONPATH"] = str(ROOT / "src")
    code = ("import os\nfrom mlx2.runtime import prefill_lookahead as P\n"
            "os.environ['MLX2_KV_PLANNED_GROWTH'] = '0' if P.KV_PLANNED_GROWTH else '1'\n"
            "print(P.planned_growth_enabled())")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == str(expected)  # a later environment change does not flip it


def test_off_plans_nothing():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(PL, "KV_PLANNED_GROWTH", False)
        with PL.prefill_lookahead([100], step=512, width=512, planned_after=[1000]):
            assert PL.allocation_rows_after(512, 1) == 0
        gen = object.__new__(BatchGenerator)
        gen._currently_processing = [[[list(range(5001))], 2048, 7049, None, 0]]
        gen._prompt_batch = type("PB", (), {"uids": [9]})()
        gen._interior_checkpoint_positions = {9: []}
        gen._inpass_checkpoint_enabled = lambda: False
        assert gen._planned_prompt_rows_after() is None
    with PL.prefill_lookahead([100], step=512, width=512, planned_after=[1000]):
        assert PL.allocation_rows_after(512, 1) == 1100


# -- the prompt loop: allocation counts and bitwise outputs ------------------------


class _AppendModel:
    def __call__(self, tokens, *, cache):
        b, n = tokens.shape
        base = tokens.astype(mx.float32)[:, None, :, None]
        keys = mx.broadcast_to(base, (b, 2, n, 8)) * 0.5
        for c in cache:
            c.update_and_fetch(keys, keys + 1)
        return mx.zeros((b, n, 4), mx.float32)


def _prompt_batch(cache):
    batch = object.__new__(PromptProcessingBatch)
    batch.uids = [5]
    batch.tokens = [[]]
    batch.prompt_cache = [cache]
    batch.prefill_step_size = CHUNK
    batch.model = _AppendModel()
    batch.decode_active = False
    batch.prompt_trim_rollback_tokens = 0
    return batch


def _sliced_prompt(enabled):
    cache, allocations, last = BatchKVCache([0]), 0, None
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(PL, "KV_PLANNED_GROWTH", enabled)
        for start in range(0, len(PROMPT), CHUNK):  # one scheduler slice per call
            piece = PROMPT[start:start + CHUNK]
            with PL.planned_prompt_rows([len(PROMPT) - start - len(piece)]):
                _prompt_batch(cache).prompt([piece])
            if cache.keys is not last:
                allocations, last = allocations + 1, cache.keys
    return cache, allocations


def test_off_reproduces_one_reallocation_per_chunk_and_is_bitwise():
    on, on_count = _sliced_prompt(True)
    off, off_count = _sliced_prompt(False)
    assert on_count == 1
    assert off_count == -(-len(PROMPT) // CHUNK)
    assert on._idx == off._idx == len(PROMPT)
    for a, b in zip(on.keys_and_values(), off.keys_and_values()):
        assert a.shape == b.shape and bool(mx.array_equal(a, b).item())


def test_off_keeps_trims_off():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(PL, "KV_PLANNED_GROWTH", False)
        row = H.HY4FlashLayerCache(H.INDEXED, ratio=4)
        row.kv_data, row.kv_head, row.kv_end = mx.zeros((1, 4096, 4)), 0, 10
        assert H._without_planned_capacity(row).kv_data.shape[1] == 4096
        c = BatchKVCache([0])
        c.update_and_fetch(mx.zeros((1, 2, 10, 8)), mx.zeros((1, 2, 10, 8)))
        c.keys = mx.concatenate([c.keys, mx.zeros((1, 2, 4096, 8))], axis=2)
        c.values = mx.concatenate([c.values, mx.zeros((1, 2, 4096, 8))], axis=2)
        width = c.keys.shape[2]
        c.filter([0])
        assert c.keys.shape[2] == width


# -- the engine: receipt, allocation counts and greedy tokens ----------------------


def _tiny_model():
    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_27b import TextModel

    args = TextModelArgs(
        model_type="qwen3_5", hidden_size=64, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, vocab_size=128, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=3, full_attention_interval=4,
        mtp_num_hidden_layers=0, partial_rotary_factor=0.5,
        rope_parameters=None, max_position_embeddings=1 << 14,
    )
    mx.random.seed(7)
    model = TextModel(args)
    model.eval()
    mx.eval(model.parameters())
    return model


def _engine_run(enabled, monkeypatch):
    monkeypatch.setattr(PL, "KV_PLANNED_GROWTH", enabled)
    growth = {"count": 0}
    originals = {cls: cls.update_and_fetch for cls in (KVCache, BatchKVCache)}

    def counting(cls):
        def update_and_fetch(self, keys, values):
            before = self.keys
            out = originals[cls](self, keys, values)
            if self.keys is not before:
                growth["count"] += 1
            return out
        return update_and_fetch

    for cls in originals:
        monkeypatch.setattr(cls, "update_and_fetch", counting(cls))
    gen = BatchGenerator(_tiny_model(), completion_batch_size=1, prefill_batch_size=1,
                         prefill_step_size=CHUNK, prefill_batch_window=1,
                         adaptive_prefill=False)
    assert gen.scheduler_stats["kv_planned_growth"] is enabled
    uid = gen.insert([PROMPT], max_tokens=[6])[0]
    tokens, prefill_growth, prompt_done = [], None, False
    for _ in range(200):
        prompts, responses = gen.next()
        if not prompt_done and any(r.uid == uid and r.end_of_prompt for r in prompts):
            prompt_done, prefill_growth = True, growth["count"]
        tokens += [r.token for r in responses if r.uid == uid]
        if any(r.uid == uid and r.finish_reason for r in responses):
            break
    for cls, fn in originals.items():
        monkeypatch.setattr(cls, "update_and_fetch", fn)
    return tokens, prefill_growth, gen


def test_engine_arms_are_token_identical_and_report_the_arm(monkeypatch):
    on_tokens, on_growth, on_gen = _engine_run(True, monkeypatch)
    off_tokens, off_growth, off_gen = _engine_run(False, monkeypatch)
    assert len(on_tokens) == 6 and on_tokens == off_tokens
    assert on_gen.scheduler_stats["kv_planned_growth"] is True
    assert off_gen.scheduler_stats["kv_planned_growth"] is False
    # The tiny model has one attention layer.  Off: one growth per prompt
    # chunk (the final prompt token is decode's); on: one for the whole span.
    assert off_growth == -(-(len(PROMPT) - 1) // CHUNK) == 7
    assert on_growth == 1
