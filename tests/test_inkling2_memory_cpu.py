"""Inkling 2 cache budget: an upper bound on the real lane cache, CPU only."""

import dataclasses
import json
from pathlib import Path

import mlx.core as mx
import pytest

from mlx2.adapters.inkling2_memory import Inkling2CacheBudget
from mlx2.runtime.models.cache import KVCache, RotatingKVCache
from mlx2.runtime.models.inkling2 import Model, TextConfig

from test_inkling2_cpu import TINY

REAL = Path.home() / "mlx-models/Inkling-2-Mini/config.json"


@pytest.fixture(autouse=True)
def _cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def _allocated(cache):
    total = 0
    for c in cache:
        if isinstance(c, (KVCache, RotatingKVCache)):
            total += sum(a.nbytes for a in (c.keys, c.values) if a is not None)
        else:
            total += sum(a.nbytes for a in c.cache if a is not None)
    return total


@pytest.mark.parametrize("tokens", [5, 40, 120, 300])
def test_projection_bounds_the_allocated_lane_cache(tokens):
    config = TextConfig.from_dict(TINY)
    model = Model(config)
    mx.eval(model.parameters())
    step = 16
    budget = dataclasses.replace(Inkling2CacheBudget.from_config(config), prefill_rows=step,
                                 item_bytes=4)  # the tiny model computes in FP32
    cache = model.make_cache()
    ids = mx.random.randint(1, 400, (1, tokens), key=mx.random.key(tokens))
    for a in range(0, tokens, step):
        mx.eval(model(ids[:, a:a + step], cache=cache))
    actual = _allocated(cache)
    projected = budget.project(tokens)
    assert actual <= projected
    # Tight enough to be useful once the 256-token allocation step stops
    # dominating (a 5-token prompt is charged one whole step per visit).
    if tokens >= budget.allocation_step:
        assert projected <= 2 * actual


def test_adapter_budgets_executed_loop_visits_and_prefill_step():
    from mlx2.adapters.inkling2 import Inkling2Adapter
    from mlx2.serving import bind_adapter_prefill_step

    adapter = Inkling2Adapter.__new__(Inkling2Adapter)
    adapter.model = Model(TextConfig.from_dict(TINY))
    adapter.model.model.set_num_passes(8)
    bind_adapter_prefill_step(adapter, 4096)
    budget = adapter.cache_budget(mtp=False)
    cache = adapter.model.make_cache()
    rings = sum(isinstance(c, RotatingKVCache) for c in cache)
    assert (budget.global_visits, budget.sliding_visits) == (len(cache) // 2 - rings, rings)
    assert budget.prefill_rows == 4096


def test_ordinary_budget_counts_live_restore_snapshots(monkeypatch):
    from mlx2.runtime.models.cache import record_state_checkpoints

    monkeypatch.setenv("MLX_LM_STATE_CHECKPOINT_MAX", "4")
    monkeypatch.setenv("MLX_LM_STATE_CHECKPOINT_STRIDE", "128")
    config = TextConfig.from_dict({**TINY, "sliding_window_size": 256})
    budget = Inkling2CacheBudget.from_config(config, prefill_rows=128)
    model = Model(config)
    cache = model.make_cache()
    # Populate the real cache planes at their serving dtype without building
    # an attention graph; record the snapshots the ordinary prefill loop takes.
    block = mx.zeros((1, config.num_key_value_heads, 128, config.head_dim), mx.bfloat16)
    for pos in range(128, 513, 128):
        for attn in cache[::2]:
            attn.update_and_fetch(block, block)
        record_state_checkpoints(cache, [pos])
    saved = sum(k.nbytes + v.nbytes for c in cache if isinstance(c, RotatingKVCache)
                for _, k, v in c._checkpoints)
    assert saved > 0
    assert budget.project(512) >= _allocated(cache) + saved


def test_visit_geometry_and_draft_term():
    config = TextConfig.from_dict(TINY)
    plain = Inkling2CacheBudget.from_config(config)
    visits = config.layer_indices
    assert plain.global_visits + plain.sliding_visits == len(visits)
    assert plain.sliding_visits == sum(config.layer_is_sliding(i) for i in visits)
    draft = {"num_hidden_layers": 3, "local_layer_ids": [0, 2], "num_key_value_heads": 2,
             "head_dim": 32, "sliding_window_size": 4, "target_layer_ids": [1, 2, 3, 4],
             "hidden_size": 128}
    spec = Inkling2CacheBudget.from_config(config, draft=draft, num_draft=7)
    assert spec.conv_copies == 9
    assert spec.project(100) > plain.project(100)
    with pytest.raises(ValueError):
        plain.project(-1)
    assert plain.as_dict()["schema"] == "inkling2-visit-kv-shortconv-geometry-v1"


@pytest.mark.skipif(not REAL.exists(), reason="Inkling-2-Mini config not on this host")
def test_real_geometry_covers_measured_apcv2_planes():
    """APCv2 COW plane bytes per entry on Inkling-2-Mini@440956c8 (GSM8K,
    ~650-token prompts, 16eed37f5-dsgsm8k): ring 122.3 MB, global KV 29.0 MB,
    short conv 3.8 MB, DSpark draft 9.8 MB."""
    raw = json.loads(REAL.read_text())
    config = TextConfig.from_dict(raw["text_config"])
    plain = Inkling2CacheBudget.from_config(config)
    assert (plain.global_visits, plain.sliding_visits) == (8, 40)
    assert plain.project(652) >= 122.3e6 + 29.0e6 + 3.8e6
    spec = Inkling2CacheBudget.from_config(config, draft=raw["mtp_config"], num_draft=7)
    assert spec.project(652) >= 122.3e6 + 29.0e6 + 3.8e6 + 9.8e6
    # 128K context stays a few GiB per lane (global KV dominates).
    assert spec.project(131072) < 6 * 2**30
