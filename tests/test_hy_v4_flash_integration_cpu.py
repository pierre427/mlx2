"""Hy4-flash cross-feature checks after merging the feature branches (CPU).

Each feature branch tested itself alone; these cover the interactions:
fp8 latent rows with the multi-lane AST executor and parent-conditioned
DFly2 trees, vision spans on fp8 rows (rollback refusal, state round trip),
and the descriptor/layout matrix across APCv2, fp8 latent KV, packed Engram
tables and vision.  Tiny random checkpoints only; no real weights.
"""

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from test_hy_v4_flash_cpu import (  # noqa: E402
    LONG,
    _drain,
    _greedy,
    _tiny_artifact,
    build,
    checkpoint,
    tiny_config,
)
from mlx2.runtime.models import hy_v4_flash as H  # noqa: E402


@pytest.fixture(scope="module")
def fp8_tiny():
    config = tiny_config()
    W = checkpoint(config)
    model, drafter = build(config, W)
    model.args.latent_kv = "fp8"
    return config, W, model, drafter


def test_descriptor_matrix_keeps_every_feature_honest():
    from mlx2.adapters import hy_v4_flash as A
    from mlx2.contracts import Capability

    plain = A.descriptor_for()
    assert {Capability.APC_V2, Capability.PREFIX_REUSE, Capability.TOOLS,
            Capability.REASONING, Capability.GRAMMAR} <= plain.capabilities
    assert Capability.VISION not in plain.capabilities
    combo = A.descriptor_for(dfly_selected=True, latent_kv="fp8", vision=True,
                             engram_table_format="affine4_g32")
    assert Capability.VISION in combo.capabilities
    # Approximate fp8 rows are never published as exact prefixes.
    assert Capability.APC_V2 not in combo.capabilities and Capability.PREFIX_REUSE not in combo.capabilities
    assert combo.cache_layout == A.FP8_CACHE_LAYOUT
    assert "approximate" in combo.metadata["engram_tables"]
    assert A.descriptor_for(dfly_selected=True).cache_layout == A.DFLY_CACHE_LAYOUT


def test_fp8_rows_through_the_multilane_ast_route_equal_fp8_greedy(fp8_tiny):
    from mlx2.adapters.hy_v4_flash_ast import Hy4ChainTreeExecutor
    from mlx2.runtime.drafters.hy4_dfly import Hy4DFlyDraftModel
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    _, _, model, head = fp8_tiny
    model.record_layers = frozenset(head.index_layers)
    try:
        draft = Hy4DFlyDraftModel(model, head, num_draft=3, revision="test")
        executor = Hy4ChainTreeExecutor.create()
        batch = ExternalDraftBatchGenerator(
            model, draft_model=draft, binding="test", num_draft=3, prefill_step_size=4,
            completion_batch_size=2, dynamic_singleton_tree=True, dynamic_tree_max_width=4,
            target_tree_executor_provider=executor, tree_node_budget_by_lanes={1: 7, 2: 3, 3: 1, 4: 1})
        prompts = [LONG[9:21], LONG[60:81]]
        uids = batch.insert(prompts, max_tokens=[10, 10], sampling_configs=[{"sampling_temp": 0}] * 2)
        out = _drain(batch, uids, 10)
        for uid, prompt in zip(uids, prompts):
            assert out[uid][:10] == _greedy(model, prompt, 10)
        assert draft.stats["block_drafts"] > 0
        assert type(model.make_cache()[0]) is H.HY4FlashFP8LayerCache
    finally:
        model.record_layers = frozenset()


def test_conditioned_tree_tails_read_fp8_history(fp8_tiny):
    """DFly2 conditioned trees on fp8 target rows: every node equals its own
    sequential recomputation (parent-conditioned) through the fp8 history."""
    _, _, model, head = fp8_tiny
    model.record_layers = frozenset(head.index_layers)
    try:
        cache = model.make_cache()
        hidden = model.hidden(mx.array([LONG[:30]]), cache)
    finally:
        model.record_layers = frozenset()
    h = head.block_hidden(model, cache, mx.array([7]), hidden[:, -1], anchor=30, num_draft=3)
    tokens, logits = head.propose(model, cache, mx.array([7]), hidden[:, -1], anchor=30, num_draft=3)
    previous = mx.array([7])
    for step in range(3):
        row = head.step_logits(model, h[:, step + 1], previous)
        np.testing.assert_allclose(np.array(row), np.array(logits[:, step]), rtol=1e-5, atol=1e-5)
        previous = tokens[:, step]


def test_vision_spans_on_fp8_rows_refuse_split_and_survive_state():
    cache = H.HY4FlashFP8LayerCache(H.INDEXED, window=8, ratio=2, rank=12, rope=4)
    rows = mx.zeros((1, 20, 16))
    cache.append(rows)
    cache.mrope_spans = ((5, 5, 1, 2, 2),)  # a vision span covering slots 5..(5+len)
    end = H.span_end(cache.mrope_spans[0]) if hasattr(H, "span_end") else None
    assert end is not None and 5 < end <= 20
    assert not cache.can_trim(20 - (end - 1))  # landing inside the span
    assert cache.can_trim(20 - end)  # landing right after it
    restored = type(cache).from_state(cache.state, cache.meta_state)
    assert restored.mrope_spans == cache.mrope_spans and type(restored) is H.HY4FlashFP8LayerCache


def test_adapter_fp8_and_dfly_route_has_no_apc_declaration(tmp_path):
    from mlx2.adapters import hy_v4_flash as A
    from mlx2.contracts import Capability

    config = tiny_config()
    W = checkpoint(config)
    adapter = A.HYV4FlashAdapter(str(_tiny_artifact(tmp_path, config, W)),
                                 execution_policy={"draft_model": "embedded_dfly", "latent_kv": "fp8"})
    try:
        assert Capability.APC_V2 not in adapter.descriptor.capabilities
        # The external-draft mixin folds the route identity into the layout.
        assert adapter.diagnostics()["latent_kv"]["cache_layout"].startswith(A.FP8_CACHE_LAYOUT)
        d = adapter.diagnostics()
        assert d["approximate_state"] and d["route"] == "external_draft"
        batch = adapter.create_external_batch(completion_batch_size=2, prefill_step_size=4)
        prompts = [LONG[9:21], LONG[30:44]]
        uids = batch.insert(prompts, max_tokens=[8, 8], sampling_configs=[{"sampling_temp": 0}] * 2)
        out = _drain(batch, uids, 8)
        for uid, prompt in zip(uids, prompts):
            assert out[uid][:8] == _greedy(adapter.model, prompt, 8)
    finally:
        adapter.close()


@pytest.mark.parametrize("policy,suffix", [
    ({}, None),
    ({"engram_precision": "vllm"}, "+engram-precision-vllm"),
    ({"latent_kv": "fp8"}, "fp8"),
    ({"draft_model": "embedded_dfly"}, "dfly2"),
])
def test_every_numerics_knob_changes_the_apc_namespace_and_layouts_agree(tmp_path, policy, suffix):
    from mlx2.adapters import hy_v4_flash as A

    config = tiny_config()
    W = checkpoint(config)
    path = _tiny_artifact(tmp_path, config, W)
    adapter = A.HYV4FlashAdapter(str(path), execution_policy=policy or None)
    try:
        layout = adapter.layout
        assert adapter.descriptor.cache_layout == layout
        assert adapter.diagnostics()["latent_kv"]["cache_layout"] == layout
        if suffix is None:
            # Band attention 512 is the Hy4 default (2026-11-21): its own namespace.
            assert layout == A.CACHE_LAYOUT + "+swa-band512-v1+sparse-prefill-v1"
        else:
            assert layout != A.CACHE_LAYOUT + "+swa-band512-v1+sparse-prefill-v1" and suffix in layout
    finally:
        adapter.close()


def test_numerics_layout_separates_engram_formats_and_profiles():
    from mlx2.adapters import hy_v4_flash as A

    layouts = {A.numerics_layout(A.CACHE_LAYOUT, engram_table_format=f, engram_precision=p)
               for f in ("bf16", "fp8_e4m3_row", "affine4_g32") for p in ("transformers", "vllm")}
    assert len(layouts) == 6
