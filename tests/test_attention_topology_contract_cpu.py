from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from mlx2.adapters.flash_next import FlashNextAdapter
from mlx2.adapters.muse_glimmer import MuseGlimmerAdapter
from mlx2.adapters.qwen36_27b import Qwen3627BAdapter
from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter
from mlx2.adapters.qwen38_27b import Qwen3827BAdapter
from mlx2.adapters.standard_decoder import StandardDecoderAdapter
from mlx2.runtime.attention_topology import (
    FULL_KV,
    INDEXED_KV,
    RECURRENT,
    SLIDING_KV,
    AttentionTopology,
    AttentionTopologyError,
    apcv2_cache_layout_fingerprint,
    require_adapter_topology,
)


def _adapter(cls, *, config, revision="a" * 64, layout="layout-v1"):
    adapter = object.__new__(cls)
    adapter.identity = {"fingerprint": revision}
    adapter.layout = layout
    if cls in (
        FlashNextAdapter,
        Qwen3827BAdapter,
        Qwen3627BAdapter,
        Qwen3635BA3BAdapter,
    ):
        adapter._attention_topology_config = config
    elif cls is MuseGlimmerAdapter:
        adapter._config = config
    else:
        adapter.config = config
    return adapter


def test_flash_next_declares_indexed_recurrent_and_mtp_planes():
    adapter = _adapter(
        FlashNextAdapter,
        config={
            "layer_types": ["full_attention", "linear_attention", "full_attention"],
            "mtp_num_hidden_layers": 2,
        },
    )
    topology = adapter.attention_topology()
    assert topology.layers == (INDEXED_KV, RECURRENT, INDEXED_KV)
    assert topology.auxiliary_layers == (INDEXED_KV, INDEXED_KV)
    assert topology.indexed_layout == "qwen4-qsa-raw-pooled-index-v1"
    assert topology.recurrent_layout == "gated-delta-state-v1"


@pytest.mark.parametrize("cls", [Qwen3827BAdapter, Qwen3627BAdapter])
def test_dense_qwen_hybrid_topology_is_family_neutral_and_ordered(cls):
    adapter = _adapter(
        cls,
        config={
            "num_hidden_layers": 4,
            "full_attention_interval": 2,
            "layer_types": [
                "linear_attention",
                "full_attention",
                "linear_attention",
                "full_attention",
            ],
            "mtp_num_hidden_layers": 1,
        },
    )
    topology = adapter.attention_topology()
    assert topology.layers == (RECURRENT, FULL_KV, RECURRENT, FULL_KV)
    assert topology.auxiliary_layers == (FULL_KV,)
    assert "qwen" not in topology.schema


def test_muse_declares_full_and_sliding_window_layers():
    adapter = _adapter(
        MuseGlimmerAdapter,
        config={
            "layer_types": ["full_attention", "sliding_attention"] * 2,
            "sliding_window": 2048,
        },
    )
    topology = adapter.attention_topology()
    assert topology.layers == (FULL_KV, SLIDING_KV, FULL_KV, SLIDING_KV)
    assert topology.sliding_window == 2048


def test_qwen36_35b_declares_its_distinct_recurrent_layout():
    adapter = _adapter(
        Qwen3635BA3BAdapter,
        config={
            "num_hidden_layers": 4,
            "full_attention_interval": 2,
            "mtp_num_hidden_layers": 1,
        },
    )
    topology = adapter.attention_topology()
    assert topology.layers == (RECURRENT, FULL_KV, RECURRENT, FULL_KV)
    assert topology.auxiliary_layers == (FULL_KV,)
    assert topology.recurrent_layout == "qwen36-gdn-32v16k-state-v1"


def test_standard_decoder_declares_only_full_kv():
    adapter = _adapter(StandardDecoderAdapter, config={"num_hidden_layers": 3})
    assert adapter.attention_topology().layers == (FULL_KV,) * 3


def test_binding_is_stable_revision_and_layout_bound_and_immutable():
    topology = AttentionTopology(
        state_revision="b" * 64,
        cache_layout="hybrid-v1",
        layers=(FULL_KV, RECURRENT),
        recurrent_layout="gdn-v1",
    )
    assert topology.fingerprint == topology.fingerprint
    assert topology.binding()["topology_fingerprint"] == topology.fingerprint
    with pytest.raises(FrozenInstanceError):
        topology.cache_layout = "other"
    assert AttentionTopology(
        state_revision="c" * 64,
        cache_layout="hybrid-v1",
        layers=(FULL_KV, RECURRENT),
        recurrent_layout="gdn-v1",
    ).fingerprint != topology.fingerprint


def test_selected_capability_fails_closed_on_unknown_or_mismatched_topology():
    class OrdinaryOnly:
        pass

    # Merely constructing/using an ordinary adapter does not inspect topology.
    ordinary = OrdinaryOnly()
    assert ordinary is not None
    with pytest.raises(AttentionTopologyError, match="promotion"):
        require_adapter_topology(ordinary, capability="atomic state promotion")

    adapter = _adapter(StandardDecoderAdapter, config={"num_hidden_layers": 2})
    binding = adapter.attention_topology().binding()
    assert require_adapter_topology(
        adapter, capability="APCv2 state restore"
    ).validate_binding(binding, capability="APCv2 state restore") == binding
    with pytest.raises(AttentionTopologyError, match="revision or layout mismatch"):
        require_adapter_topology(
            adapter, capability="APCv2 state restore"
        ).validate_binding(
            {**binding, "cache_layout": "stale-layout"},
            capability="APCv2 state restore",
        )


def test_apcv2_consumes_declared_topology_but_preserves_legacy_adapters():
    class Legacy:
        layout = "legacy-layout"

    assert apcv2_cache_layout_fingerprint(Legacy()) == "legacy-layout"
    adapter = _adapter(StandardDecoderAdapter, config={"num_hidden_layers": 2})
    fingerprint = apcv2_cache_layout_fingerprint(adapter)
    assert fingerprint == (adapter.layout, adapter.attention_topology().fingerprint)

    adapter.layout = "drifted"
    # The adapter provider reads the same live layout, so force a malformed
    # provider to prove the selected APC seam validates rather than trusts it.
    topology = AttentionTopology(
        state_revision=adapter.identity["fingerprint"],
        cache_layout="stale",
        layers=(FULL_KV,),
    )
    adapter.attention_topology = lambda: topology
    with pytest.raises(AttentionTopologyError, match="revision or layout mismatch"):
        apcv2_cache_layout_fingerprint(adapter)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"layers": ("mystery",)}, "unknown layer kind"),
        ({"layers": (SLIDING_KV,)}, "positive window"),
        ({"layers": (INDEXED_KV,)}, "indexed layout"),
        ({"layers": (RECURRENT,)}, "recurrent layout"),
    ],
)
def test_incomplete_topology_descriptors_are_refused(kwargs, message):
    with pytest.raises(AttentionTopologyError, match=message):
        AttentionTopology(
            state_revision="d" * 64,
            cache_layout="layout-v1",
            **kwargs,
        )
