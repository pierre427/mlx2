"""Approximate KV composed with native self-MTP (target-only quantization).

``compose_mtp`` quantizes each lane's target attention planes and keeps the MTP
head's draft cache exact.  The segmented true-batched self-MTP path gets a
quantized row view (``SegmentedBatchQuantizedKVCache``) so quantized lanes do
not fall back to per-lane B1 forwards.  CPU only.
"""

import json
import time

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mlx2 import memory, serving
from mlx2.qualification import required_feature_checks
from mlx2.runtime import os_memory
from mlx2.runtime import segmented_self_mtp as segmented
from mlx2.runtime.approximate_kv import (
    KVQuantizationDescriptor,
    ServingApproximateKVPolicy,
    standard_kv_quantization_operations,
)
from mlx2.runtime.models.cache import KVCache, QuantizedKVCache
from mlx2.runtime.segmented_batch_cache import (
    SegmentedBatchUnsupported,
    build_segmented_batch_cache_group,
)
from mlx2.runtime.segmented_plain_kv import SegmentedBatchQuantizedKVCache
from mlx2.server import approximate_kv_mode, build_parser
from mlx2.serving import ServingEngine

from test_apc_hits_hybrid_gdn_self_mtp import make_adapter, tiny_qwen38_mtp

OPS = dict(standard_kv_quantization_operations(group_size=32))
# A deliberately lossy operation so the quantized route is distinguishable
# from the exact one on a tiny random model (tests only; never declared by a
# real adapter).
OPS["kv_q2"] = KVQuantizationDescriptor(2, 2, group_size=32)
PROMPT = list(range(1, 60))


def amplified_model():
    model, vocab = tiny_qwen38_mtp()
    for layer in model.model.layers:
        attention = getattr(layer, "self_attn", None)
        if attention is not None:
            attention.v_proj.weight = attention.v_proj.weight * 8
            attention.o_proj.weight = attention.o_proj.weight * 8
    mx.eval(model.parameters())
    return model, vocab


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "src"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    # Production Qwen3.8/3.6 adapters select segmented self-MTP.
    monkeypatch.setenv("MLX_LM_SEGMENTED_SELF_MTP", "1")


def engine(*, mtp, approximate_kv=None, max_lanes=1):
    model, vocab = amplified_model()
    adapter = make_adapter(model, vocab)
    adapter.approximate_kv_operations = lambda self: OPS
    result = ServingEngine(
        "tiny", adapter_factory=adapter, qualification_mode=True, mtp=mtp,
        approximate_kv=approximate_kv, max_lanes=max_lanes,
    )
    assert result.ready.wait(60), result.error
    return result


def collect(job):
    text = ""
    while True:
        event = job.events.get(timeout=120)
        if "error" in event:
            raise AssertionError(event)
        if "delta" in event:
            text += event["delta"].get("content", "")
        if "finish_reason" in event:
            return [int(t) for t in text.split()], event["receipt"]


def run(eng, tokens, max_tokens=16):
    return collect(eng.submit({"tokens": list(tokens), "max_tokens": max_tokens, "temperature": 0}))


def segmented_delta(before):
    after = segmented.segmented_self_mtp_stats()
    return {
        key: after[key] - before.get(key, 0)
        for key, value in after.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }


def served(mtp, approximate_kv=None):
    eng = engine(mtp=mtp, approximate_kv=approximate_kv)
    before = dict(segmented.segmented_self_mtp_stats())
    try:
        tokens, receipt = run(eng, PROMPT)
        time.sleep(1.3)
        status = eng.status()
    finally:
        eng.close()
    return tokens, receipt, status, segmented_delta(before)


def test_policy_key_is_strict_default_off_and_hash_stable():
    assert ServingApproximateKVPolicy.from_value(None).compose_mtp is False
    plain = ServingApproximateKVPolicy.from_value({"operation": "kv_q8", "enabled": True})
    # Existing ordinary-route settings (and their qualification hashes) are
    # unchanged: the key appears only when selected.
    assert "compose_mtp" not in plain.as_dict()
    composed = ServingApproximateKVPolicy.from_value(
        {"operation": "kv_q8", "enabled": True, "compose_mtp": True}
    )
    assert composed.as_dict()["compose_mtp"] is True
    with pytest.raises(ValueError, match="compose_mtp"):
        ServingApproximateKVPolicy.from_value(
            {"operation": "kv_q8", "enabled": True, "compose_mtp": "yes"}
        )


def test_engine_refuses_mtp_without_compose_and_compose_without_mtp():
    with pytest.raises(ValueError, match="MTP unless compose_mtp"):
        ServingEngine(
            "unused", qualification_mode=True, mtp=True,
            approximate_kv={"operation": "kv_q8", "enabled": True},
        )
    with pytest.raises(ValueError, match="compose_mtp requires native MTP"):
        ServingEngine(
            "unused", qualification_mode=True, mtp=False,
            approximate_kv={"operation": "kv_q8", "enabled": True, "compose_mtp": True},
        )
    with pytest.raises(ValueError, match="prompt lookup"):
        ServingEngine(
            "unused", qualification_mode=True, mtp=False, prompt_lookup=True,
            approximate_kv={"operation": "kv_q8", "enabled": True},
        )


def test_cli_accepts_compose_only_on_native_mtp():
    parser = build_parser()
    base = ["--model", "m"] if any(
        action.dest == "model" and action.option_strings for action in parser._actions
    ) else ["m"]
    composed = json.dumps({"operation": "kv_q8", "enabled": True, "compose_mtp": True})
    args = parser.parse_args(base + ["--qualification-mode", "--approximate-kv", composed])
    assert approximate_kv_mode(args, native_mtp=True)["compose_mtp"] is True
    ordinary = parser.parse_args(
        base + ["--ordinary", "--qualification-mode", "--approximate-kv", composed]
    )
    with pytest.raises(ValueError, match="requires native MTP"):
        approximate_kv_mode(ordinary, native_mtp=False)
    plain = json.dumps({"operation": "kv_q8", "enabled": True})
    args = parser.parse_args(base + ["--qualification-mode", "--approximate-kv", plain])
    with pytest.raises(ValueError, match="compose_mtp"):
        approximate_kv_mode(args, native_mtp=True)


def test_mtp_quantized_target_matches_ordinary_quantized_route(host):
    """Speculation stays lossless relative to the quantized target."""
    exact, _, _, _ = served(True)
    ordinary, ordinary_receipt, _, _ = served(False, {"operation": "kv_q2", "enabled": True})
    composed, receipt, status, stats = served(
        True, {"operation": "kv_q2", "enabled": True, "compose_mtp": True}
    )
    # Discriminating: the lossy operation really changed greedy output ...
    assert ordinary != exact
    # ... and the MTP route reproduces the quantized target exactly.
    assert composed == ordinary
    assert ordinary_receipt["approximate_kv"]["route"] == "ordinary"
    block = receipt["approximate_kv"]
    assert block["status"] == "applied" and block["route"] == "mtp"
    assert block["draft_cache"] == "exact"
    assert block["quantized_layers"] == 1 and block["apcv2_publication"] == "skipped"
    assert receipt["mtp"]["route"] == "segmented_self_mtp"
    assert receipt["mtp"]["stats"]["cycles"] > 0
    assert status["approximate_kv"]["mtp_lanes"] == 1
    assert status["approximate_kv"]["compose_mtp"] is True
    assert status["approximate_kv"]["draft_cache"] == "exact"
    assert status["apcv2"]["stores"] == 0
    # Mechanism: the quantized segmented view carried target attention on the
    # true-batched path; the draft (MTP head) attention stayed on exact rows.
    assert stats["quantized_kv_segmented_layers"] > 0
    assert stats["quantized_kv_segmented_attention_calls"] > 0
    assert stats["true_batched_engaged"] > 0
    assert stats["true_batched_declined"] == 0
    assert stats["segmented_attention_calls"] > stats["quantized_kv_segmented_attention_calls"]


def test_two_quantized_mtp_lanes_batch_on_the_segmented_path(host):
    eng = engine(
        mtp=True, max_lanes=2,
        approximate_kv={"operation": "kv_q8", "enabled": True, "compose_mtp": True},
    )
    before = dict(segmented.segmented_self_mtp_stats())
    try:
        jobs = [
            eng.submit({"tokens": PROMPT + [i], "max_tokens": 10, "temperature": 0})
            for i in (70, 71)
        ]
        results = [collect(job) for job in jobs]
        time.sleep(1.3)
        status = eng.status()
    finally:
        eng.close()
    stats = segmented_delta(before)
    assert all(len(tokens) == 10 for tokens, _ in results)
    assert all(r["approximate_kv"]["route"] == "mtp" for _, r in results)
    assert status["approximate_kv"]["mtp_lanes"] == 2
    assert stats["true_batched_engaged"] > 0 and stats["true_batched_declined"] == 0
    assert stats["b1_target_forwards"] == 0
    assert stats["quantized_kv_segmented_attention_calls"] > 0


def _filled(cache, n=5, seed=0):
    mx.random.seed(seed)
    keys = mx.random.normal((1, 1, n, 32))
    values = mx.random.normal((1, 1, n, 32))
    cache.update_and_fetch(keys, values)
    return cache


def test_segmented_quantized_view_matches_b1_quantized_attention():
    from mlx2.runtime.models.base import scaled_dot_product_attention

    rows = [_filled(KVCache(), 5, 1).to_quantized(group_size=32, bits=8),
            _filled(KVCache(), 3, 2).to_quantized(group_size=32, bits=8)]
    reference = [_filled(KVCache(), 5, 1).to_quantized(group_size=32, bits=8),
                 _filled(KVCache(), 3, 2).to_quantized(group_size=32, bits=8)]
    notes = []
    (view,) = build_segmented_batch_cache_group(
        [[rows[0]], [rows[1]]], note=lambda key, amount=1: notes.append(key)
    )
    assert isinstance(view, SegmentedBatchQuantizedKVCache)
    assert "quantized_kv_segmented_layers" in notes
    mx.random.seed(3)
    keys = mx.random.normal((2, 1, 2, 32))
    values = mx.random.normal((2, 1, 2, 32))
    queries = mx.random.normal((2, 2, 2, 32))
    view.prepare(lengths=[2, 2])
    mask = view.make_mask(2)
    assert view.update_and_fetch(keys, values) == (None, None)
    batched = view.bucketed_attention(queries, 0.25, mask)
    for index, row in enumerate(reference):
        k, v = row.update_and_fetch(keys[index : index + 1], values[index : index + 1])
        expected = scaled_dot_product_attention(
            queries[index : index + 1], k, v, row, scale=0.25, mask="causal"
        )
        assert mx.allclose(batched[index : index + 1], expected, atol=1e-5)
    assert [row.offset for row in rows] == [7, 5]
    # Rollback trims the authoritative rows.
    view.finalize()
    assert view.trim_ragged([1, 2]) == [1, 2]
    assert [row.offset for row in rows] == [6, 3]
    snapshot = view.extract(0)
    assert type(snapshot) is QuantizedKVCache and snapshot.offset == 6
    assert snapshot.keys[0] is not rows[0].keys[0]


def test_segmented_quantized_view_refuses_mixed_or_normalized_rows():
    quantized = _filled(KVCache()).to_quantized(group_size=32, bits=8)
    with pytest.raises(SegmentedBatchUnsupported):
        SegmentedBatchQuantizedKVCache([quantized, _filled(KVCache())])
    other = _filled(KVCache()).to_quantized(group_size=32, bits=4)
    with pytest.raises(SegmentedBatchUnsupported):
        SegmentedBatchQuantizedKVCache([quantized, other])
    normalized = _filled(KVCache()).to_quantized(group_size=32, bits=8, normalize=True)
    with pytest.raises(SegmentedBatchUnsupported):
        SegmentedBatchQuantizedKVCache([normalized])
    with pytest.raises(ValueError, match="distinct owners"):
        SegmentedBatchQuantizedKVCache([quantized, quantized])


def test_qualification_demands_fidelity_and_mtp_observation():
    ordinary = {"mtp": False, "approximate_kv": {"enabled": True, "operation": "kv_q8"}}
    required = required_feature_checks(ordinary)
    assert {"feature_approximate_kv", "feature_approximate_kv_fidelity"} <= required
    assert "feature_approximate_kv_mtp" not in required
    composed = {
        "mtp": True,
        "approximate_kv": {"enabled": True, "operation": "kv_q8", "compose_mtp": True},
    }
    assert {
        "feature_approximate_kv",
        "feature_approximate_kv_fidelity",
        "feature_approximate_kv_mtp",
    } <= required_feature_checks(composed)
    assert not {
        "feature_approximate_kv_fidelity", "feature_approximate_kv_mtp"
    } & required_feature_checks({"mtp": False, "approximate_kv": {"enabled": False}})
