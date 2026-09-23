"""KV quantization through the approximate-state seam (CPU, tiny hybrid model)."""

import json
import time
from types import SimpleNamespace as NS

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mlx2 import memory, serving
from mlx2.adapters.base import approximate_kv_operations
from mlx2.qualification import required_feature_checks
from mlx2.runtime import os_memory
from mlx2.runtime.approximate_kv import (
    KVQuantizationDescriptor,
    KVQuantizationOperation,
    LaneKVState,
    ServingApproximateKVPolicy,
    operation_revision,
    prompt_cache_is_approximate,
    standard_kv_quantization_operations,
)
from mlx2.runtime.approximate_state import (
    ApproximateKVController,
    ApproximateKVPolicy,
    ApproximateStateError,
)
from mlx2.runtime.models.cache import KVCache, RotatingKVCache
from mlx2.server import approximate_kv_mode, build_parser
from mlx2.serving import ServingEngine

POLICY = {"operation": "kv_k8v4", "enabled": True}


def tiny_model():
    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_27b import TextModel

    args = TextModelArgs(
        model_type="qwen3_5",
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        vocab_size=128,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
        full_attention_interval=4,
        mtp_num_hidden_layers=0,
        partial_rotary_factor=0.5,
        rope_parameters=None,
        max_position_embeddings=256,
    )
    mx.random.seed(7)
    model = TextModel(args)
    model.eval()
    mx.eval(model.parameters())
    return model


class Detok:
    def __init__(self):
        self.last_segment = ""

    def reset(self):
        self.last_segment = ""

    def add_token(self, token):
        self.last_segment = f"{int(token)} "

    def finalize(self):
        pass


class Parser:
    stopped = False
    tool_count = 0

    def push(self, text, final=False):
        return [{"content": text}] if text else []


class Tokenizer:
    vocab_size = 128
    eos_token_ids = []

    @property
    def detokenizer(self):
        return Detok()


def make_adapter(model, *, operations):
    class Adapter:
        max_context = 256
        identity = {"fingerprint": "tiny-hybrid"}
        environment = {}
        layout = "tiny-layout"
        tokenizer = Tokenizer()

        def __init__(self, _path):
            self.model = model

        def profile_name(self, _mtp):
            return "tiny-ordinary"

        def execution_config(self, **_kwargs):
            return {"num_draft": 0}

        def prompt_tokens(self, request):
            return list(request["tokens"])

        def output_parser(self, _request):
            return Parser()

        def diagnostics(self):
            return {}

        def close(self):
            pass

    if operations is not None:
        Adapter.approximate_kv_operations = lambda self: operations
    return Adapter


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "src"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)


def run(engine, tokens, *, max_tokens=6):
    job = engine.submit({"tokens": list(tokens), "max_tokens": max_tokens, "temperature": 0})
    text = ""
    while True:
        event = job.events.get(timeout=60)
        if "error" in event:
            raise AssertionError(event)
        if "delta" in event:
            text += event["delta"].get("content", "")
        if "finish_reason" in event:
            return [int(t) for t in text.split()], event["receipt"]


def settled_status(engine):
    # APCv2 counters reach ``status`` through the worker's 1 s snapshot.
    time.sleep(1.3)
    return engine.status()


def engine_for(model, *, operations, **kwargs):
    engine = ServingEngine(
        "tiny",
        adapter_factory=make_adapter(model, operations=operations),
        qualification_mode=True,
        mtp=False,
        **kwargs,
    )
    engine.thread.join(0)
    return engine


OPS32 = standard_kv_quantization_operations(group_size=32)
LONG = list(range(1, 40))
SHORT = list(range(60, 72))


def test_policy_is_default_off_and_strict():
    assert ServingApproximateKVPolicy.from_value(None).enabled is False
    assert ServingApproximateKVPolicy.from_value(False).as_dict() == {
        "enabled": False, "operation": None, "start_tokens": 0, "evidence": [],
    }
    # Naming an operation does not enable it.
    assert ServingApproximateKVPolicy.from_value({"operation": "kv_q8"}).enabled is False
    for bad in (
        True,
        "kv_q8",
        {"enabled": True},
        {"operation": "", "enabled": True},
        {"operation": "kv_q8", "enabled": "yes"},
        {"operation": "kv_q8", "enabled": True, "start_tokens": -1},
        {"operation": "kv_q8", "enabled": True, "start_tokens": True},
        {"operation": "kv_q8", "enabled": True, "evidence": "receipt.json"},
        {"operation": "kv_q8", "enabled": True, "bits": 8},
    ):
        with pytest.raises(ValueError):
            ServingApproximateKVPolicy.from_value(bad)


def test_descriptor_and_revision_binding():
    assert OPS32["kv_q8"].as_dict() == {
        "key_bits": 8, "value_bits": 8, "group_size": 32, "rotate": False, "start": 0,
    }
    assert OPS32["kv_k8v4"].value_bits == 4
    for bad in (dict(key_bits=7, value_bits=8), dict(key_bits=8, value_bits=8, group_size=48),
                dict(key_bits=8, value_bits=8, start=16), dict(key_bits=True, value_bits=8)):
        with pytest.raises(ValueError):
            KVQuantizationDescriptor(**bad)
    base = operation_revision("artifact-a", "kv_k8v4", OPS32["kv_k8v4"])
    assert base != operation_revision("artifact-b", "kv_k8v4", OPS32["kv_k8v4"])
    assert base != operation_revision("artifact-a", "kv_q8", OPS32["kv_q8"])
    assert base != operation_revision(
        "artifact-a", "kv_k8v4", standard_kv_quantization_operations()["kv_k8v4"]
    )


def test_adapter_capability_defaults_to_empty_and_real_adapters_are_explicit():
    from mlx2.adapters.flash_next import FlashNextAdapter
    from mlx2.adapters.muse_glimmer import MuseGlimmerAdapter
    from mlx2.adapters.north_mini_code import NorthMiniCodeAdapter
    from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter

    assert approximate_kv_operations(object()) == {}
    new = lambda cls: cls.__new__(cls)  # capability needs no loaded artifact
    assert approximate_kv_operations(new(FlashNextAdapter)) == {}
    assert approximate_kv_operations(new(MuseGlimmerAdapter)) == {}
    assert approximate_kv_operations(new(NorthMiniCodeAdapter)) == {}
    for cls in (Qwen3827BAdapter, Qwen3635BA3BAdapter):
        assert set(approximate_kv_operations(new(cls))) == {"kv_q8", "kv_k8v4"}


def test_controller_candidate_mode_and_private_stage_stay_fail_closed():
    model = tiny_model()
    operation = KVQuantizationOperation(
        "kv_k8v4", OPS32["kv_k8v4"], adapter_fingerprint="tiny"
    )
    unqualified = ApproximateKVController(ApproximateKVPolicy("kv_k8v4", enabled=True))
    state = LaneKVState(operation.revision, tuple(model.make_cache()))
    with pytest.raises(ApproximateStateError, match="not qualified"):
        unqualified.apply(
            request_id="r", state_revision=operation.revision, state=state,
            adapters={"kv_k8v4": operation},
        )
    with pytest.raises(ApproximateStateError, match="private"):
        unqualified.apply(
            request_id="r", state_revision=operation.revision, state=state,
            adapters={"kv_k8v4": operation}, candidate=True, stage=lambda s: s,
        )
    with pytest.raises(ApproximateStateError, match="revision mismatch"):
        unqualified.apply(
            request_id="r", state_revision="other", state=state,
            adapters={"kv_k8v4": operation}, candidate=True,
        )
    from mlx2.runtime.approximate_kv import stage_lane_state

    updated, receipt = unqualified.apply(
        request_id="r", state_revision=operation.revision, state=state,
        adapters={"kv_k8v4": operation}, candidate=True, stage=stage_lane_state,
    )
    assert receipt["status"] == "applied" and receipt["qualified"] is False
    assert receipt["target_revision"] == operation.revision + ":kv_k8v4"
    assert updated.quantized_planes == 1
    assert prompt_cache_is_approximate(updated.planes)
    assert not prompt_cache_is_approximate(state.planes)


def test_operation_refuses_unsupported_planes():
    operation = KVQuantizationOperation(
        "kv_k8v4", OPS32["kv_k8v4"], adapter_fingerprint="tiny"
    )
    with pytest.raises(ApproximateStateError, match="Asymmetric"):
        operation.apply(
            LaneKVState(operation.revision, (RotatingKVCache(max_size=8, keep=0),))
        )

    class Refusing(KVCache):
        kv_quantization_unsupported = "index ledger is exact-only"

    with pytest.raises(ApproximateStateError, match="exact-only"):
        operation.apply(LaneKVState(operation.revision, (Refusing(),)))
    with pytest.raises(ApproximateStateError, match="no quantizable"):
        operation.apply(LaneKVState(operation.revision, ()))


def test_engine_construction_fails_closed():
    with pytest.raises(ValueError, match="qualification"):
        ServingEngine("unused", mtp=False, approximate_kv=POLICY)
    with pytest.raises(ValueError, match="MTP"):
        ServingEngine("unused", qualification_mode=True, mtp=True, approximate_kv=POLICY)
    with pytest.raises(ValueError, match="prompt lookup"):
        ServingEngine(
            "unused", qualification_mode=True, mtp=False, prompt_lookup=True,
            approximate_kv=POLICY,
        )
    with pytest.raises(ValueError, match="cache capsules"):
        ServingEngine(
            "unused", qualification_mode=True, mtp=False, cache_capsules=True,
            approximate_kv=POLICY,
        )
    with pytest.raises(ValueError, match="Spomin"):
        ServingEngine(
            "unused", qualification_mode=True, mtp=False, max_lanes=1,
            spomin_live_surgery={"enabled": True, "capacity_tokens": 16},
            approximate_kv=POLICY,
        )
    with pytest.raises(ValueError, match="max_lanes=1"):
        ServingEngine(
            "unused", qualification_mode=True, mtp=False, max_lanes=2,
            approximate_kv=dict(POLICY, start_tokens=8),
        )
    with pytest.raises(ValueError, match="unknown approximate KV policy keys"):
        ServingEngine(
            "unused", qualification_mode=True, mtp=False,
            approximate_kv=dict(POLICY, bits=4),
        )


def test_approximate_route_rejects_interior_checkpoint_policy(host):
    engine = engine_for(
        tiny_model(),
        operations=OPS32,
        approximate_kv=POLICY,
        execution_policy={
            "apc_interior_checkpoints": {"count": 3, "min_stride": 4}
        },
    )
    try:
        engine.thread.join(10)
        assert not engine.ready.is_set()
        assert "APCv2 interior checkpoints" in engine.error
        assert "approximate KV" in engine.error
    finally:
        engine.close()


def test_adapter_without_the_capability_is_refused(host):
    engine = engine_for(tiny_model(), operations=None, approximate_kv=POLICY)
    try:
        engine.thread.join(10)
        assert not engine.ready.is_set()
        assert "does not declare approximate KV operation 'kv_k8v4'" in engine.error
        with pytest.raises(RuntimeError):
            engine.submit({"tokens": [1, 2, 3]})
    finally:
        engine.close()


def test_external_draft_route_is_refused(host):
    adapter = make_adapter(tiny_model(), operations=OPS32)
    adapter.execution_config = lambda self, **_kw: {
        "num_draft": 2, "backend": "external_draft",
    }
    engine = ServingEngine(
        "tiny", adapter_factory=adapter, qualification_mode=True, mtp=False,
        approximate_kv=POLICY,
    )
    try:
        engine.thread.join(10)
        assert "incompatible with external draft" in engine.error
    finally:
        engine.close()


def test_adapter_whose_cache_cannot_take_the_operation_is_refused(host):
    model = tiny_model()
    exact = model.make_cache
    model.make_cache = lambda: exact()[:3] + [RotatingKVCache(max_size=64, keep=0)]
    engine = engine_for(model, operations=OPS32, approximate_kv=POLICY)
    try:
        engine.thread.join(10)
        assert not engine.ready.is_set()
        assert "Asymmetric" in engine.error
    finally:
        engine.close()


def exact_reference(model, host_tokens):
    engine = engine_for(model, operations=OPS32)
    assert engine.ready.wait(30), engine.error
    try:
        return [run(engine, tokens) for tokens in host_tokens], settled_status(engine)
    finally:
        engine.close()


def test_approximate_lane_is_applied_and_never_reaches_apcv2(host):
    model = tiny_model()
    (reference,), status = exact_reference(model, [LONG])
    assert status["approximate_kv"]["enabled"] is False
    assert status["apcv2"]["stores"] == 2  # prompt boundary + end of request
    assert status["apcv2"]["layer_segments"]["entries"] > 0
    assert status["apcv2"]["idle_disk"]["resident_bytes"] > 0
    assert status["settings"]["approximate_kv"] == {
        "enabled": False, "operation": None, "start_tokens": 0, "evidence": [],
        "descriptor": None,
    }

    engine = engine_for(
        model,
        operations=OPS32,
        approximate_kv=POLICY,
        max_lanes=2,
    )
    assert engine.ready.wait(30), engine.error
    try:
        before = engine.status()
        tokens, receipt = run(engine, LONG)
        again, warm = run(engine, LONG + [41, 42])
        status = settled_status(engine)
    finally:
        engine.close()
    assert len(tokens) == 6 and all(0 <= t < 128 for t in tokens)
    block = receipt["approximate_kv"]
    assert block["status"] == "applied" and block["selected"] is True
    assert block["fidelity"] == "approximate" and block["operation"] == "kv_k8v4"
    assert block["qualified"] is False and block["reason"] == "committed"
    assert block["quantized_layers"] == 1 and block["start_tokens"] == 0
    assert block["descriptor"]["value_bits"] == 4
    assert block["target_revision"] == block["source_revision"] + ":kv_k8v4"
    assert block["apcv2_publication"] == "skipped"
    # No exact-cache entries, bytes or hits are attributable to either lane.
    assert warm["cached_tokens"] == 0
    assert warm["approximate_kv"]["requantized_prefix_tokens"] == 0
    for field in ("stores", "hits", "cached_tokens"):
        assert status["apcv2"][field] == before["apcv2"][field] == 0
    assert status["apcv2"]["layer_segments"]["entries"] == 0
    assert status["apcv2"]["layer_segments"]["logical_bytes"] == 0
    assert status["apcv2"]["idle_disk"]["resident_bytes"] == 0
    assert status["apcv2"]["idle_disk"]["disk_bytes"] == 0
    assert status["counts"]["apcv2_store_skipped_approximate"] == 4
    assert status["approximate_kv"]["applied"] == 2
    assert status["approximate_kv"]["state"] == "candidate"
    assert status["approximate_kv"]["fidelity"] == "approximate"
    assert status["settings"]["approximate_kv"]["descriptor"] == OPS32["kv_k8v4"].as_dict()
    assert reference[0] is not None  # the exact run is the comparison anchor


def test_q8_operation_executes_on_actual_tiny_hybrid_model(host):
    policy = {"operation": "kv_q8", "enabled": True}
    engine = engine_for(
        tiny_model(), operations=OPS32, approximate_kv=policy, max_lanes=1
    )
    assert engine.ready.wait(30), engine.error
    try:
        tokens, receipt = run(engine, LONG)
        status = settled_status(engine)
    finally:
        engine.close()
    assert len(tokens) == 6 and all(0 <= token < 128 for token in tokens)
    block = receipt["approximate_kv"]
    assert block["status"] == "applied"
    assert block["operation"] == "kv_q8"
    assert block["descriptor"]["key_bits"] == 8
    assert block["descriptor"]["value_bits"] == 8
    assert block["quantized_layers"] == 1
    assert block["apcv2_publication"] == "skipped"
    assert status["approximate_kv"]["applied"] == 1
    assert status["apcv2"]["stores"] == 0


def test_exact_lane_after_approximate_traffic_is_bit_identical(host):
    model = tiny_model()
    (long_exact, short_exact), _ = exact_reference(model, [LONG, SHORT])

    engine = engine_for(
        model, operations=OPS32, max_lanes=1,
        approximate_kv=dict(POLICY, start_tokens=20),
    )
    assert engine.ready.wait(30), engine.error
    try:
        _, approximate = run(engine, LONG)
        mid = settled_status(engine)
        short_tokens, short_receipt = run(engine, SHORT)
        # The exact lane stored its state; an approximate lane may read it, but
        # only through a privately requantized copy.
        prefix = SHORT + short_tokens[:4] + list(range(80, 100))
        _, requantized = run(engine, prefix)
        short_again, short_again_receipt = run(engine, SHORT)
        status = settled_status(engine)
    finally:
        engine.close()
    assert approximate["approximate_kv"]["status"] == "applied"
    assert mid["apcv2"]["stores"] == 0
    assert short_receipt["approximate_kv"]["status"] == "declined"
    assert short_receipt["approximate_kv"]["reason"] == "below_start_tokens"
    assert short_receipt["approximate_kv"]["fidelity"] == "exact"
    assert short_tokens == short_exact[0]
    assert long_exact[0] is not None
    assert requantized["approximate_kv"]["status"] == "applied"
    assert requantized["cached_tokens"] > 0
    assert (
        requantized["approximate_kv"]["requantized_prefix_tokens"]
        == requantized["cached_tokens"]
    )
    # The shared exact entry survived the approximate reader untouched.
    assert short_again == short_exact[0]
    assert short_again_receipt["cached_tokens"] > 0
    assert status["approximate_kv"]["requantized_prefix_hits"] == 1
    assert status["approximate_kv"]["applied"] == 2
    assert status["approximate_kv"]["declined"] == 2
    # Only the two exact lanes stored (boundary + end each).
    assert status["apcv2"]["stores"] == 4
    assert status["counts"]["apcv2_store_skipped_approximate"] == 4


def test_operation_refuses_planes_from_another_producer():
    from mlx2.runtime.apc_v2 import APCv2
    from mlx2.runtime.approximate_kv import (
        SourceBoundKVQuantization,
        lane_source_revision,
        source_state_revision,
        stage_lane_state,
    )
    from mlx2.runtime.cow_cache import freeze_prompt_cache

    operation = KVQuantizationOperation("kv_q8", OPS32["kv_q8"], adapter_fingerprint="model-a")
    own = source_state_revision("model-a", "layout-a")
    bound = {"kv_q8": SourceBoundKVQuantization(operation, own)}
    controller = ApproximateKVController(
        ApproximateKVPolicy("kv_q8", enabled=True, qualified=True, evidence=("r",))
    )

    def branch(adapter, layout):
        cache = KVCache()
        cache.update_and_fetch(mx.zeros((1, 1, 4, 32)), mx.zeros((1, 1, 4, 32)))
        key = APCv2.key(adapter, adapter=adapter, cache_layout_fingerprint=layout)
        frozen, _ = freeze_prompt_cache([cache], key=key, tokens=range(4), cache_type="kv")
        return frozen.branch()

    def apply(planes, *, warm=True):
        revision = lane_source_revision(planes, fresh_revision=own, warm=warm)
        return controller.apply(
            request_id="r", state_revision=revision,
            state=LaneKVState(revision, tuple(planes)), adapters=bound,
            stage=stage_lane_state,
        )

    _, receipt = apply(branch("model-a", "layout-a"))
    assert receipt["status"] == "applied" and receipt["source_revision"] == own
    assert receipt["source_revision"] != operation.revision
    for adapter, layout in (("model-b", "layout-a"), ("model-a", "layout-b")):
        with pytest.raises(ApproximateStateError, match="revision mismatch"):
            apply(branch(adapter, layout))
    with pytest.raises(ApproximateStateError, match="no provenance"):
        apply([KVCache()])
    assert apply([KVCache()], warm=False)[1]["source_revision"] == own


def test_warm_prefix_from_another_producer_fails_closed(host, monkeypatch):
    import dataclasses

    from mlx2.runtime import apc_v2
    from mlx2.runtime.approximate_kv import source_state_revision

    forge = []
    original = apc_v2.APCv2.lookup

    def lookup(self, key, tokens, **kwargs):
        hit = original(self, key, tokens, **kwargs)
        metadata = getattr(hit.cache, "cow_metadata", None)
        if forge and metadata is not None and hit.cached_tokens:
            # The restored planes claim another artifact produced them.
            hit.cache.cow_metadata = dataclasses.replace(
                metadata, key=dataclasses.replace(metadata.key, adapter="other")
            )
        return hit

    monkeypatch.setattr(apc_v2.APCv2, "lookup", lookup)
    engine = engine_for(
        tiny_model(), operations=OPS32, max_lanes=1,
        approximate_kv=dict(POLICY, start_tokens=20),
    )
    assert engine.ready.wait(30), engine.error
    try:
        short_tokens, _ = run(engine, SHORT)
        prefix = SHORT + short_tokens[:4] + list(range(80, 100))
        forge.append(True)
        job = engine.submit({"tokens": prefix, "max_tokens": 6, "temperature": 0})
        refused = job.events.get(timeout=60)
        forge.clear()
        _, warm = run(engine, prefix)
        alive = engine.thread.is_alive()
    finally:
        engine.close()
    assert refused.get("status") == 500 and "finish_reason" not in refused
    assert alive
    block = warm["approximate_kv"]
    assert block["status"] == "applied" and warm["cached_tokens"] > 0
    assert block["source_revision"] == source_state_revision("tiny-hybrid", "tiny-layout")


def test_structurally_approximate_cache_is_never_stored_even_if_unflagged():
    engine = ServingEngine.__new__(ServingEngine)
    from collections import Counter

    engine.counts = Counter()
    value = mx.zeros((1, 1, 4, 32))
    cache = KVCache()
    cache.update_and_fetch(value, value)

    class APC:
        calls = 0

        def store(self, *_args, **_kwargs):
            self.calls += 1
            return NS(stored=True)

    apc = APC()
    assert engine._publish_checkpoint(apc, "k", [1], [cache]) is True
    assert engine._publish_checkpoint(apc, "k", [1], [cache], approximate=True) is False
    quantized = [cache.to_quantized(group_size=32, bits=8)]
    assert engine._publish_checkpoint(apc, "k", [1], quantized) is False
    assert apc.calls == 1
    assert engine.counts["apcv2_store_skipped_approximate"] == 2


def test_parallel_samples_prefill_independently_when_approximate(host):
    engine = engine_for(tiny_model(), operations=OPS32, approximate_kv=POLICY, max_lanes=2)
    assert engine.ready.wait(30), engine.error
    try:
        jobs = engine.submit_many(
            [{"tokens": LONG, "max_tokens": 3, "temperature": 0, "seed": i} for i in range(2)]
        )
        receipts = []
        for job in jobs:
            while True:
                event = job.events.get(timeout=60)
                assert "error" not in event, event
                if "finish_reason" in event:
                    receipts.append(event["receipt"])
                    break
        status = settled_status(engine)
    finally:
        engine.close()
    assert all(r["approximate_kv"]["status"] == "applied" for r in receipts)
    assert all(r["parallel_prefill"] is None for r in receipts)
    assert status["counts"]["approximate_kv_fanout_bypassed"] == 1
    assert status["apcv2"]["stores"] == 0


def test_cli_policy_requires_qualification_and_ordinary_route():
    parser = build_parser()
    raw = json.dumps(POLICY)
    base = ["--model", "m"] if any(
        action.dest == "model" and action.option_strings for action in parser._actions
    ) else ["m"]
    args = parser.parse_args(base + ["--ordinary", "--qualification-mode", "--approximate-kv", raw])
    assert approximate_kv_mode(args, native_mtp=False) == {
        "enabled": True, "operation": "kv_k8v4", "start_tokens": 0, "evidence": [],
    }
    assert approximate_kv_mode(parser.parse_args(base), native_mtp=True) is None
    with pytest.raises(ValueError, match="--ordinary"):
        approximate_kv_mode(args, native_mtp=True)
    unqualified = parser.parse_args(base + ["--ordinary", "--approximate-kv", raw])
    with pytest.raises(ValueError, match="qualification"):
        approximate_kv_mode(unqualified, native_mtp=False)
    broken = parser.parse_args(base + ["--qualification-mode", "--approximate-kv", "{"])
    with pytest.raises(ValueError, match="JSON"):
        approximate_kv_mode(broken, native_mtp=False)


def test_qualification_requires_observed_application():
    settings = {"mtp": False, "approximate_kv": {"enabled": True}}
    assert "feature_approximate_kv" in required_feature_checks(settings)
    assert "feature_approximate_kv" not in required_feature_checks(
        {"mtp": False, "approximate_kv": {"enabled": False}}
    )
    assert "feature_approximate_kv" not in required_feature_checks({"mtp": False})
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts" / "qualify_serving.py"
    spec = importlib.util.spec_from_file_location("qualify_serving_approx", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.feature_observations({"approximate_kv": {"applied": 3}})["approximate_kv"] == 3
    assert module.feature_observations({})["approximate_kv"] == 0
