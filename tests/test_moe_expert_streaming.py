"""MoE expert disk streaming: correctness first, policy plumbing second.

Every test here runs on the CPU device on purpose.  The GPU on this host is
leased and these tests must never take it; the arithmetic they check is
device-independent anyway.
"""

import json
import struct

import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402

from mlx2.runtime import expert_atlas  # noqa: E402
from mlx2.runtime import weight_stream  # noqa: E402
from mlx2.runtime.models.switch_layers import SwitchGLU  # noqa: E402
from mlx2.runtime.weight_stream import (  # noqa: E402
    SafetensorsIndex,
    StreamingUnavailable,
    WorkingSetTooSmall,
    install_expert_streaming,
    plan_cache_experts,
    resident_fraction,
)


@pytest.fixture(autouse=True)
def cpu_only():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


NUM_EXPERTS = 16
TOP_K = 3
HIDDEN = 64
INTERMEDIATE = 128
LAYERS = 3
GROUP_SIZE = 64
BITS = 4


class TinyMoE(nn.Module):
    """Smallest thing with the shape that matters: stacked expert tables."""

    def __init__(self):
        super().__init__()
        self.layers = [
            SwitchGLU(HIDDEN, INTERMEDIATE, NUM_EXPERTS, bias=False)
            for _ in range(LAYERS)
        ]

    def __call__(self, x, indices):
        for layer in self.layers:
            x = x + mx.sum(layer(x, indices), axis=-2)
        return x


def _build_model(seed=7):
    mx.random.seed(seed)
    model = TinyMoE()
    nn.quantize(model, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(model.parameters())
    return model


def _flatten(model):
    from mlx.utils import tree_flatten

    return dict(tree_flatten(model.parameters()))


@pytest.fixture
def checkpoint(tmp_path):
    """A quantized MoE saved as safetensors, plus a resident reference."""
    model = _build_model()
    weights = {key: value for (key, value) in _flatten(model).items()}
    path = tmp_path / "model"
    path.mkdir()
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    return (path, weights)


def _reference(weights):
    model = _build_model()
    model.load_weights([(k, v) for (k, v) in weights.items()])
    mx.eval(model.parameters())
    return model


def _inputs(tokens=5, seed=11):
    rng = np.random.default_rng(seed)
    x = mx.array(rng.standard_normal((tokens, HIDDEN)).astype(np.float32))
    indices = mx.array(
        np.stack(
            [rng.choice(NUM_EXPERTS, size=TOP_K, replace=False) for _ in range(tokens)]
        ).astype(np.uint32)
    )
    return (x, indices)


def _run(model, x, indices):
    out = model(x, indices)
    mx.eval(out)
    return np.array(out)


# ---------------------------------------------------------------------------
# byte-range addressing
# ---------------------------------------------------------------------------


def test_index_addresses_expert_rows_byte_exactly(checkpoint):
    (path, weights) = checkpoint
    index = SafetensorsIndex.from_model_path(path)
    name = "layers.0.gate_proj.weight"
    location = index[name]
    assert location.shape[0] == NUM_EXPERTS
    stride = location.row_bytes()
    assert stride * NUM_EXPERTS == location.nbytes

    spec = weight_stream.ExpertSliceSpec.single("weight", location)
    reader = weight_stream.ExpertSliceReader(
        [spec], handles=weight_stream._FileHandles()
    )
    for expert in (0, 1, NUM_EXPERTS - 1):
        (rebuilt,) = reader.materialize(reader.read_bytes(expert))
        assert np.array_equal(np.array(rebuilt), np.array(weights[name][expert]))


def test_unsupported_dtype_is_refused(checkpoint):
    (path, _) = checkpoint
    index = SafetensorsIndex.from_model_path(path)
    location = index["layers.0.gate_proj.weight"]
    broken = weight_stream.TensorLocation(
        path=location.path,
        dtype="F8_E4M3",
        shape=location.shape,
        begin=location.begin,
        end=location.end,
    )
    reader = weight_stream.ExpertSliceReader(
        [weight_stream.ExpertSliceSpec.single("weight", broken)],
        handles=weight_stream._FileHandles(),
    )
    with pytest.raises(StreamingUnavailable):
        reader.materialize(reader.read_bytes(0))


# ---------------------------------------------------------------------------
# THE CORRECTNESS GATE
# ---------------------------------------------------------------------------


def _ceiling_for(path, capacity):
    """Byte ceiling that sizes the per-layer cache to ``capacity`` experts."""
    index = SafetensorsIndex.from_model_path(path)
    per_expert = sum(
        index[f"layers.0.gate_proj.{key}"].row_bytes()
        for key in ("weight", "scales", "biases")
    )
    caches = LAYERS * 3  # one per quantized projection
    return per_expert * caches * capacity


@pytest.mark.parametrize("capacity", [NUM_EXPERTS, 8, TOP_K])
def test_streamed_output_is_bit_identical_to_resident(checkpoint, capacity):
    """Streamed == resident, bit for bit, at every cache size.

    Whether a weight is resident, evicted or being fetched may change timing
    and page-in counts.  It may never change arithmetic.  The smallest
    capacity here forces an eviction on nearly every access.
    """
    (path, weights) = checkpoint
    (x, indices) = _inputs()
    expected = _run(_reference(weights), x, indices)

    streamed = _reference(weights)
    manager = install_expert_streaming(
        streamed,
        path,
        ceiling_bytes=_ceiling_for(path, capacity),
        top_k=TOP_K,
        read_workers=4,
    )
    try:
        actual = _run(streamed, x, indices)
        assert np.array_equal(actual, expected), "streamed output diverged"
        assert actual.dtype == expected.dtype
        assert manager.stats.page_ins > 0, "nothing was actually paged in"
        assert manager.stats.resident_bytes <= manager.plan.ceiling_bytes
    finally:
        manager.close()


def test_page_in_counts_differ_with_capacity(checkpoint):
    """If the cache never evicts, the identity test proved nothing."""
    (path, weights) = checkpoint
    (x, indices) = _inputs()
    results = {}
    for capacity in (NUM_EXPERTS, TOP_K):
        model = _reference(weights)
        manager = install_expert_streaming(
            model, path, ceiling_bytes=_ceiling_for(path, capacity), top_k=TOP_K
        )
        try:
            for _ in range(4):
                _run(model, x, indices)
            results[capacity] = (manager.stats.page_ins, manager.stats.evictions)
        finally:
            manager.close()
    (big_page_ins, big_evictions) = results[NUM_EXPERTS]
    (small_page_ins, small_evictions) = results[TOP_K]
    assert big_evictions == 0
    assert small_evictions > 0
    assert small_page_ins > big_page_ins


def test_repeated_steps_stay_identical_as_the_cache_churns(checkpoint):
    """Output must not become a function of cache history."""
    (path, weights) = checkpoint
    reference = _reference(weights)
    model = _reference(weights)
    manager = install_expert_streaming(
        model, path, ceiling_bytes=_ceiling_for(path, TOP_K), top_k=TOP_K
    )
    try:
        for step in range(6):
            (x, indices) = _inputs(tokens=4, seed=100 + step)
            assert np.array_equal(
                _run(model, x, indices), _run(reference, x, indices)
            )
        assert manager.stats.evictions > 0
    finally:
        manager.close()


def test_seeding_goes_through_the_fetch_path(checkpoint):
    """A prefix slice of a loaded weight pins the parent buffer and frees
    nothing; the resident set is seeded by fetching instead."""
    (path, weights) = checkpoint
    model = _reference(weights)
    manager = install_expert_streaming(
        model, path, ceiling_bytes=_ceiling_for(path, NUM_EXPERTS), top_k=TOP_K
    )
    try:
        cache = next(iter(manager.caches.values()))
        cache.seed(range(4))
        assert cache.resident == 4
        assert manager.stats.page_ins == 4
    finally:
        manager.close()


# ---------------------------------------------------------------------------
# refusals and sizing
# ---------------------------------------------------------------------------


def test_ceiling_below_one_step_refuses_to_start(checkpoint):
    (path, _) = checkpoint
    model = _build_model()
    with pytest.raises(WorkingSetTooSmall) as error:
        install_expert_streaming(model, path, ceiling_bytes=1024, top_k=TOP_K)
    assert "floor" in str(error.value)


def test_a_step_wider_than_the_ceiling_overflows_then_trims(checkpoint):
    """A wide batch is served, not failed, and the cache trims back after.

    Every index must resolve before the gather runs, so the alternatives are
    a transient overshoot or a failed request.  The overshoot is bounded by
    one call's working set and is counted.
    """
    (path, weights) = checkpoint
    model = _reference(weights)
    manager = install_expert_streaming(
        model, path, ceiling_bytes=_ceiling_for(path, TOP_K), top_k=TOP_K
    )
    try:
        cache = next(iter(manager.caches.values()))
        wide = list(range(cache.capacity + 2))
        resolved = cache.acquire(wide)
        assert sorted(resolved) == wide  # every expert really was resolved
        assert cache.resident == cache.capacity  # and trimmed straight back
        assert manager.stats.overflows == 1
    finally:
        manager.close()


def test_fused_gate_up_is_reassembled_from_checkpoint_rows(checkpoint, tmp_path):
    """The live tree may fuse gate+up; the checkpoint ships them separate.

    Qwen3-Next-family adapters concatenate gate_proj and up_proj into
    gate_up_proj at load, and that lever defaults to ON, so this is the
    normal case for our MoE models rather than an exotic one.  A fused
    expert row is the concatenation of the two checkpoint rows, which is
    still pure byte-range addressing.
    """
    (path, weights) = checkpoint
    index = SafetensorsIndex.from_model_path(path)
    prefix = "layers.0"
    spec = weight_stream._resolve_spec(index, f"{prefix}.gate_up_proj", "weight")
    assert spec is not None, "fused projection was not resolved from its parts"
    assert len(spec.parts) == 2
    reader = weight_stream.ExpertSliceReader(
        [spec], handles=weight_stream._FileHandles()
    )
    for expert in (0, NUM_EXPERTS - 1):
        (fused,) = reader.materialize(reader.read_bytes(expert))
        expected = mx.concatenate(
            [
                weights[f"{prefix}.gate_proj.weight"][expert],
                weights[f"{prefix}.up_proj.weight"][expert],
            ],
            axis=-2,
        )
        mx.eval(expected)
        assert np.array_equal(np.array(fused), np.array(expected))


def test_shared_expert_folding_is_refused_not_mis_addressed(checkpoint, monkeypatch):
    """A transform that changes the expert count must refuse, loudly.

    Folding the shared expert in as routed index E makes the live table one
    row wider than the checkpoint's.  Byte ranges would still resolve, and
    would silently address the wrong expert -- exactly the failure the
    correctness gate exists to prevent.
    """
    (path, weights) = checkpoint
    model = _reference(weights)
    folded = next(
        module
        for (_, module) in model.named_modules()
        if hasattr(module, "num_experts") and hasattr(module, "bits")
    )
    monkeypatch.setattr(
        type(folded), "num_experts", property(lambda self: NUM_EXPERTS + 1)
    )
    with pytest.raises(StreamingUnavailable, match="experts"):
        install_expert_streaming(
            model, path, ceiling_bytes=_ceiling_for(path, TOP_K), top_k=TOP_K
        )


def test_model_without_addressable_experts_is_refused(tmp_path):
    model = _build_model()
    path = tmp_path / "renamed"
    path.mkdir()
    renamed = {
        key.replace("gate_proj", "fused_gate_up"): value
        for (key, value) in _flatten(model).items()
    }
    mx.save_safetensors(str(path / "model.safetensors"), renamed)
    with pytest.raises(StreamingUnavailable):
        install_expert_streaming(model, path, ceiling_bytes=1 << 30, top_k=TOP_K)


def test_resident_fraction_follows_the_measured_curve():
    # The table fits: hold all of it. There is nothing to stream below 1.0.
    assert resident_fraction(50, 100) == pytest.approx(1.0)
    assert resident_fraction(100, 100) == pytest.approx(1.0)
    assert resident_fraction(105, 100) == pytest.approx(0.45)
    assert resident_fraction(180, 100) == pytest.approx(0.30)
    assert resident_fraction(260, 100) == pytest.approx(0.20)
    # Between measured points: interpolate. Beyond them: clamp, never
    # extrapolate a collapse curve into a made-up constant.
    assert 0.30 < resident_fraction(140, 100) < 0.45
    assert resident_fraction(10_000, 100) == pytest.approx(0.20)
    assert resident_fraction(10, 0) == 0.0


def test_capacity_is_bounded_by_both_curve_and_ceiling():
    capacity = plan_cache_experts(
        table_bytes=1 << 30,
        expert_bytes=1 << 20,
        num_layers=8,
        top_k=4,
        ceiling_bytes=1 << 29,
    )
    assert capacity >= 4
    assert capacity * 8 * (1 << 20) <= (1 << 29)


# ---------------------------------------------------------------------------
# atlas: collection, validation, counterfactual
# ---------------------------------------------------------------------------


def test_atlas_collects_but_never_changes_residency(checkpoint, tmp_path):
    (path, weights) = checkpoint
    (x, indices) = _inputs()
    plain_model = _reference(weights)
    plain = install_expert_streaming(
        plain_model, path, ceiling_bytes=_ceiling_for(path, TOP_K), top_k=TOP_K
    )
    collector = expert_atlas.AtlasCollector(
        path, sink=tmp_path / "weight_atlas.json", trace_path=tmp_path / "trace.bin"
    )
    traced_model = _reference(weights)
    traced = install_expert_streaming(
        traced_model,
        path,
        ceiling_bytes=_ceiling_for(path, TOP_K),
        top_k=TOP_K,
        collector=collector,
    )
    try:
        for step in range(5):
            (x, indices) = _inputs(tokens=4, seed=200 + step)
            plain_out = _run(plain_model, x, indices)
            traced_out = _run(traced_model, x, indices)
            assert np.array_equal(plain_out, traced_out)
        # Collection changed nothing about what was resident or fetched.
        assert traced.stats.page_ins == plain.stats.page_ins
        assert traced.stats.evictions == plain.stats.evictions
        assert collector.observations > 0
        collector.close()
    finally:
        plain.close()
        traced.close()

    atlas = expert_atlas.load_atlas(
        tmp_path / "weight_atlas.json", expect_digest=expert_atlas.index_digest(path)
    )
    assert atlas is not None
    assert atlas.manifest["pinning"] is False
    assert int(atlas.counts.sum()) == collector.observations


def test_atlas_is_ignored_when_stale_torn_or_mismatched(tmp_path):
    counts = np.arange(12, dtype=np.uint64).reshape(3, 4)
    sink = tmp_path / "weight_atlas.json"
    manifest = expert_atlas.build_manifest(
        digest="a" * 64,
        num_layers=3,
        num_units=4,
        total_observations=99,
        generations=[],
    )
    expert_atlas.write_atlas(sink, manifest, counts)

    assert expert_atlas.load_atlas(sink) is not None
    assert expert_atlas.load_atlas(sink, expect_digest="b" * 64) is None
    assert expert_atlas.load_atlas(sink, geometry=(4, 4)) is None

    # Truncated / torn write.
    data = sink.with_suffix(".bin")
    blob = data.read_bytes()
    data.write_bytes(blob[:-9])
    assert expert_atlas.load_atlas(sink) is None
    # Corrupted counters, correct length: the CRC catches it.
    data.write_bytes(blob[: expert_atlas.DATA_OFFSET] + b"\xff" * (len(blob) - expert_atlas.DATA_OFFSET))
    assert expert_atlas.load_atlas(sink) is None
    # Wrong format.
    data.write_bytes(blob)
    bad = json.loads(sink.read_text())
    bad["format"] = "something-else"
    sink.write_text(json.dumps(bad))
    assert expert_atlas.load_atlas(sink) is None


def test_atlas_merge_decays_old_generations():
    prior = np.array([[1000, 0]], dtype=np.uint64)
    observed = np.array([[0, 5]], dtype=np.uint64)
    merged = expert_atlas.merge_counts(
        prior, observed, observations=expert_atlas.DEFAULT_HALF_LIFE
    )
    assert int(merged[0][0]) == 500
    assert int(merged[0][1]) == 5


def test_atlas_checkpoints_merge_each_observation_once(tmp_path):
    sink = tmp_path / "weight_atlas.json"
    collector = expert_atlas.AtlasCollector(None, sink=sink, checkpoint_every=10)
    collector.bind(num_layers=1, num_units=4)
    for _ in range(10):
        collector.observe(0, [0])
    first = expert_atlas.load_atlas(sink)
    assert first.counts.tolist() == [[10, 0, 0, 0]]
    for _ in range(10):
        collector.observe(0, [1])
    second = expert_atlas.load_atlas(sink)
    # Expert 0 was persisted by the first checkpoint and may only decay; it
    # must not be merged a second time (the pre-fix value was 19).
    assert second.total_observations == 20
    assert int(second.counts[0][0]) <= 10
    assert int(second.counts[0][1]) == 10
    assert int(second.counts.sum()) <= second.total_observations
    # Closing with no new observations must leave the counts as they were.
    collector.close()
    closed = expert_atlas.load_atlas(sink)
    assert closed.total_observations == 20
    assert closed.counts.tolist() == second.counts.tolist()


def test_counterfactual_reports_what_pinning_would_have_done(tmp_path):
    trace = tmp_path / "trace.bin"
    layers = 2
    units = 8
    rng = np.random.default_rng(3)
    # Skewed routing: experts 0 and 1 dominate, so pinning should look good.
    hot = rng.choice([0, 1], size=600)
    cold = rng.integers(0, units, size=200)
    sequence = np.concatenate([hot, cold])
    rng.shuffle(sequence)
    records = np.empty((sequence.size * layers, 2), dtype=np.uint32)
    for layer in range(layers):
        block = records[layer * sequence.size : (layer + 1) * sequence.size]
        block[:, 0] = layer
        block[:, 1] = sequence
    trace.write_bytes(
        expert_atlas.TRACE_MAGIC + struct.pack("<II", layers, units) + records.tobytes()
    )

    counts = np.zeros((layers, units), dtype=np.uint64)
    for layer in range(layers):
        for unit in range(units):
            counts[layer][unit] = int((sequence == unit).sum())

    report = expert_atlas.replay_counterfactual(
        trace, capacity=3, atlas=counts, pin_fractions=(0.0, 0.33), min_samples=1
    )
    rows = {row["pin_fraction"]: row for row in report["rows"]}
    assert rows[0.0]["page_ins_vs_lru"] == 0
    assert rows[0.33]["pinned_per_layer"] == 0  # 3 * 0.33 floors to 0
    assert report["actual_policy"] == "lru"

    wider = expert_atlas.replay_counterfactual(
        trace, capacity=6, atlas=counts, pin_fractions=(0.0, 0.5), min_samples=1
    )
    wider_rows = {row["pin_fraction"]: row for row in wider["rows"]}
    assert wider_rows[0.5]["pinned_per_layer"] == 3
    # The point of the analysis is that it can say either way; assert only
    # that it produced a comparable, signed number rather than a hoped-for one.
    assert isinstance(wider_rows[0.5]["page_ins_vs_lru"], int)
    assert "pinning" in wider["verdict"]


def test_counterfactual_ignores_units_below_the_trust_floor(tmp_path):
    trace = tmp_path / "trace.bin"
    records = np.array([[0, 1], [0, 2], [0, 1]], dtype=np.uint32)
    trace.write_bytes(
        expert_atlas.TRACE_MAGIC + struct.pack("<II", 1, 4) + records.tobytes()
    )
    counts = np.array([[0, 3, 1, 0]], dtype=np.uint64)
    report = expert_atlas.replay_counterfactual(
        trace, capacity=4, atlas=counts, pin_fractions=(0.5,), min_samples=1_000_000
    )
    # Every unit is below min_samples; the fallback pins only units that were
    # actually observed, and never invents one with a zero count.
    assert report["rows"][0]["layers"][0]["pinned_units"] <= 2


# ---------------------------------------------------------------------------
# serving policy plumbing
# ---------------------------------------------------------------------------


def test_policy_defaults_and_validation():
    from mlx2.serving import moe_expert_streaming_policy

    assert moe_expert_streaming_policy(None) == {
        "enabled": False,
        "cache_gib": 0.0,
        "read_workers": 16,
        "atlas": False,
        "atlas_path": None,
        "trace_path": None,
    }
    for bad in (
        [],
        {"nope": 1},
        {"enabled": 1},
        {"enabled": True},  # cache_gib must be positive when enabled
        {"cache_gib": -1.0},
        {"cache_gib": float("inf")},
        {"read_workers": 0},
        {"read_workers": True},
        {"atlas_path": 5},
        {"trace_path": "t.bin"},  # needs atlas
    ):
        with pytest.raises(ValueError):
            moe_expert_streaming_policy(bad)
    assert moe_expert_streaming_policy({"enabled": True, "cache_gib": 4})[
        "cache_gib"
    ] == 4.0


def test_engine_rejects_multi_lane_streaming():
    from mlx2.serving import ServingEngine

    with pytest.raises(ValueError, match="max_lanes=1"):
        ServingEngine(
            "unused",
            max_lanes=2,
            execution_policy={
                "moe_expert_streaming": {"enabled": True, "cache_gib": 1.0}
            },
        )


def test_admission_subtracts_the_ceiling_once():
    from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController

    plain = SelfMTPLaneAdmissionController(host_memory_gib=36.0)
    streamed = SelfMTPLaneAdmissionController(
        host_memory_gib=36.0, stream_reserve_gib=8.0
    )
    assert streamed.hard_reserve_gib == pytest.approx(plain.hard_reserve_gib + 8.0)
    with pytest.raises(ValueError):
        SelfMTPLaneAdmissionController(host_memory_gib=36.0, stream_reserve_gib=-1.0)

    free = 24.0
    plain_plan = plain.decide([4096], free, max_draft=2, cache_gib=[0.5])
    streamed_plan = streamed.decide([4096], free, max_draft=2, cache_gib=[0.5])
    assert streamed_plan.usable_gib == pytest.approx(plain_plan.usable_gib - 8.0)


def test_qualification_requires_observed_streaming():
    from mlx2.qualification import required_feature_checks

    base = {"speculation": "self_mtp", "execution_policy": {}, "environment": {}}
    assert "feature_moe_expert_streaming" not in required_feature_checks(base)
    assert "feature_moe_expert_streaming" in required_feature_checks(
        {**base, "moe_expert_streaming": {"enabled": True, "cache_gib": 4.0}}
    )


def test_streaming_counters_are_exported():
    from mlx2.prometheus import _ENGINE_EVENTS, _STREAM_GAUGES

    for key in (
        "stream_page_ins_total",
        "stream_expert_hits_total",
        "stream_expert_misses_total",
        "stream_evictions_total",
        "atlas_observations_total",
    ):
        assert key in _ENGINE_EVENTS
    assert "stream_resident_bytes" in _STREAM_GAUGES


def test_streaming_counters_reach_the_metrics_scrape():
    import threading
    from collections import Counter

    from mlx2.batch_metrics import BatchRuntimeMetrics
    from mlx2.serving import ServingEngine

    class Stream:
        def counters(self):
            return {
                "stream_page_ins_total": 37,
                "stream_expert_hits_total": 900,
                "stream_expert_misses_total": 37,
                "stream_evictions_total": 5,
                "stream_resident_bytes": 8 << 30,
                "atlas_observations_total": 937,
            }

    engine = ServingEngine.__new__(ServingEngine)
    engine.lock = threading.Lock()
    engine.counts = Counter()
    engine.queued_jobs = 0
    engine.snapshot = {"state": "ready", "model": "m", "qualification": "qualified"}
    engine.batch_metrics = BatchRuntimeMetrics()
    engine.expert_stream = Stream()
    text = engine.prometheus_metrics()
    for event, value in (
        ("page_in", 37),
        ("hit", 900),
        ("miss", 37),
        ("eviction", 5),
    ):
        assert (
            f'mlx2_runtime_events_total{{component="expert_stream",event="{event}"}} {value}'
            in text
        )
    assert (
        'mlx2_runtime_events_total{component="expert_atlas",event="observation"} 937'
        in text
    )
    assert f"mlx2_expert_stream_resident_bytes {8 << 30}" in text
