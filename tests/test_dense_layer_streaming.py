"""Synthetic CPU implementation evidence; no real checkpoint qualification."""

import numpy as np
import pytest
from mlx.utils import tree_flatten
from test_disk_weight_streaming import (
    PROMPT,
    _dense_artifact,
    _dense_checkpoint,
    _dense_model,
    _EvalSpy,
    _logits,
    _quantizer,
    _resident,
    _shards,
    _stub_tokenizer,
    mx,
    nn,
)

from mlx2.runtime.layer_stream import trunk_layer_targets
from mlx2.runtime.streamed_load import WeightStreamRequest, load_streamed
from mlx2.runtime.weight_stream import StreamingUnavailable, WorkingSetTooSmall


@pytest.fixture(autouse=True)
def exact_synthetic_evidence(tmp_path, record_property, request):
    import hashlib
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    record_property(
        "source_revision",
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
    )
    source = hashlib.sha256()
    paths = [
        *root.glob("src/mlx2/runtime/*stream*.py"),
        root / "src/mlx2/serving.py",
        root / "src/mlx2/prometheus.py",
        root / "src/mlx2/adapters/qwen38_27b.py",
        root / "src/mlx2/adapters/standard_decoder.py",
        Path(__file__),
    ]
    for path in sorted(paths):
        source.update(str(path.relative_to(root)).encode())
        source.update(path.read_bytes())
    record_property("working_source_sha256", source.hexdigest())
    record_property("evidence_kind", "synthetic_cpu_functional_not_qualification")
    params = getattr(getattr(request.node, "callspec", None), "params", {})
    model = params.get("family", "qwen3_5")
    if request.node.name.startswith("test_standard_moe_adapter"):
        model = "qwen3_moe"
    if request.node.name.startswith("test_structural_decoder_discovery"):
        model = "synthetic_generic_decoder"
    metadata_only = request.node.name.startswith("test_automatic_moe_candidate")
    record_property("model", model)
    record_property(
        "execution_kind", "metadata_only" if metadata_only else "functional_cpu"
    )
    yield
    if metadata_only:
        record_property("checkpoint", "not_applicable_metadata_only")
        return
    # These fixtures are deterministic generated artifacts, not released models.
    roots = sorted({p.parent for p in tmp_path.rglob("*.safetensors")})
    if not roots:
        record_property("checkpoint", "not_applicable_policy_only")
    for index, artifact in enumerate(roots):
        digest = hashlib.sha256()
        for path in sorted(artifact.iterdir()):
            if path.suffix in (".safetensors", ".json"):
                digest.update(path.name.encode())
                digest.update(path.read_bytes())
        record_property(
            "checkpoint" + (str(index) if index else ""),
            f"synthetic-dense-layer-fixture@{digest.hexdigest()}",
        )


PREFIX = "language_model.model.layers."


def _load(
    model, root, weight_map, records, budget=1 << 30, targets=None, resident=None
):
    return load_streamed(
        model,
        root,
        _shards(weight_map),
        request=WeightStreamRequest(
            mode="dense_layers", budget_bytes=budget, read_workers=2
        ),
        sanitize=model.sanitize,
        quantize=_quantizer(model),
        records=records,
        weight_map=weight_map,
        layer_resident_tensor=resident,
        layer_targets=targets
        or (lambda m: trunk_layer_targets(m, layers_prefix=PREFIX)),
    )


@pytest.mark.parametrize("bounded", [False, True])
def test_layers_exact_logits_state_and_layer_zero_resident(
    tmp_path, monkeypatch, bounded
):
    root, weight_map, records = _dense_checkpoint(tmp_path)
    reference = _resident(_dense_model(), root, weight_map)
    model = _dense_model()
    spy = _EvalSpy(
        monkeypatch,
        lambda name: (
            ".layers." in name and ".layers.0." not in name and "_proj." in name
        ),
    )
    loaded = _load(model, root, weight_map, records)
    manager = loaded.manager
    try:
        assert spy.watched and not spy.watched_evaluated()
        assert isinstance(model.layers[0].mlp.gate_proj, nn.QuantizedLinear)
        assert manager.plan.resident_paths == (PREFIX + "0",)
        assert not any(key.startswith(PREFIX + "0.") for key in manager.streamed_keys)
        if bounded:
            manager.plan.ceiling_bytes = 2 * manager.plan.max_layer_bytes
        manager.begin_serving()
        state, control = model.make_cache(), reference.make_cache()
        for tokens in (PROMPT, [9], [33]):
            assert np.array_equal(
                _logits(model, tokens, state), _logits(reference, tokens, control)
            )
            for left, right in zip(state, control):
                for (_, a), (_, b) in zip(
                    tree_flatten(left.state), tree_flatten(right.state)
                ):
                    if a is not None:
                        assert np.array_equal(np.array(a), np.array(b))
            assert manager.live_bytes() <= manager.reserved_bytes()
        if bounded:
            assert manager.evictions > 0
            assert manager.stats.page_ins == 9
        else:
            assert manager.hits == 6
            assert manager.stats.page_ins == 3
        assert manager.receipt()["qualification"] == "unqualified"
    finally:
        manager.close()
    assert manager.live_bytes() == 0
    with pytest.raises(StreamingUnavailable):
        _logits(model, [9])


def test_layer_budget_and_layer_zero_refused_before_eval(tmp_path, monkeypatch):
    root, weight_map, records = _dense_checkpoint(tmp_path)
    spy = _EvalSpy(monkeypatch, lambda name: True)
    with pytest.raises(WorkingSetTooSmall, match="twice"):
        _load(_dense_model(), root, weight_map, records, budget=1)
    assert not spy.watched_evaluated()
    with pytest.raises(StreamingUnavailable, match="layer 0"):
        _load(
            _dense_model(), root, weight_map, records, targets=lambda _: [PREFIX + "0"]
        )


def test_adapter_layer_route(tmp_path, monkeypatch):
    from mlx2.adapters import qwen38_27b as module
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter

    root, weight_map, records = _dense_checkpoint(tmp_path)
    artifact = _dense_artifact(root, weight_map, records)
    monkeypatch.setattr(
        Qwen3827BAdapter, "artifact_inspector", staticmethod(lambda _: artifact)
    )
    monkeypatch.setattr(
        Qwen3827BAdapter, "environment_configurator", staticmethod(dict)
    )
    _stub_tokenizer(monkeypatch, module)
    request = WeightStreamRequest(
        mode="dense_layers", budget_bytes=1 << 30, read_workers=2
    )
    adapter = Qwen3827BAdapter(str(root), weight_streaming=request)
    try:
        assert adapter.weight_stream.mode == "dense_layers"
        before = adapter.weight_stream.hits
        _logits(adapter.model, PROMPT)
        assert adapter.weight_stream.hits > before
    finally:
        adapter.close()


def test_layer_policy_and_request_delivery():
    from mlx2.serving import ServingEngine, dense_weight_streaming_policy

    policy = dense_weight_streaming_policy(
        {"enabled": True, "mode": "dense_layers", "cache_gib": 1}
    )
    engine = object.__new__(ServingEngine)
    engine.dense_weight_streaming_policy = policy
    engine._preload_admission_limit_bytes = lambda: 4 << 30
    request = engine._weight_stream_request()
    assert request.mode == "dense_layers" and request.budget_bytes == 1 << 30
    for change in (
        {"cache_gib": True},
        {"cache_gib": float("nan")},
        {"staging_gib": 1},
        {"mode": "unknown"},
    ):
        with pytest.raises(ValueError):
            dense_weight_streaming_policy({**policy, **change})


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
@pytest.mark.parametrize("quantized", [False, True])
def test_standard_dense_adapter_exact_under_eviction(
    tmp_path, monkeypatch, family, quantized
):
    import json

    from mlx2.adapters import standard_decoder as module
    from mlx2.adapters.standard_decoder import StandardDecoderAdapter
    from mlx2.runtime.models.standard_decoder import Model, ModelArgs

    config = {
        "model_type": family,
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 64,
        "max_position_embeddings": 256,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
    }
    mx.random.seed(7)
    reference = Model(ModelArgs.from_dict(config))
    if quantized:
        nn.quantize(reference, group_size=64, bits=4)
        config["quantization"] = {"group_size": 64, "bits": 4}
    mx.eval(reference.parameters())
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(reference.parameters()))
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(module, "resolve_eos_token_ids", lambda *_: [0], raising=False)
    _stub_tokenizer(monkeypatch, module)
    monkeypatch.setattr("mlx2.adapters.eos.artifact_eos_token_ids", lambda *_: [0])
    # Tokenizer-integrity repair is irrelevant to this generated fixture.
    monkeypatch.setattr(
        "mlx2.runtime.tokenizer_integrity.repair_loaded_tokenizer", lambda *_: {}
    )
    adapter = StandardDecoderAdapter(
        str(tmp_path),
        weight_streaming=WeightStreamRequest(
            mode="dense_layers", budget_bytes=1 << 30, read_workers=2
        ),
    )
    try:
        manager = adapter.weight_stream
        manager.plan.ceiling_bytes = 2 * manager.plan.max_layer_bytes
        state, control = adapter.model.make_cache(), reference.make_cache()
        for tokens in (PROMPT, [9], [33]):
            assert np.array_equal(
                _logits(adapter.model, tokens, state),
                _logits(reference, tokens, control),
            )
        assert manager.evictions > 0
        assert manager.stats.peak_tracked_bytes <= manager.reserved_bytes()
    finally:
        adapter.close()


@pytest.mark.parametrize("lookup", [False, True])
def test_serving_cold_warm_apcv2_with_layer_eviction(tmp_path, monkeypatch, lookup):
    from route_harness import collect, make_adapter, patch_host

    from mlx2.serving import ServingEngine

    patch_host(monkeypatch)
    root, weight_map, records = _dense_checkpoint(tmp_path)
    model = _dense_model()
    base = make_adapter(model, 64)

    class Adapter(base):
        def __init__(self, path, *, weight_streaming):
            super().__init__(path)
            loaded = load_streamed(
                model,
                root,
                _shards(weight_map),
                request=weight_streaming,
                sanitize=model.sanitize,
                quantize=_quantizer(model),
                records=records,
                weight_map=weight_map,
                layer_targets=lambda m: trunk_layer_targets(m, layers_prefix=PREFIX),
            )
            self.weight_stream = loaded.manager
            self.weight_stream.plan.ceiling_bytes = (
                2 * self.weight_stream.plan.max_layer_bytes
            )
            self.weight_stream.begin_serving()

        def close(self):
            self.weight_stream.close()

    engine = ServingEngine(
        str(root),
        adapter_factory=Adapter,
        max_lanes=1,
        max_inflight=2,
        qualification_mode=True,
        mtp=False,
        prompt_lookup=lookup,
        prefill_step=8,
        execution_policy={
            "dense_weight_streaming": {
                "enabled": True,
                "mode": "dense_layers",
                "cache_gib": 1,
            }
        },
    )
    try:
        assert engine.ready.wait(30), engine.error
        assert engine.error is None, engine.error
        request = {"tokens": [3, 7, 1, 12] * 4, "max_tokens": 4, "temperature": 0}
        cold = collect(engine.submit(dict(request)), timeout=30)
        assert "error" not in cold, cold
        warm = collect(engine.submit(dict(request)), timeout=30)
        assert "error" not in warm, warm
        assert cold["tokens"] == warm["tokens"]
        assert warm["receipt"]["cached_tokens"] > 0
        assert engine.expert_stream.evictions > 0
    finally:
        engine.close()


def test_changed_source_and_failed_reads_fail_closed(tmp_path, monkeypatch):
    from mlx2.runtime.weight_stream import WeightSourceChanged

    root, weight_map, records = _dense_checkpoint(tmp_path)
    model = _dense_model()
    loaded = _load(model, root, weight_map, records)
    manager = loaded.manager
    try:
        shard = root / _shards(weight_map)[0]
        import os

        st = shard.stat()
        os.utime(shard, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
        with pytest.raises(WeightSourceChanged):
            _logits(model, PROMPT)
        with pytest.raises(StreamingUnavailable, match="failed"):
            _logits(model, PROMPT)
    finally:
        manager.close()


def test_read_failure_retains_owner_and_refuses_reuse(tmp_path, monkeypatch):
    root, weight_map, records = _dense_checkpoint(tmp_path)
    model = _dense_model()
    loaded = _load(model, root, weight_map, records)
    manager = loaded.manager
    first = next(iter(manager.readers.values()))
    monkeypatch.setattr(
        first, "load", lambda: (_ for _ in ()).throw(OSError("injected I/O error"))
    )
    try:
        with pytest.raises(OSError, match="injected"):
            _logits(model, PROMPT)
        with pytest.raises(StreamingUnavailable, match="failed"):
            _logits(model, PROMPT)
        assert manager._handles is not None
    finally:
        manager.close()


def test_qwen_sanitized_small_tensors_stay_resident(tmp_path, monkeypatch):
    from test_disk_weight_streaming import (
        _is_trunk_mlp,
        _official,
        _quantize_all,
        _write_checkpoint,
    )

    from mlx2.adapters import qwen38_27b as module
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter

    source = _dense_model()
    source.set_dtype(mx.bfloat16)
    _quantize_all(source)
    weights = {
        _official(k): v.astype(mx.float32) if "layernorm.weight" in k else v
        for k, v in tree_flatten(source.parameters())
    }
    root = tmp_path / "normalized-qwen"
    weight_map, records = _write_checkpoint(root, weights, _is_trunk_mlp)
    reference = _resident(_dense_model(), root, weight_map)
    artifact = _dense_artifact(root, weight_map, records)
    monkeypatch.setattr(
        Qwen3827BAdapter, "artifact_inspector", staticmethod(lambda _: artifact)
    )
    monkeypatch.setattr(
        Qwen3827BAdapter, "environment_configurator", staticmethod(dict)
    )
    _stub_tokenizer(monkeypatch, module)
    adapter = Qwen3827BAdapter(
        str(root),
        weight_streaming=WeightStreamRequest(
            mode="dense_layers", budget_bytes=1 << 30, read_workers=2
        ),
    )
    try:
        manager = adapter.weight_stream
        manager.plan.ceiling_bytes = 2 * manager.plan.max_layer_bytes
        assert manager.plan.retained_tensor_bytes > 0
        assert any(p.endswith("A_log") for p in manager.plan.retained_tensor_paths)
        actual = adapter.model(mx.array([PROMPT])).astype(mx.float32)
        expected = reference(mx.array([PROMPT])).astype(mx.float32)
        mx.eval(actual, expected)
        assert np.array_equal(np.array(actual), np.array(expected))
    finally:
        adapter.close()


def test_standard_constructor_failure_closes_installed_stream(monkeypatch):
    from types import SimpleNamespace

    from mlx2.adapters.standard_decoder import StandardDecoderAdapter

    closed = []

    def fail(self, *args, **kwargs):
        self.weight_stream = SimpleNamespace(close=lambda: closed.append(True))
        raise RuntimeError("failure after installation")

    monkeypatch.setattr(StandardDecoderAdapter, "_initialize", fail)
    with pytest.raises(RuntimeError, match="after installation"):
        StandardDecoderAdapter("unused")
    assert closed == [True]


def test_dense_scan_preserves_useful_cached_layers(tmp_path):
    root, weight_map, records = _dense_checkpoint(tmp_path)
    model = _dense_model()
    loaded = _load(model, root, weight_map, records)
    manager = loaded.manager
    # Space for one largest cached block beside a complete progress window.
    manager.plan.ceiling_bytes = 3 * manager.plan.max_layer_bytes
    manager.begin_serving()
    try:
        state = model.make_cache()
        _logits(model, PROMPT, state)
        first = manager.stats.page_ins
        _logits(model, [9], state)
        assert manager.hits > 0
        assert manager.stats.page_ins - first < len(manager.blocks)
        assert manager.stats.peak_tracked_bytes <= manager.reserved_bytes()
        assert manager.bypasses > 0
    finally:
        manager.close()


def test_empty_streamable_layer_refuses_before_eval(tmp_path, monkeypatch):
    root, weight_map, records = _dense_checkpoint(tmp_path)
    spy = _EvalSpy(monkeypatch, lambda name: True)
    with pytest.raises(StreamingUnavailable, match="no streamable parameters"):
        _load(_dense_model(), root, weight_map, records, resident=lambda *_: True)
    assert not spy.watched_evaluated()


def test_structural_decoder_discovery_without_adapter_targets(tmp_path):
    from mlx2.runtime.layer_stream import discover_layer_targets

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(8, 8)
            self.coefficient = mx.array([1.0])

        def __call__(self, x):
            return self.projection(x) * self.coefficient

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.stages = [Block() for _ in range(3)]

        def __call__(self, x):
            for stage in self.stages:
                x = stage(x)
            return x

    mx.random.seed(31)
    reference = Model()
    mx.eval(reference.parameters())
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(reference.parameters()))
    )
    model = Model()
    assert discover_layer_targets(model) == ["stages.1", "stages.2"]
    loaded = load_streamed(
        model,
        tmp_path,
        ["model.safetensors"],
        request=WeightStreamRequest(
            mode="dense_layers", budget_bytes=1024, read_workers=2
        ),
        sanitize=lambda weights: weights,
        quantize=lambda _: None,
    )
    try:
        x = mx.ones((1, 8))
        actual, expected = model(x), reference(x)
        mx.eval(actual, expected)
        assert np.array_equal(np.array(actual), np.array(expected))
        assert loaded.manager.plan.resident_paths == ("stages.0",)
        assert loaded.manager.plan.retained_tensor_paths == (
            "stages.1.coefficient",
            "stages.2.coefficient",
        )
    finally:
        loaded.manager.close()
    model = Model()
    model.ambiguous = [Block(), Block()]
    with pytest.raises(StreamingUnavailable, match="unambiguous"):
        discover_layer_targets(model)


def test_automatic_moe_candidate_needs_no_mode_declaration(tmp_path):
    import json

    from mlx2.runtime.streamed_load import (
        accepts_streaming_request,
        auto_expert_streaming_decision,
    )

    class Adapter:
        def __init__(self, path, *, weight_streaming=None):
            pass

    class Child(Adapter):
        pass

    class Eager:
        def __init__(self, path, **kwargs):
            pass

    (tmp_path / "config.json").write_text(
        json.dumps({"text_config": {"num_experts": 8}})
    )
    (tmp_path / "model.safetensors").write_bytes(b"candidate-size-only" * 100)
    for factory in (Adapter, Child):
        assert accepts_streaming_request(factory)
        decision = auto_expert_streaming_decision(
            factory,
            tmp_path,
            admission_limit_bytes=1024,
            explicit=False,
            headroom_bytes=0,
        )
        assert decision["selected"]
    assert not accepts_streaming_request(Eager)
    assert not auto_expert_streaming_decision(
        Eager, tmp_path, admission_limit_bytes=1024, explicit=False, headroom_bytes=0
    )["selected"]


@pytest.mark.parametrize("quantized", [False, True])
def test_standard_moe_adapter_without_declaration(tmp_path, monkeypatch, quantized):
    import json

    from mlx2.adapters.standard_decoder import StandardDecoderAdapter
    from mlx2.runtime.models.standard_decoder import Model, ModelArgs

    config = {
        "model_type": "qwen3_moe",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 64,
        "max_position_embeddings": 256,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 64,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "norm_topk_prob": True,
    }
    mx.random.seed(13)
    reference = Model(ModelArgs.from_dict(config))
    if quantized:
        nn.quantize(reference, group_size=64, bits=4)
        config["quantization"] = {"group_size": 64, "bits": 4}
    mx.eval(reference.parameters())
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(reference.parameters()))
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    _stub_tokenizer(monkeypatch)
    monkeypatch.setattr("mlx2.adapters.eos.artifact_eos_token_ids", lambda *_: [0])
    adapter = StandardDecoderAdapter(
        str(tmp_path),
        weight_streaming=WeightStreamRequest(
            mode="moe_experts", budget_bytes=1 << 20, read_workers=2
        ),
    )
    try:
        manager = adapter.weight_stream
        assert manager.mode == "moe_experts"
        state, control = adapter.model.make_cache(), reference.make_cache()
        for tokens in (PROMPT, [9], [33]):
            assert np.array_equal(
                _logits(adapter.model, tokens, state),
                _logits(reference, tokens, control),
            )
        assert manager.stats.page_ins > 0
    finally:
        adapter.close()


def test_engine_rejects_wrong_installed_stream_mode():
    from types import SimpleNamespace

    from mlx2.serving import ServingEngine

    engine = object.__new__(ServingEngine)
    engine.dense_weight_streaming_policy = {"enabled": True, "mode": "dense_layers"}
    with pytest.raises(RuntimeError, match="different weight streaming mode"):
        engine._bind_adapter_stream(
            SimpleNamespace(weight_stream=SimpleNamespace(mode="moe_experts"))
        )
