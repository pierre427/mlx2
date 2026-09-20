"""Concurrent multi-LoRA serving (CPU, tiny qwen3.5 hybrid model)."""

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mlx import nn

from mlx2.runtime.lora import install_lora
from mlx2.runtime.multi_lora import (
    MultiLoRAManager,
    SlotUnavailable,
    bind_lora_rows,
    clear_lora_rows,
    lora_apc_scope,
    multi_lora_manager,
    write_adapter,
)

KEYS_A = (
    "model.layers.0.linear_attn.in_proj_qkv",
    "model.layers.1.linear_attn.in_proj_qkv",
    "model.layers.2.linear_attn.in_proj_qkv",
    "model.layers.3.self_attn.q_proj",
    "model.layers.3.self_attn.v_proj",
)
KEYS_B = (
    "model.layers.0.mlp.down_proj",
    "model.layers.1.mlp.down_proj",
    "model.layers.2.mlp.down_proj",
    "model.layers.3.mlp.down_proj",
    "model.layers.0.linear_attn.in_proj_qkv",
)


def tiny_model():
    from test_approximate_kv_serving import tiny_model as build

    return build()


def dims_for(model, keys):
    modules = dict(model.named_modules())
    return {key: (modules[key].weight.shape[1], modules[key].weight.shape[0]) for key in keys}


@pytest.fixture
def adapters(tmp_path):
    model = tiny_model()
    root = tmp_path / "loras"
    a = write_adapter(root / "sql", keys=KEYS_A, dims=dims_for(model, KEYS_A), rank=4, scale=2.0, seed=1)
    b = write_adapter(root / "chat", keys=KEYS_B, dims=dims_for(model, KEYS_B), rank=8, scale=1.5, seed=2)
    c = write_adapter(root / "code", keys=KEYS_A[:2], dims=dims_for(model, KEYS_A[:2]), rank=2, scale=3.0, seed=3)
    return root, {"sql": a, "chat": b, "code": c}


def reference_logits(tokens, path):
    model = tiny_model()
    if path is not None:
        install_lora(model, name="ref", path=path)
    return model(mx.array([tokens]))[0]


def mixed_logits(manager, model, rows, uids):
    for uid, name in zip(uids, rows):
        if name is None:
            continue
        slot, _ = manager.acquire(name)
        manager.bind_uid(uid, slot)
    bound = bind_lora_rows(model, uids)
    try:
        return model(mx.array(PROMPTS[: len(rows)]))
    finally:
        clear_lora_rows(bound)


PROMPTS = [
    [5, 9, 17, 33, 2, 8],
    [11, 3, 7, 40, 41, 9],
    [100, 12, 6, 6, 70, 1],
    [1, 2, 3, 4, 5, 6],
]


def test_mixed_batch_matches_per_request_single_adapter(adapters):
    root, paths = adapters
    model = tiny_model()
    base_model = tiny_model()
    manager = MultiLoRAManager(model, max_loras=2, max_lora_rank=8)
    manager.register("sql", paths["sql"])
    manager.register("chat", paths["chat"])
    rows = [None, "sql", "chat", "sql"]
    logits = mixed_logits(manager, model, rows, uids=[10, 11, 12, 13])
    base_batch = base_model(mx.array(PROMPTS[: len(rows)]))
    mx.eval(logits, base_batch)
    for index, name in enumerate(rows):
        expected = reference_logits(PROMPTS[index], paths[name] if name else None)
        if name is None:
            # Base rows are bit-exact with the unwrapped model at equal width.
            assert mx.array_equal(logits[index], base_batch[index])
            assert mx.allclose(logits[index], expected, atol=1e-5)
        else:
            assert mx.allclose(logits[index], expected, atol=1e-5, rtol=1e-5), name
            # The adapter changes the output (the mechanism is not a no-op).
            assert not mx.allclose(
                logits[index], base_model(mx.array([PROMPTS[index]]))[0], atol=1e-3
            )
    status = manager.status()
    assert status["counts"]["mixed_forwards"] == 1
    assert status["counts"]["delta_applications"] > 0
    assert status["rows_by_adapter"] == {"sql": 2, "chat": 1}
    assert status["counts"]["rows_base"] == 1


def test_all_base_batch_skips_delta_and_matches_unwrapped(adapters):
    _, paths = adapters
    model = tiny_model()
    manager = MultiLoRAManager(model, max_loras=1, max_lora_rank=8)
    manager.register("sql", paths["sql"])
    slot, _ = manager.acquire("sql")  # resident but no row uses it
    bound = bind_lora_rows(model, [1, 2])
    out = model(mx.array(PROMPTS[:2]))
    clear_lora_rows(bound)
    assert mx.array_equal(out, tiny_model()(mx.array(PROMPTS[:2])))
    counts = manager.status()["counts"]
    assert counts["base_only_forwards"] == 1
    assert counts["delta_applications"] == 0 and counts["delta_skips"] > 0
    manager.release(slot)


def test_lru_eviction_pins_and_deferral(adapters):
    _, paths = adapters
    model = tiny_model()
    manager = MultiLoRAManager(model, max_loras=2, max_lora_rank=8)
    for name in ("sql", "chat", "code"):
        manager.register(name, paths[name])
    s_sql, r1 = manager.acquire("sql")
    s_chat, r2 = manager.acquire("chat")
    assert (r1, r2) == ("loaded", "loaded")
    with pytest.raises(SlotUnavailable):
        manager.acquire("code")
    assert manager.acquire("sql") == (s_sql, "hit")
    manager.release(s_sql)
    manager.release(s_sql)
    s_code, r3 = manager.acquire("code")
    assert (s_code, r3) == (s_sql, "evicted")
    counts = manager.status()["counts"]
    assert counts["slot_deferred"] == 1 and counts["slot_evictions"] == 1
    assert counts["slot_hits"] == 1 and counts["slot_loads"] == 3
    # The evicted slot's content really is the new adapter: code-only row
    # equals the single-adapter reference.
    manager.bind_uid(7, s_code)
    bound = bind_lora_rows(model, [7])
    out = model(mx.array([PROMPTS[0]]))[0]
    clear_lora_rows(bound)
    assert mx.allclose(out, reference_logits(PROMPTS[0], paths["code"]), atol=1e-5)
    with pytest.raises(ValueError, match="in use"):
        manager.unregister("code")
    manager.release(s_code)
    manager.unregister("code")
    assert "code" not in manager.status()["registered"]


def test_row_binding_mismatch_fails_closed(adapters):
    _, paths = adapters
    model = tiny_model()
    manager = MultiLoRAManager(model, max_loras=1, max_lora_rank=8)
    manager.register("sql", paths["sql"])
    slot, _ = manager.acquire("sql")
    manager.bind_uid(1, slot)
    bound = bind_lora_rows(model, [1])
    with pytest.raises(RuntimeError, match="row binding"):
        model(mx.array(PROMPTS[:2]))
    clear_lora_rows(bound)


def test_registration_validation(adapters, tmp_path):
    _, paths = adapters
    model = tiny_model()
    manager = MultiLoRAManager(model, max_loras=1, max_lora_rank=4)
    with pytest.raises(ValueError, match="max_lora_rank"):
        manager.register("chat", paths["chat"])  # rank 8 > 4
    manager.register("sql", paths["sql"])
    with pytest.raises(ValueError, match="already loaded"):
        manager.register("sql", paths["sql"])
    with pytest.raises(ValueError, match="already carries"):
        MultiLoRAManager(model, max_loras=1, max_lora_rank=4)
    for bad in (0, 65, True, "2"):
        with pytest.raises(ValueError):
            MultiLoRAManager(tiny_model(), max_loras=bad, max_lora_rank=4)
    manager.detach()
    assert multi_lora_manager(model) is None
    assert not any(
        type(module).__name__ == "MultiLoRALinear" for _, module in model.named_modules()
    )
    assert mx.array_equal(model(mx.array([PROMPTS[0]])), tiny_model()(mx.array([PROMPTS[0]])))


def test_same_content_same_fingerprint_different_content_differs(adapters, tmp_path):
    _, paths = adapters
    model = tiny_model()
    manager = MultiLoRAManager(model, max_loras=2, max_lora_rank=8)
    first = manager.register("sql", paths["sql"])
    copy = tmp_path / "copy"
    copy.mkdir()
    for name in ("adapter_config.json", "adapters.safetensors"):
        (copy / name).write_bytes((paths["sql"] / name).read_bytes())
    second = manager.register("sql-alias", copy)
    third = manager.register("code", paths["code"])
    assert first["fingerprint"] == second["fingerprint"] != third["fingerprint"]
    assert lora_apc_scope(None, None) is None
    assert lora_apc_scope("m1", None) == "m1"
    assert lora_apc_scope(None, "f") == "|lora:f"
    assert lora_apc_scope("m1", "f") == "m1|lora:f"


def test_quantized_base_is_supported(adapters):
    _, paths = adapters
    model = tiny_model()
    nn.quantize(model, group_size=32, bits=4, class_predicate=lambda p, m: isinstance(m, nn.Linear) and m.weight.shape[1] % 32 == 0 and p.endswith("down_proj"))
    manager = MultiLoRAManager(model, max_loras=1, max_lora_rank=8)
    manager.register("chat", paths["chat"])
    slot, _ = manager.acquire("chat")
    manager.bind_uid(1, slot)
    bound = bind_lora_rows(model, [1, 2])
    out = model(mx.array(PROMPTS[:2]))
    clear_lora_rows(bound)
    mx.eval(out)
    assert manager.status()["counts"]["delta_applications"] > 0


# --------------------------------------------------------------------------
# Engine integration (ordinary route, qualification mode, tiny hybrid model)
# --------------------------------------------------------------------------

from mlx2 import memory, serving  # noqa: E402
from mlx2.runtime import os_memory  # noqa: E402
from mlx2.serving import ServingEngine  # noqa: E402

LONG = list(range(1, 40))


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "src"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)


def make_engine(root=None, **kwargs):
    from test_approximate_kv_serving import make_adapter

    engine = ServingEngine(
        "tiny",
        adapter_factory=make_adapter(tiny_model(), operations=None),
        qualification_mode=True,
        mtp=False,
        lora_root=root,
        coalesce_window_ms=50.0,
        **kwargs,
    )
    assert engine.ready.wait(60), engine.error
    return engine


def collect(job):
    text = ""
    while True:
        event = job.events.get(timeout=60)
        if "error" in event:
            raise AssertionError(event)
        if "delta" in event:
            text += event["delta"].get("content", "")
        if "finish_reason" in event:
            return [int(t) for t in text.split()], event["receipt"]


def request(tokens, model=None, max_tokens=6):
    body = {"tokens": list(tokens), "max_tokens": max_tokens, "temperature": 0}
    if model is not None:
        body["model"] = model
    return body


def test_engine_mixed_batch_equals_sequential_and_single_adapter_route(host, adapters):
    root, _ = adapters
    engine = make_engine(root, max_loras=2, max_lora_rank=8, max_lanes=4)
    try:
        for name in ("sql", "chat"):
            receipt = engine.load_lora_adapter(name, name)
            assert receipt["mode"] == "concurrent"
        rows = [(PROMPTS[0], None), (PROMPTS[1], "sql"), (PROMPTS[2], "chat"), (PROMPTS[3], "sql")]
        jobs = [engine.submit(request(t, m)) for t, m in rows]
        mixed = [collect(job) for job in jobs]
        counts = engine.status()["multi_lora"]["counts"]
        # Mechanism assertion: the rows really shared mixed forwards.
        assert counts["mixed_forwards"] > 0, counts
        assert counts["delta_applications"] > 0
        sequential = [collect(engine.submit(request(t, m))) for t, m in rows]
        for (tokens_m, receipt_m), (tokens_s, _), (_, name) in zip(mixed, sequential, rows):
            assert tokens_m == tokens_s, name
            lora = receipt_m["lora"]
            assert lora["schema"] == "mlx2.multi-lora.v1"
            assert lora["name"] == name
            assert lora["apc_namespace"] == ("adapter" if name else "base")
        by_adapter = engine.status()["multi_lora"]["rows_by_adapter"]
        assert by_adapter["sql"] > 0 and by_adapter["chat"] > 0
        assert engine.status()["counts"]["multi_lora_requests"] == 6
        assert engine.status()["counts"]["multi_lora_base_requests"] == 2
        metrics = engine.prometheus_metrics()
        assert 'mlx2_multi_lora_events_total{event="mixed_forwards"}' in metrics
        assert "mlx2_multi_lora_resident_adapters 2" in metrics
    finally:
        engine.close()
    # Reference route: the single-adapter drain-to-swap path.
    reference = make_engine(root, max_lanes=1)
    try:
        for index, (tokens, name) in enumerate(rows):
            if name is None:
                continue
            reference.load_lora_adapter(name, name)
            expected, receipt = collect(reference.submit(request(tokens)))
            reference.unload_lora_adapter(name)
            assert mixed[index][0] == expected, name
            assert "lora" not in receipt
    finally:
        reference.close()


def test_engine_apc_namespace_isolates_adapters(host, adapters):
    root, _ = adapters
    engine = make_engine(root, max_loras=2, max_lora_rank=8, max_lanes=1)
    try:
        engine.load_lora_adapter("sql", "sql")
        engine.load_lora_adapter("chat", "chat")
        cached = {}
        for label, model in (("sql", "sql"), ("chat", "chat"), ("base", None), ("sql2", "sql")):
            _, receipt = collect(engine.submit(request(LONG, model, max_tokens=3)))
            cached[label] = receipt["cached_tokens"]
        assert cached["sql"] == 0
        assert cached["chat"] == 0, "cross-adapter prefix reuse"
        assert cached["base"] == 0, "adapter prefix leaked into the base namespace"
        assert cached["sql2"] > 0, "same-adapter prefix reuse"
    finally:
        engine.close()


def test_engine_slot_pressure_defers_then_evicts(host, adapters):
    root, _ = adapters
    engine = make_engine(root, max_loras=1, max_lora_rank=8, max_lanes=2)
    try:
        engine.load_lora_adapter("sql", "sql")
        engine.load_lora_adapter("chat", "chat")
        jobs = [
            engine.submit(request(PROMPTS[1], "sql", max_tokens=12)),
            engine.submit(request(PROMPTS[2], "chat")),
        ]
        results = [collect(job) for job in jobs]
        status = engine.status()
        assert status["counts"]["multi_lora_slot_deferred"] >= 1
        assert status["multi_lora"]["counts"]["slot_evictions"] >= 1
        assert results[1][1]["lora"]["residency"] == "evicted"
        assert status["multi_lora"]["pinned_slots"] == 0
        # The deferred adapter row still equals its sequential output.
        again, _ = collect(engine.submit(request(PROMPTS[2], "chat")))
        assert again == results[1][0]
    finally:
        engine.close()


def test_engine_register_unregister_lifecycle(host, adapters):
    root, _ = adapters
    engine = make_engine(root, max_loras=1, max_lora_rank=4)
    try:
        first = engine.load_lora_adapter("sql", "sql")
        assert first["drained"] is True  # new module keys: structural wrap
        second = engine.load_lora_adapter("code", "code")
        assert second["drained"] is False  # keys already wrapped: no drain
        assert first["fingerprint"] != second["fingerprint"]
        with pytest.raises(ValueError, match="already loaded"):
            engine.load_lora_adapter("sql", "sql")
        with pytest.raises(ValueError, match="max_lora_rank"):
            engine.load_lora_adapter("chat", "chat")  # rank 8 > 4
        with pytest.raises(ValueError, match="base model"):
            engine.load_lora_adapter("tiny", "sql")
        with pytest.raises(ValueError, match="within the configured"):
            engine.load_lora_adapter("escape", "../outside")
        assert engine.unload_lora_adapter("code")["mode"] == "concurrent"
        with pytest.raises(ValueError, match="not loaded"):
            engine.unload_lora_adapter("code")
        assert engine.status()["multi_lora"]["registered"] == ["sql"]
        # An unregistered name is the base model (historical behavior).
        _, receipt = collect(engine.submit(request(PROMPTS[0], "code")))
        assert receipt["lora"]["name"] is None and receipt["lora"]["slot"] == 0
    finally:
        engine.close()


def test_multi_lora_is_ordinary_route_only_and_needs_lora_dir(tmp_path):
    base = dict(adapter_factory=object, qualification_mode=True)
    with pytest.raises(ValueError, match="requires --lora-dir"):
        ServingEngine.validate_arguments("tiny", max_loras=2, mtp=False, **base)
    for extra, label in (
        ({"mtp": True}, "MTP"),
        ({"mtp": False, "prompt_lookup": True}, "prompt lookup"),
        ({"mtp": False, "int8_prefill": {"enabled": True, "scope": "all"}}, "int8"),
    ):
        with pytest.raises(ValueError, match="ordinary route|int8|prompt lookup|MTP"):
            ServingEngine.validate_arguments(
                "tiny", max_loras=2, lora_root=str(tmp_path), **extra, **base
            )
    for bad in (-1, 65, True):
        with pytest.raises(ValueError, match="max_loras"):
            ServingEngine.validate_arguments(
                "tiny", max_loras=bad, lora_root=str(tmp_path), mtp=False, **base
            )
    ServingEngine.validate_arguments(
        "tiny", max_loras=2, lora_root=str(tmp_path), mtp=False, **base
    )


def test_default_off_is_unchanged(host, adapters):
    root, _ = adapters
    engine = make_engine(root)
    try:
        assert "multi_lora" not in engine.status()["settings"]
        assert engine.status()["multi_lora"] == {"enabled": False}
        _, receipt = collect(engine.submit(request(PROMPTS[0], "sql")))
        assert "lora" not in receipt
        assert engine.status()["counts"].get("multi_lora_requests", 0) == 0
    finally:
        engine.close()


def test_e2e_harness_collect_includes_reasoning_channel():
    """Parity must compare reasoning text too, not two empty content strings."""
    import importlib.util
    import queue
    from pathlib import Path
    from types import SimpleNamespace

    path = Path(__file__).resolve().parents[1] / "scripts" / "gpu_multi_lora_throughput.py"
    spec = importlib.util.spec_from_file_location("gpu_multi_lora_throughput", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    events = queue.Queue()
    for event in (
        {"delta": {"reasoning_content": "think "}},
        {"delta": {"content": "answer"}},
        {"finish_reason": "length", "receipt": {"completion_tokens": 2}},
    ):
        events.put(event)
    pieces, final = module.collect(SimpleNamespace(events=events))
    assert "".join(pieces) == "think answer"
    assert final["finish_reason"] == "length"
