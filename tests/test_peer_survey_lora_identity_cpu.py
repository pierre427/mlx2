"""Compiled-model invalidation and reused LoRA slots execute fresh identities.

These are real CPU tensor forwards through the serving invalidation seam and
slot manager, not evidence about Metal graph capture or a model artifact.
"""

import mlx.core as mx
import pytest
from mlx import nn
from qsa_oracle import tiny_args

from mlx2.runtime.lora import install_lora
from mlx2.runtime.models import qwen4_exp as QE
from mlx2.runtime.multi_lora import MultiLoRAManager, lora_apc_scope, write_adapter


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3, bias=False)
        self.linear.weight = mx.arange(12, dtype=mx.float32).reshape(3, 4) / 16

    def __call__(self, x):
        return self.linear(x)


def _artifacts(tmp_path):
    return {name: write_adapter(tmp_path / name, keys=("linear",), dims={"linear": (4, 3)},
                                rank=2, scale=2.0, seed=seed)
            for name, seed in (("a", 21), ("b", 42))}


def _reference(x, path=None):
    model = _Model()
    if path is not None:
        install_lora(model, name="reference", path=path)
    return model(x)


class _PLEModel(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(7)
        self.ple = QE.PLELayer(tiny_args(ple_layer_ids=[1]), 0, 0)


def _ple_artifacts(tmp_path):
    dims = {"ple.key_proj": (16, 64), "ple.value_proj": (16, 16)}
    return {name: write_adapter(tmp_path / name, keys=tuple(dims), dims=dims,
                                rank=2, scale=2.0, seed=seed)
            for name, seed in (("a", 21), ("b", 42))}


def _ple_head(layer, hidden, embeddings):
    signature = (tuple(hidden.shape), tuple(embeddings.shape), "cpu-identity-diagnostic")
    head, _tail = layer._compiled_chain(signature, False, False, False)
    result = head(hidden, embeddings)
    expected = layer._chain_gate(hidden, embeddings)
    mx.eval(result, expected)
    return result, expected


def test_compiled_ple_base_a_b_unload_a_retraces_actual_wrapper_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(QE, "_PLE_COMPILE_MIN_SEEN", 1)
    model = _PLEModel()
    paths = _ple_artifacts(tmp_path)
    hidden = mx.random.normal((1, 2, 64))
    embeddings = mx.random.normal((1, 2, 16))
    session = None
    try:
        for name in (None, "a", "b", None, "a"):
            if session is not None:
                session.restore(model)
                session = None
            if name is not None:
                session = install_lora(model, name=name, path=paths[name])
            actual, expected = _ple_head(model.ple, hidden, embeddings)
            assert all(mx.allclose(a, b, atol=1e-6, rtol=1e-6) for a, b in zip(actual, expected, strict=True)), name
    finally:
        if session is not None:
            session.restore(model)


def test_compiled_ple_tracks_base_and_changed_multi_lora_row_bindings(tmp_path, monkeypatch):
    monkeypatch.setattr(QE, "_PLE_COMPILE_MIN_SEEN", 1)
    model = _PLEModel()
    paths = _ple_artifacts(tmp_path)
    manager = MultiLoRAManager(model, max_loras=2, max_lora_rank=2)
    for name, path in paths.items():
        manager.register(name, path)
    hidden = mx.random.normal((2, 2, 64))
    embeddings = mx.random.normal((2, 2, 16))
    try:
        a, _ = manager.acquire("a")
        b, _ = manager.acquire("b")
        manager.bind_uid(1, a)
        manager.bind_uid(2, b)
        for rows in ((-1, -1), (1, 2), (2, 1), (-1, 1), (-1, -1)):
            manager.bind_rows(rows)
            actual, expected = _ple_head(model.ple, hidden, embeddings)
            assert all(mx.allclose(x, y, atol=1e-6, rtol=1e-6) for x, y in zip(actual, expected, strict=True)), rows
            manager.clear_rows()
        manager.unbind_uid(1)
        manager.unbind_uid(2)
        manager.release(a)
        manager.release(b)
    finally:
        manager.detach()


def test_ple_identity_oracle_detects_the_old_direct_weight_only_guard(tmp_path, monkeypatch):
    """The old guard accepts a different wrapper's already-compiled graph."""
    monkeypatch.setattr(QE, "_PLE_COMPILE_MIN_SEEN", 1)
    model = _PLEModel()
    paths = _ple_artifacts(tmp_path)
    hidden = mx.random.normal((1, 2, 64))
    embeddings = mx.random.normal((1, 2, 16))

    def old_guard():
        return tuple(value for module in (model.ple.key_proj, model.ple.value_proj,
                                          model.ple.conv1d, model.ple.norm_key,
                                          model.ple.norm_query, model.ple.norm_conv)
                     for name in ("weight", "scales", "biases")
                     if (value := getattr(module, name, None)) is not None)

    monkeypatch.setattr(model.ple, "_chain_params", old_guard)
    first = install_lora(model, name="a", path=paths["a"])
    _ple_head(model.ple, hidden, embeddings)
    first.restore(model)
    second = install_lora(model, name="b", path=paths["b"])
    try:
        stale, expected = _ple_head(model.ple, hidden, embeddings)
        assert any(not mx.allclose(a, b, atol=1e-6, rtol=1e-6)
                   for a, b in zip(stale, expected, strict=True))
    finally:
        second.restore(model)


def test_slot_eviction_and_reuse_preserve_queued_forward_and_apc_identity(tmp_path):
    paths = _artifacts(tmp_path)
    model = _Model()
    manager = MultiLoRAManager(model, max_loras=1, max_lora_rank=2)
    x = mx.arange(8, dtype=mx.float32).reshape(2, 4) / 8
    for name, path in paths.items():
        manager.register(name, path)
    identities, pending = {}, []
    try:
        for uid, name in enumerate(("a", "b", "a")):
            slot, residency = manager.acquire(name)
            assert slot == 1
            assert residency == ("loaded" if uid == 0 else "evicted")
            manager.bind_uid(uid, slot)
            manager.bind_rows([uid, -1])
            # Leave the first graph lazy until the slot has been overwritten.
            # Functional slot replacement must keep its arrays alive.
            pending.append((name, model(x)))
            manager.clear_rows()
            manager.unbind_uid(uid)
            manager.release(slot)
            fingerprint = manager.lookup(name)[1]
            identities.setdefault(name, lora_apc_scope("base", fingerprint))
        for name, output in pending:
            mx.eval(output)
            assert mx.allclose(output[0], _reference(x[:1], paths[name])[0], atol=1e-7)
            assert mx.array_equal(output[1], _reference(x[1:])[0])
        assert identities["a"] != identities["b"]
        assert not any(manager.pins.values())
        assert not manager.uid_slots
    finally:
        manager.detach()


def test_failed_lora_weight_load_restores_base_before_next_forward(tmp_path, monkeypatch):
    paths = _artifacts(tmp_path)
    model = _Model()
    original = model.linear
    x = mx.ones((1, 4))
    with monkeypatch.context() as patch:
        patch.setattr(mx, "load", lambda *_args, **_kwargs: (_ for _ in ()).throw(MemoryError("injected load")))
        with pytest.raises(MemoryError, match="injected load"):
            install_lora(model, name="a", path=paths["a"])
    assert model.linear is original
    assert mx.array_equal(model(x), _reference(x))
    session = install_lora(model, name="b", path=paths["b"])
    assert mx.array_equal(model(x), _reference(x, paths["b"]))
    session.restore(model)
