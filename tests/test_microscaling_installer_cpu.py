"""SC-MXFP4 instance ownership, ordinary-decode and APCv2 identity contracts.

CPU synthetic tensors only; no target checkpoint or GPU qualification.
"""
from contextlib import contextmanager
from types import SimpleNamespace

import mlx.core as mx
from mlx import nn
import numpy as np
import pytest

from mlx2.contracts import Fidelity
from mlx2.runtime.microscaling import installer as mi
from mlx2.runtime.microscaling.format import pack_numpy, to_mxfp4_scales_numpy
from mlx2.runtime.models.switch_layers import QuantizedSwitchLinear


@pytest.fixture(autouse=True)
def cpu_device(monkeypatch):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mi, "require_supported_device", lambda: "synthetic-cpu-device")
    yield
    mx.set_default_device(previous)


def linear(*, grouped=False, mode="mxfp4", bias=False):
    k, n, e = 256, 128, 4
    if grouped:
        module = QuantizedSwitchLinear(k, n, e, bias=bias, group_size=32,
            bits=4, mode=mode)
    else:
        module = nn.QuantizedLinear(k, n, bias=bias, group_size=32,
            bits=4, mode=mode)
    if mode == "mxfp4":
        shape = (e, n, k) if grouped else (n, k)
        packed = pack_numpy(np.random.default_rng(7).normal(0, .02, shape),
            allow_requantization=True)
        module.weight = mx.array(packed.codes)
        module.scales = mx.array(to_mxfp4_scales_numpy(packed))
    return module


class ToyModel(nn.Module):
    def __init__(self, *, grouped=False, mode="mxfp4"):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate_proj = linear(grouped=grouped, mode=mode)
        self.grouped = grouped
        self.raise_after_projection = False

    def __call__(self, inputs, cache=None, *, assignment_rows=None):
        rows = assignment_rows or int(inputs.size)
        if self.grouped:
            x = mx.ones((rows, 1, 256), mx.bfloat16)
            indices = mx.zeros((rows,), mx.uint32)
            y = self.mlp.gate_proj(x, indices, sorted_indices=True)
        else:
            x = mx.ones((*inputs.shape, 256), mx.bfloat16)
            y = self.mlp.gate_proj(x)
        if self.raise_after_projection:
            raise RuntimeError("model failure")
        return y


def install(model, policy="all", **kwargs):
    return mi.apply(model, policy, artifact_fingerprint="synthetic-sc32@fixture:7", **kwargs)


def fake_kernels(monkeypatch):
    from mlx2.runtime.microscaling import kernels
    calls = []
    def dense(x, codes, bcodes, scales):
        calls.append(("dense", x.shape))
        return mx.zeros((*x.shape[:-1], codes.shape[-2]), mx.bfloat16)
    def grouped(x, indices, codes, bcodes, scales, sorted_indices=False):
        calls.append(("grouped", x.shape, sorted_indices))
        return mx.zeros((*indices.shape, 1, codes.shape[-2]), mx.bfloat16)
    monkeypatch.setattr(kernels, "dense", dense)
    monkeypatch.setattr(kernels, "grouped", grouped)
    return calls


def test_policy_is_opt_in_and_approximate():
    assert not mi.MicroscalingPolicy().enabled
    assert mi.MicroscalingPolicy().fidelity is Fidelity.EXACT
    assert mi.MicroscalingPolicy.from_value("experts").fidelity is Fidelity.APPROXIMATE
    assert mi.MicroscalingPolicy.from_value(True).scope == "experts"
    for value in (None, False, "off"):
        assert mi.MicroscalingPolicy.from_value(value) == mi.MicroscalingPolicy()
    with pytest.raises(ValueError): mi.MicroscalingPolicy(row_threshold=True)
    with pytest.raises(ValueError): mi.MicroscalingPolicy(row_threshold=1)
    with pytest.raises(ValueError): mi.MicroscalingPolicy(scope="auto")
    with pytest.raises(ValueError): mi.MicroscalingPolicy.from_value({"enabled": True, "other": 1})
    mi.validate_decode_row_bound("all", 511)
    with pytest.raises(mi.MicroscalingError, match="assignment bound"):
        mi.validate_decode_row_bound("all", 512)
    with pytest.raises(mi.MicroscalingError, match="assignment bound"):
        mi.validate_decode_row_bound("experts", 68 * 10)


def test_disabled_install_has_no_side_effects_or_device_probe(monkeypatch):
    model = ToyModel()
    original = type(model.mlp.gate_proj)
    monkeypatch.setattr(mi, "require_supported_device", lambda: pytest.fail("device queried"))
    handle = mi.apply(model, "off")
    assert type(model.mlp.gate_proj) is original
    assert type(model) is ToyModel
    assert handle.receipt() is None and not handle.active
    assert mi.apc_semantic_fingerprint(("base",), handle) == ("base",)
    assert not mi.remove(handle)


def test_per_model_scope_and_exact_class_restoration(monkeypatch):
    model, draft = ToyModel(), ToyModel()
    original = type(model.mlp.gate_proj)
    original_weight = model.mlp.gate_proj.weight
    handle = install(model)
    assert isinstance(model.mlp.gate_proj, original)
    assert type(model.mlp.gate_proj) is not original
    assert type(draft.mlp.gate_proj) is original
    assert model.mlp.gate_proj.weight is original_weight
    assert handle.status()["module_kinds"] == {"dense": 1, "grouped": 0}
    assert handle.status()["resident_int8_weight_bytes"] == 0
    assert handle.metadata_bytes() == 128 * (256 // 128 + 4)
    assert handle.unfilled_reserve_bytes() > 128 * 256
    assert mi.remove(handle)
    assert type(model.mlp.gate_proj) is original and type(model) is ToyModel
    assert not mi.remove(handle)
    assert not handle._bound and handle.unfilled_reserve_bytes() == 0


def test_decode_and_bare_calls_stay_ordinary_even_above_assignment_threshold(monkeypatch):
    calls = fake_kernels(monkeypatch)
    model = ToyModel(grouped=True)
    reference = model(mx.zeros((1, 1), mx.int32), assignment_rows=600)
    handle = install(model, "experts")
    actual = model(mx.zeros((1, 1), mx.int32), assignment_rows=600)
    assert np.array_equal(np.array(actual.astype(mx.float32)), np.array(reference.astype(mx.float32)))
    assert calls == [] and handle.counts["fallback_phase"] == 1
    # A standalone projection call cannot fabricate a prefill phase.
    model.mlp.gate_proj(mx.ones((600, 1, 256), mx.bfloat16), mx.zeros((600,), mx.uint32), True)
    assert calls == []
    model(mx.zeros((1, 2), mx.int32), assignment_rows=600)
    assert calls[-1][0] == "grouped" and handle.counts["grouped_calls"] == 1
    assert handle.unfilled_reserve_bytes() > 0
    mi.remove(handle)


def test_prefill_threshold_dtype_and_exception_scope(monkeypatch):
    calls = fake_kernels(monkeypatch)
    model = ToyModel()
    handle = install(model)
    model(mx.zeros((1, 511), mx.int32))
    assert not calls and handle.counts["fallback_rows"] == 1
    model(mx.zeros((1, 512), mx.int32))
    assert len(calls) == 1 and handle.counts["engaged_rows"] == 512
    with handle.prefill_scope():
        model.mlp.gate_proj(mx.ones((512, 256), mx.float32))
    assert handle.counts["fallback_dtype"] == 1 and len(calls) == 1
    model.raise_after_projection = True
    with pytest.raises(RuntimeError, match="model failure"):
        model(mx.zeros((1, 512), mx.int32))
    assert handle._prefill.get() is None
    model.raise_after_projection = False
    model(mx.zeros((1, 1), mx.int32))
    assert len(calls) == 2  # failed forward dispatched, next decode did not
    mi.remove(handle)


def test_verify_and_stepped_cache_never_admit_prefill(monkeypatch):
    calls = fake_kernels(monkeypatch)
    model = ToyModel()
    handle = install(model)
    for cache in ([SimpleNamespace(speculating=True)], [SimpleNamespace(_step_lengths=[1])]):
        model(mx.zeros((1, 512), mx.int32), cache=cache)
    from mlx2.runtime import verify_scope
    with verify_scope.verify_forward():
        model(mx.zeros((1, 512), mx.int32))
    assert not calls and handle.counts["fallback_phase"] == 3
    mi.remove(handle)


def test_lossless_metadata_refusal_is_atomic():
    model = ToyModel()
    model.mlp.up_proj = linear()
    scales = np.full(model.mlp.up_proj.scales.shape, 127, np.uint8)
    scales[:, -1] = 131
    model.mlp.up_proj.scales = mx.array(scales)
    original_types = [type(model.mlp.gate_proj), type(model.mlp.up_proj)]
    with pytest.raises(mi.MicroscalingError, match="not losslessly"):
        install(model)
    assert type(model) is ToyModel
    assert [type(model.mlp.gate_proj), type(model.mlp.up_proj)] == original_types
    assert mi._STATE_ATTR not in model.mlp.gate_proj.__dict__


def test_affine_weights_are_never_runtime_requantized():
    model = ToyModel(mode="affine")
    original = model.mlp.gate_proj.weight
    with pytest.raises(mi.MicroscalingError, match="no eligible MXFP4"):
        install(model)
    assert model.mlp.gate_proj.weight is original
    with pytest.raises(mi.MicroscalingError, match="fingerprint"):
        mi.apply(ToyModel(), "all")


def test_revision_binds_artifact_mask_threshold_and_metadata():
    models = [ToyModel() for _ in range(4)]
    handles = [install(models[0]), install(models[1], {"enabled":True,"scope":"all","row_threshold":1024}),
        mi.apply(models[2], "all", artifact_fingerprint="other-artifact"), install(models[3])]
    assert handles[0].revision == handles[3].revision
    assert len({h.revision for h in handles[:3]}) == 3
    assert mi.apc_semantic_fingerprint("ordinary", handles[0]) != "ordinary"
    assert handles[0].receipt()["qualified"] is False
    models[0].mlp.gate_proj.scales = models[0].mlp.gate_proj.scales + 0
    with pytest.raises(mi.MicroscalingError, match="revision-bound"):
        models[0](mx.zeros((1, 1), mx.int32))
    for handle in handles:
        mi.remove(handle)


def test_scope_alias_and_adapter_declarations():
    model = ToyModel()
    model.outside = model.mlp.gate_proj
    with pytest.raises(mi.MicroscalingError, match="alias"):
        install(model, "mlp")
    adapter = SimpleNamespace(model=ToyModel(), identity={"fingerprint":"fixture"})
    with pytest.raises(mi.MicroscalingError, match="does not declare"):
        mi.apply_for_adapter(adapter, "all")
    adapter.microscaling_supported = lambda: {"mlp", "all"}
    handle = mi.apply_for_adapter(adapter, "all")
    with pytest.raises(mi.MicroscalingError, match="already carries"):
        mi.apply_for_adapter(adapter, "all")
    assert mi.remove(handle)


def test_body_forward_uses_structural_input_trunk_boundary(monkeypatch):
    calls = fake_kernels(monkeypatch)
    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = ToyModel()
        def __call__(self, inputs, cache=None):
            return self.trunk(inputs, cache)
        def prefill_body(self, inputs, cache=None):
            return self.trunk(inputs, cache)
    model = Outer()
    base = type(model.trunk)
    handle = install(model)
    model.prefill_body(mx.zeros((1, 512), mx.int32))
    assert len(calls) == 1 and handle.counts["engaged_calls"] == 1
    model.prefill_body(mx.zeros((1, 1), mx.int32))
    assert len(calls) == 1
    assert mi.remove(handle)
    assert type(model.trunk) is base


def test_real_switch_paths_are_selectable_without_affine_exclusions():
    module = linear(grouped=True)
    for scope in ('experts','mlp','all'):
        assert mi.default_select(scope)('model.layers.0.mlp.switch_mlp.gate_proj', module)
        assert mi.default_select(scope)('model.layers.0.mlp.experts.down_proj', module)
        assert not mi.default_select(scope)('mtp.layers.0.mlp.switch_mlp.gate_proj', module)
    assert not mi.default_select('experts')('model.layers.0.mlp.gate_proj', linear())
