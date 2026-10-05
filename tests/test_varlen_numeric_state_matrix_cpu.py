"""Preflight the bounded GPU matrix without constructing Metal objects."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/research/varlen_numeric_state_matrix.py"
SPEC = importlib.util.spec_from_file_location("varlen_numeric_state_matrix", PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_all_matrix_geometries_are_admissible():
    gate.validate_cases()
    assert len(gate.CASES) == 6
    assert {case[3] for case in gate.CASES} == {128, 256}
    assert any(case[-1] for case in gate.CASES)
    assert any(len(case[4]) > 1 for case in gate.CASES)


def test_invalid_gqa_ratio_refused_before_gpu(monkeypatch):
    monkeypatch.setattr(gate, "CASES", (("bad-gqa", 1, 2, 128, (1,), (1,), None),))
    with pytest.raises(ValueError, match="invalid native matrix case"):
        gate.validate_cases()


def test_bf16_opaque_native_read_fails_closed_before_mlx_import(monkeypatch):
    import mlx2.runtime.paged_attention_native as native_read
    from mlx2.runtime.paged_attention_pack import PackedTokenRead

    class FakeBackend:
        pass

    backend = FakeBackend()
    writer = SimpleNamespace(backend=backend, poisoned=False)
    packed = PackedTokenRead(writer, SimpleNamespace(dtype="bfloat16"), None, None)
    monkeypatch.setattr(native_read, "NativeWriteBackend", FakeBackend)
    with pytest.raises(ValueError, match="fp16 only"):
        native_read.native_paged_attention_read_fp16(
            packed, backend, object(), object(), permit_candidate=True)
