"""Optional compiled binding checks; no arena creation or Metal submission."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_native_array_and_stream_conversion_without_gpu_work():
    # Run in a fresh interpreter so a synthetic module installed by another
    # test cannot masquerade as the optional compiled binding.
    probe = r'''
import importlib
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.find_spec("_paged_kv_native")
if spec is None:
    raise SystemExit(5)
if not isinstance(spec.loader, importlib.machinery.ExtensionFileLoader):
    raise RuntimeError(f"not a compiled extension: {spec.loader!r}")
native = importlib.import_module("_paged_kv_native")
origin = Path(spec.origin).resolve()
if Path(native.__file__).resolve() != origin:
    raise RuntimeError(f"loaded native identity mismatch: {native.__file__} != {origin}")
if not any(str(origin).endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES):
    raise RuntimeError(f"unexpected extension suffix: {origin}")

import mlx.core as mx

key = mx.array([1, 2], dtype=mx.uint8)
cpu_stream = mx.default_stream(mx.cpu)
assert native.validate_source_types(key, key, cpu_stream) is False
try:
    native.validate_source_types(object(), key, cpu_stream)
except TypeError as exc:
    assert "incompatible function arguments" in str(exc)
else:
    raise AssertionError("non-array source unexpectedly accepted")
assert native.write.__doc__.count("mlx.core.array") == 3
'''
    result = subprocess.run(
        [sys.executable, "-c", probe], text=True, capture_output=True, check=False
    )
    if result.returncode == 5:
        pytest.skip("optional compiled _paged_kv_native binding unavailable")
    assert result.returncode == 0, result.stderr


def test_native_backend_remains_default_off():
    from mlx2.runtime.paged_kv_write import NativeWriteBackend

    with pytest.raises(RuntimeError, match="explicit candidate"):
        NativeWriteBackend(64, object())


def test_gpu_probe_requires_explicit_flag_before_optional_imports(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/research/varlen_native_write_gpu_probe.py"
    receipt = tmp_path / "receipt.json"
    result = subprocess.run(
        [sys.executable, "-S", str(script), "--receipt", str(receipt)],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert "--execute-gpu" in result.stderr
    assert not receipt.exists()
