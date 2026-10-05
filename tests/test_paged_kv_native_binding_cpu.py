"""Optional compiled binding checks; no arena creation or Metal submission."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_native_array_and_stream_conversion_without_gpu_work():
    native = pytest.importorskip("_paged_kv_native")
    import mlx.core as mx

    key = mx.array([1, 2], dtype=mx.uint8)
    cpu_stream = mx.default_stream(mx.cpu)
    # A CPU stream is rejected after both MLX array and Stream conversions.
    assert native.validate_source_types(key, key, cpu_stream) is False
    with pytest.raises(TypeError, match="incompatible function arguments"):
        native.validate_source_types(object(), key, cpu_stream)
    assert native.write.__doc__.count("mlx.core.array") == 3


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
