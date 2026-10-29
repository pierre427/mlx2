"""CPU metadata/reset/timing guards; no model or GPU execution."""
import importlib.util
from pathlib import Path
import sys
import subprocess

import numpy as np
import pytest

P = Path(__file__).resolve().parents[1] / "scripts/research/ds41_synthetic_kv_components.py"
spec = importlib.util.spec_from_file_location("synthetic_kv_cpu", P)
kv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kv)


@pytest.mark.parametrize("ratio", (1, 2))
@pytest.mark.parametrize("mode", kv.MODES)
def test_actual_methods_and_one_bounded_operation(ratio, mode):
    state, operation, meta = kv._make_case(np, 128, ratio, mode)
    before = state._pooled
    sample = kv._measure(np, operation, state)
    assert all(value >= 0 for value in sample.values())
    assert meta["input_array_bytes"] < 2 * 1024**3
    if mode == "append_growth":
        assert state._capacity == 256 and state._pooled is not before
    elif mode == "append_no_grow":
        assert state._pooled is before
    elif mode == "full_kv_storage_update":
        state.validate_state()
        assert state.offset == 128 and state._pool_count == 128 // ratio
    elif mode == "update_local":
        assert state.offset == 128 and state.keys.shape == (1, 1, 128, 512)


def test_largest_context_cap_and_invalid_shapes():
    for n in kv.CONTEXTS:
        for r in kv.RATIOS:
            for mode in kv.MODES:
                assert kv._shape_bytes(n, r, mode)[2] < 43_000_000
    for args in ((99, 1, "concat"), (100, 4, "concat"), (100, 1, "unknown")):
        with pytest.raises(ValueError):
            kv._shape_bytes(*args)


def test_resets_fresh_and_full_run_interleaves(monkeypatch):
    monkeypatch.setattr(kv, "CONTEXTS", (100,))
    monkeypatch.setattr(kv, "RATIOS", (2,))
    monkeypatch.setattr(kv, "REPETITIONS", 2)
    checks = []
    report = kv.run(np, lambda: checks.append(True))
    assert len(report["cells"]) == 7 and len(checks) == 14
    assert all(len(c["samples"]) == 2 for c in report["cells"])
    assert report["execution_order"][0][3] == "concat"
    assert report["execution_order"][7][3] == "local_owned_copy"
    assert report["checkpoint"] == "none-synthetic-no-weights"
    a, _, _ = kv._make_case(np, 100, 2, "append_growth")
    b, _, _ = kv._make_case(np, 100, 2, "append_growth")
    assert a._pooled is not b._pooled and a.keys is not b.keys


def test_no_mlx_import_at_module_load():
    # pytest's repository conftest imports MLX CPU; use a fresh interpreter.
    subprocess.run([sys.executable, "-c",
                    "import ds41_synthetic_kv_components, sys; "
                    "assert 'mlx.core' not in sys.modules"], check=True)
