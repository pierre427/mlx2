from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).parents[1] / "scripts/research/bench_large_m_qmm.py"
SPEC = importlib.util.spec_from_file_location("bench_large_m_qmm", PATH)
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_default_plan_covers_both_real_projection_geometries_and_row_scales():
    rows = M.plan(list(M.SHAPES), list(M.DEFAULT_ARMS), list(M.DEFAULT_MS))
    assert {row["shape"] for row in rows} == {"gate_up", "down"}
    assert {row["M"] for row in rows} == {6985, 20883}
    assert all(row["bits"] == 4 and row["group_size"] == 64 for row in rows)


def test_chunk_arm_records_actual_maximum_qmm_rows():
    rows = M.plan(["gate_up"], ["stock", "chunk4096"], [20883])
    assert rows[0]["effective_qmm_M_max"] == 20883
    assert rows[1]["effective_qmm_M_max"] == 4096


@pytest.mark.parametrize("arm,size", [("chunk1", 1), ("chunk8192", 8192)])
def test_chunk_size(arm, size):
    assert M.chunk_size(arm) == size


@pytest.mark.parametrize("arm", ["chunk0", "chunkx", "other"])
def test_bad_chunk_arm_fails_closed(arm):
    with pytest.raises(ValueError):
        M.chunk_size(arm)


def test_plan_requires_stock_control():
    with pytest.raises(ValueError, match="stock"):
        M.plan(["gate_up"], ["chunk4096"], [6985])


def test_affine_working_set_uses_packed_weights_and_bf16_metadata():
    n, k = M.SHAPES["gate_up"]
    expected = n * k // 2 + 4 * n * (k // 64)
    assert M.affine_bytes(n, k) == expected
    assert M.copies_for_target(n, k, 0) == 1
    copies = M.copies_for_target(n, k, 1024)
    assert copies * expected >= 1024 << 20
    assert (copies - 1) * expected < 1024 << 20


def test_plan_rejects_unknown_geometry():
    with pytest.raises(ValueError, match="unknown shapes"):
        M.plan(["unknown"], ["stock"], [1])


def test_dry_run_never_imports_mlx(monkeypatch, capsys):
    real_import = __import__

    def guarded(name, *args, **kwargs):
        if name == "mlx" or name.startswith("mlx."):
            raise AssertionError("dry-run imported MLX")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded)
    assert M.main(["--dry-run", "--shapes", "gate_up", "--ms", "6985",
                   "--arms", "stock,chunk4096"]) == 0
    assert '"effective_qmm_M_max": 4096' in capsys.readouterr().out
