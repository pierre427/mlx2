from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).parents[1] / "scripts/research/bench_large_m_mlp_layer.py"
SPEC = importlib.util.spec_from_file_location("bench_large_m_mlp_layer", PATH)
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_plan_is_complete_four_arm_layer_graph():
    rows = M.plan([6985, 20883], 1024)
    assert {row["M"] for row in rows} == {6985, 20883}
    assert {row["arm"] for row in rows} == set(M.ARMS)
    assert len(rows) == 8
    assert all(row["copies"] >= 2 for row in rows)


def test_layer_working_set_has_gate_up_and_down_tables():
    expected = 2 * M.affine_bytes(M.HIDDEN, M.DIM)
    expected += M.affine_bytes(M.DIM, M.HIDDEN)
    assert M.layer_affine_bytes() == expected
    copies = M.copies_for_target(1024)
    assert copies * expected >= 1024 << 20
    assert (copies - 1) * expected < 1024 << 20


@pytest.mark.parametrize("bad", ["", "0", "-1", "1,0"])
def test_bad_m_fails_closed(bad):
    with pytest.raises(ValueError, match="positive"):
        M.parse_ms(bad)


def test_dry_run_does_not_import_mlx(monkeypatch, capsys):
    real_import = __import__

    def guarded(name, *args, **kwargs):
        if name == "mlx" or name.startswith("mlx."):
            raise AssertionError("dry-run imported MLX")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded)
    assert M.main(["--dry-run", "--ms", "20883"]) == 0
    assert M.SCHEMA in capsys.readouterr().out
