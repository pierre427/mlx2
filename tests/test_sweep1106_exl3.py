"""Bug sweep 2026-11-06, lane exl3: CPU regressions."""

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from mlx2.runtime.models import exl3
from mlx2.runtime.trellis.upstream import (
    Exl3Plan,
    TrellisLinearSpec,
    install_upstream,
)

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _dense_tensors(rng, i, o, words, bias=False):
    t = {
        "trellis": mx.array(rng.integers(0, 65536, (i // 16, o // 16, words), np.uint16)),
        "suh": mx.array((rng.normal(0, 1, i) * 0.03).astype(np.float16)),
        "svh": mx.array(np.sign(rng.normal(size=o)).astype(np.float16)),
    }
    if bias:
        t["bias"] = mx.array(rng.normal(0, 1, o).astype(np.float16))
    return t


# EXL3-1: a MUL1 checkpoint is admitted by the plan (the host oracle decodes
# it) but the trellis modules decode MCG only; install must refuse it.
def test_install_refuses_codebooks_the_runtime_cannot_decode():
    rng = np.random.default_rng(1)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(128, 128, bias=False)

    for codebook, refused in (("mul1", True), ("mcg", False)):
        spec = TrellisLinearSpec("proj", 128, 128, 32, codebook, False)
        plan = Exl3Plan({"codebook": codebook}, {"proj": spec}, {}, frozenset())
        tensors = _dense_tensors(rng, 128, 128, 32)
        model = Tiny()

        def read(name, tensors=tensors):
            return tensors[name.rsplit(".", 1)[-1]]

        if refused:
            with pytest.raises(ValueError, match="mul1 has no runtime decoder"):
                install_upstream(model, read, plan, lambda key: key)
            assert type(model.proj) is nn.Linear  # nothing swapped
        else:
            install_upstream(model, read, plan, lambda key: key)
            assert isinstance(model.proj, exl3.Exl3DenseLinear)


# EXL3-2: the nax_fused receipt names the width thresholds its laws use.
@pytest.fixture
def _qwen36_profile(monkeypatch):
    from mlx2.runtime.models import qwen3_next

    monkeypatch.setattr(qwen3_next, "_MOE_SHARED_IN_GATHER", False)
    monkeypatch.setattr(qwen3_next, "_MOE_FUSED_GATE_UP", False)


@pytest.mark.parametrize("execution", ["nax", "nax_fused"])
def test_nax_receipts_name_their_width_thresholds(tmp_path, execution, _qwen36_profile):
    sys.path.insert(0, str(ROOT / "tests"))
    try:
        from test_exl3_upstream_model_cpu import _checkpoints, _load_exl3
    finally:
        sys.path.remove(str(ROOT / "tests"))
    exl3_path, _ = _checkpoints(tmp_path)
    _, receipt = _load_exl3(exl3_path, execution)
    assert receipt["nax_min_rows"] == exl3.NAX_MIN_ROWS
    assert receipt["nax_bank_min_rows"] == exl3.NAX_BANK_MIN_ROWS


# EXL3-3: the shared-slot decode tail reads only the shared banks, so a
# biased shared projection must not ride as slots (its bias would vanish).
def test_shared_slots_refuse_a_biased_shared_projection():
    rng = np.random.default_rng(2)
    e, h, i, words = 2, 128, 128, 32

    def bank(i_dim, o_dim):
        return exl3.Exl3Linear(
            mx.array(rng.integers(0, 65536, (e, i_dim // 16, o_dim // 16, words), np.uint16)),
            mx.array((rng.normal(0, 1, (e, i_dim)) * 0.03).astype(np.float16)),
            mx.array(np.sign(rng.normal(size=(e, o_dim))).astype(np.float16)))

    routed = exl3.Exl3SwitchGLU(bank(h, i), bank(h, i), bank(i, h), execution="fused_chain_nax")

    def mlp(down_bias):
        parts = [exl3.Exl3DenseLinear(**_dense_tensors(rng, h, i, words), law="fused_nax"),
                 exl3.Exl3DenseLinear(**_dense_tensors(rng, h, i, words), law="fused_nax"),
                 exl3.Exl3DenseLinear(**_dense_tensors(rng, i, h, words, bias=down_bias), law="fused_nax")]
        return exl3.Exl3FusedMLP(*parts)

    assert exl3.shared_slot_refusal(routed, mlp(False)) is None
    assert exl3.shared_slot_refusal(routed, mlp(True)) == "shared down_proj carries a bias"


# EXL3-4: the serving probes' reference children import this tree and the
# reference arm loads the same resident choice as the server arm.
@pytest.mark.parametrize("script", ["probe_exl3_serving", "classify_exl3_divergence"])
def test_probe_scripts_import_this_tree_without_pythonpath(script):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    code = (f"import sys; sys.path.insert(0, {str(ROOT / 'scripts')!r}); import {script}; "
            "import mlx2; print(mlx2.__file__)")
    out = subprocess.run([PY, "-c", code], env=env, cwd="/", capture_output=True, text=True, check=True)
    assert Path(out.stdout.strip()).is_relative_to(ROOT / "src")


def test_probe_reference_arm_keeps_the_resident_choice(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import probe_exl3_serving as probe
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    args = Namespace(model=tmp_path / "m", output=tmp_path / "o", execution="nax_fused", resident_dense="off")
    command, env = probe.reference_command(args)
    assert command[command.index("--resident-dense") + 1] == "off"
    assert "--reference-only" in command
    assert env["PYTHONPATH"] == str(ROOT / "src")
    assert probe.execution_policy(args) == {"exl3_execution": "nax_fused", "exl3_resident_dense": "off"}
    default = Namespace(**{**vars(args), "resident_dense": None})
    assert "--resident-dense" not in probe.reference_command(default)[0]
    assert probe.execution_policy(default) == {"exl3_execution": "nax_fused"}
    # The child parses what the parent forwards.
    parsed = subprocess.run([PY, str(ROOT / "scripts/probe_exl3_serving.py"), "--help"],
                            capture_output=True, text=True, check=True).stdout
    assert "--resident-dense" in parsed
    json.dumps(probe.execution_policy(args))
