"""Direct expert bindings must preserve every assignment and16-row tile order."""

import numpy as np
import pytest

from mlx2.runtime.gguf.reader import GGUFError
from mlx2.runtime.gguf.v41_row_prefill import GROUP_EXPERTS, routing_plan


@pytest.mark.parametrize("experts,rows", [(2,1), (13,17), (25,512), (384,1024)])
def test_grouped_plan_preserves_slot_order_and_covers_each_assignment_once(experts, rows):
    rng = np.random.default_rng(41024)
    ids = rng.integers(0, experts, (rows, min(6,experts)), np.uint32)
    inverse, groups = routing_plan(ids, experts)
    order = np.concatenate([g[2] for g in groups])
    np.testing.assert_array_equal(order[inverse], np.arange(ids.size))
    np.testing.assert_array_equal(ids.reshape(-1)[order], np.sort(ids.reshape(-1), kind="stable"))
    for begin, end, assignments, tiles in groups:
        assert 0 < end-begin <= GROUP_EXPERTS
        coverage = []
        for row, expert, valid, padding in tiles:
            assert 1 <= valid <= 16 and padding == 0
            selected = assignments[int(row):int(row+valid)]
            np.testing.assert_array_equal(ids.reshape(-1)[selected], np.full(valid, begin+expert))
            coverage.extend(range(int(row), int(row+valid)))
        assert coverage == list(range(len(assignments)))


def test_sparse_groups_keep_inverse_order_and_skip_empty_groups():
    ids = np.full((17, 1), 383, np.uint32)
    inverse, groups = routing_plan(ids, 384)
    assert len(groups) == 1 and groups[0][:2] == (372,384)
    np.testing.assert_array_equal(inverse, np.arange(17))
    np.testing.assert_array_equal(groups[0][3], [[0,11,16,0],[16,11,1,0]])


@pytest.mark.parametrize("ids,experts", [(np.zeros((0,6),np.uint32),6),
                                        (np.zeros((1,7),np.uint32),7),
                                        (np.array([[384]],np.uint32),384),
                                        (np.array([[-1]],np.int32),6)])
def test_invalid_route_never_publishes_a_partial_plan(ids, experts):
    with pytest.raises(GGUFError):
        routing_plan(ids,experts)


def test_profile_byte_oracle_distinguishes_signed_zero_dtype_and_shape(monkeypatch):
    import importlib
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts/research"))
    equal = importlib.import_module("profile_deepseek_v41").arrays_equal
    assert not equal(np.array([0.0],np.float32),np.array([-0.0],np.float32))
    assert not equal(np.zeros((1,2),np.float32),np.zeros((2,1),np.float32))
    assert not equal(np.zeros(1,np.float32),np.zeros(1,np.uint32))
    assert equal((None,np.array([1],np.float32)),(None,np.array([1],np.float32)))


@pytest.mark.parametrize("kind", ["gate", "down"])
@pytest.mark.parametrize("experts", [1, 3, GROUP_EXPERTS])
def test_row_group_sources_bind_every_pointer_and_template_constant(kind, experts):
    """The row-group kernels rewrite the shared V4.1 sources; every edit must
    land, every pointer it introduces be declared before use, and every
    template constant the source names be passed by the launch."""
    import re

    from mlx2.runtime.gguf import v41_row_prefill as rp

    names, source = rp.kernel_spec(kind, experts)
    pointers = ("row_w", "row_up") if kind == "gate" else ("row_w",)
    for pointer in pointers:
        declaration = f"device const uchar *{pointer}="
        assert source.count(declaration) == 1
        assert source.index(declaration) < source.index(f"{pointer}+ulong(row)*ROW_BYTES")
        assert source.index(declaration) < source.index("float sums[MT][16];")
    assert "row_up" not in source if kind == "down" else True
    # No unrebound expert-plane pointer: only w0..wN / up0..upN are inputs.
    assert not re.search(r"(?<![A-Za-z0-9_])(w|up)\+\(ulong\(expert\)", source)
    for token in re.findall(r"(?<![A-Za-z0-9_])(w|up)(\d+)(?![A-Za-z0-9_])", source):
        assert "".join(token) in names
    assert "*((device const uint4*)tiles+tile)" not in source
    stores = 3 if kind == "gate" else 1
    assert source.count("v41_bf16(") == stores
    template = dict(rp.launch_template(kind, 256, 512, 50, 6 if kind == "gate" else None))
    assert template["MT"] == 1
    constants = set(re.findall(r"(?<![A-Za-z0-9_])[A-Z][A-Z_]*(?![A-Za-z0-9_])", source))
    assert constants <= set(template), constants - set(template)


def test_row_group_spec_refuses_a_changed_template(monkeypatch):
    from mlx2.runtime.gguf import v41_prefill
    from mlx2.runtime.gguf import v41_row_prefill as rp

    monkeypatch.setattr(v41_prefill, "DOWN_SOURCE", v41_prefill.DOWN_SOURCE.replace("float sums[MT][16];",
                                                                                     "float sums[16];"))
    with pytest.raises(GGUFError, match="anchor changed"):
        rp.kernel_spec("down", 2)


@pytest.mark.parametrize("kind", ["gate", "down"])
def test_row_group_kernels_compile_offline_with_their_launch_templates(tmp_path, kind):
    """CPU-only compile with the launch's own template constants (no GPU run)."""
    import shutil
    import subprocess

    from mlx2.runtime.gguf import v41_prefill
    from mlx2.runtime.gguf import v41_row_prefill as rp

    if not shutil.which("xcrun") or subprocess.run(["xcrun", "--find", "metal"], capture_output=True).returncode:
        pytest.skip("offline Metal compiler unavailable")
    names, source = rp.kernel_spec(kind, 3)
    template = rp.launch_template(kind, 4096, 2048, 1100, 6 if kind == "gate" else None)
    dtype = lambda n: "float" if n in ("x", "scores") else "uchar" if n[0] in "wu" else "uint"
    bindings = [f"device const {dtype(n)}* {n} [[buffer({i})]]" for i, n in enumerate(names)]
    bindings.append(f"device float* y [[buffer({len(names)})]]")
    attrs = ["uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]",
             "uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]",
             "uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]",
             "uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]"]
    constants = "constant uint "+",".join(f"{k}={v}" for k, v in template)+";\n"
    path = tmp_path/(kind+".metal")
    path.write_text("#include <metal_stdlib>\nusing namespace metal;\n"+v41_prefill.HEAD+constants
                    +"kernel void probe("+",".join(bindings+attrs)+") {\n"+source+"\n}\n")
    result = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-c", str(path),
                             "-o", str(path.with_suffix(".air"))], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
