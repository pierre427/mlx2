"""Static inventory of the threadgroup sizes mlx2's custom Metal kernels request.

TensorFold #704/#715: on M1 a register-heavy pipeline allows 768 or 832
threads per threadgroup, and a host that dispatched 1024 anyway got zeros.
MLX's ``mx.fast.metal_kernel`` (pinned 39400a0d4,
``mlx/backend/metal/custom_kernel.cpp``) checks the requested size against the
pipeline's ``maxTotalThreadsPerThreadgroup`` and raises ``invalid_argument``,
so mlx2 fails closed rather than silently.  The limit itself is only known
after the pipeline compiles, so this test cannot check it.  What it pins:

* no launch requests more than 1024 threads (the Apple GPU ceiling);
* every launch above 768 threads, the floor TensorFold measured on M1, is
  listed below with the family that reaches it, so a new one is a decision.

Runtime check on M1-M4 hosts: GPU-deferred (see provenance
metal-peer-hazards-20261014.json).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "mlx2"
APPLE_MAX_THREADS = 1024
M1_REGISTER_HEAVY_FLOOR = 768

# file (relative to src/mlx2) -> number of launches above the M1 floor.
ABOVE_M1_FLOOR = {
    # Flash-Next (qwen4) attention rows / window and indexed QSA: 1024.
    "runtime/models/qwen4_attn_rows.py": 2,
    "runtime/models/qwen4_attn_window.py": 2,
    "runtime/models/qwen4_qsa_indexed.py": 1,
    # Flash-Next fused GDN decode: 32 x 32.
    "runtime/models/qwen4_fused_gdn.py": 1,
    # NAX (M5-only) MoE gather and its research twin.
    "runtime/models/moe_nax_gather.py": 1,
    "adapters/segmented_moe_prefill_research.py": 1,
    # Muse D128 attention window: the one_combine/two_combine reductions
    # launch 1024; the fused window runs only on device classes s/d
    # (_serial_plan returns None elsewhere), so M1-M4 keep the stock SDPA.
    "runtime/models/muse_attn_window.py": 2,
    # DeepSeek V4.1 Source mHC ``mega_v1`` pre kernel (ea78d82ff): one
    # 1024-thread threadgroup matching MLX's 1024-lane reduction laws. Opt-in,
    # not selected by default, behind a first-use bitwise self-check.
    "runtime/models/deepseek_v41_source_hc_mega.py": 1,
    # GLM-5.3 opt-in decode kernels (glm53_hc=mega_v1, glm53_hc_fused,
    # glm53_kda_fused=decode_v1/v2): none is selected by default.
    "runtime/models/glm53_hc_fused.py": 4,
    "runtime/models/glm53_hc_mega.py": 2,
    "runtime/models/glm53_kda_fused.py": 2,
    # EXL3 trellis kernels: fused_rows and its residual twin launch 256 * KS
    # threads (fused_ksplit caps KS at 4, i.e. 1024; the default FUSED_KS = 2
    # launches 512), and fused_gate_up launches a fixed 1024. On a register-
    # heavy M1 pipeline MLX raises rather than returning zeros; the M1-M4
    # runtime check is GPU-deferred like the rest of this list.
    "runtime/models/exl3.py": 3,
    # (QSA sidecar qsa_select.py is not listed: since 2026-11-06 it picks its
    # threadgroup per device, 1024 on applegpu_g17+ and 768 elsewhere; see
    # test_qsa_select_threadgroup_is_device_dependent.)
}


def _constant_ints(tree: ast.AST) -> dict[str, int]:
    """Module-level ``NAME = <int>`` assignments."""
    names: dict[str, int] = {}
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, int):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names[target.id] = node.value.value
    return names


def _evaluate(node: ast.AST, names: dict[str, int]) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Add)):
        left, right = _evaluate(node.left, names), _evaluate(node.right, names)
        if left is None or right is None:
            return None
        return left * right if isinstance(node.op, ast.Mult) else left + right
    return None


def threadgroup_requests(source: str) -> list[tuple[int, int | None]]:
    """(line, total threads or None when not static) per ``threadgroup=`` keyword."""
    tree = ast.parse(source)
    names = _constant_ints(tree)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != "threadgroup" or not isinstance(keyword.value, ast.Tuple):
                continue
            total = 1
            for element in keyword.value.elts:
                value = _evaluate(element, names)
                if value is None:
                    total = None
                    break
                total *= value
            found.append((keyword.value.lineno, total))
    return found


def _inventory():
    rows = {}
    for path in sorted(SRC.rglob("*.py")):
        requests = threadgroup_requests(path.read_text())
        if requests:
            rows[path.relative_to(SRC).as_posix()] = requests
    return rows


def test_scanner_sees_a_planted_oversized_threadgroup():
    planted = (
        "THREADS = 512\n"
        "k(grid=(1, 1, 1), threadgroup=(THREADS * 4, 1, 1))\n"
        "k(grid=(1, 1, 1), threadgroup=(32, n, 1))\n"
    )
    assert threadgroup_requests(planted) == [(2, 2048), (3, None)]


def test_no_launch_exceeds_the_apple_ceiling():
    inventory = _inventory()
    # The scan must see the launch sites, or the test passes vacuously.
    assert sum(len(v) for v in inventory.values()) >= 100
    over = {
        (path, line, total)
        for path, requests in inventory.items()
        for line, total in requests
        if total is not None and total > APPLE_MAX_THREADS
    }
    assert not over


def test_launches_above_the_m1_floor_are_listed():
    inventory = _inventory()
    above = {}
    for path, requests in inventory.items():
        count = sum(1 for _, total in requests if total and total > M1_REGISTER_HEAVY_FLOOR)
        if count:
            above[path] = count
    assert above == ABOVE_M1_FLOOR


@pytest.mark.parametrize("path", sorted(ABOVE_M1_FLOOR))
def test_listed_files_exist(path):
    assert (SRC / path).is_file()


def test_qsa_select_threadgroup_is_device_dependent(monkeypatch):
    monkeypatch.syspath_prepend(str(SRC.parent))
    mx = pytest.importorskip("mlx.core")
    from mlx2.runtime.models import qsa_select

    assert qsa_select.TG_FLOOR == M1_REGISTER_HEAVY_FLOOR
    assert qsa_select.select_threadgroup("applegpu_g17s") == 1024
    assert qsa_select.select_threadgroup("applegpu_g18p") == 1024
    for older in ("applegpu_g13g", "applegpu_g14s", "applegpu_g15d", "applegpu_g16p", "", "cpu"):
        assert qsa_select.select_threadgroup(older) == M1_REGISTER_HEAVY_FLOOR
    # fused_select's default really comes from the device.
    for arch, want in (("applegpu_g13g", 768), ("applegpu_g17s", 1024)):
        monkeypatch.setattr(mx, "device_info", lambda device=None, arch=arch: {"architecture": arch})
        assert qsa_select._device_threadgroup.__wrapped__() == want
