"""CPU source checks: a destination cooperative tensor reused as a matmul left
input must be guarded by MPP ``is_compatible_as_left_input``.

``matmul2d::get_left_input_cooperative_tensor<...>(src)`` converts a
destination cooperative tensor into the op's left-input layout.  macOS 27
MetalPerformancePrimitives exposes ``is_compatible_as_left_input<...>(src)``
for that conversion; it is a runtime call into the TensorOps backend (not a
``constexpr``), so the guard is a simdgroup-uniform branch, and the
incompatible arm routes P through threadgroup memory (store,
``simdgroup_barrier(mem_threadgroup)``, run from the threadgroup tensor).

The no-argument ``get_left_input_cooperative_tensor<...>()`` form only makes
an empty input tensor the kernel fills itself; it reuses no destination
layout, so the compatibility query does not apply to it.

Nothing here compiles or launches Metal.
"""

import hashlib
import re
from pathlib import Path

import pytest

from mlx2.runtime.tensorfold_qwen38 import lane_attention as la
from mlx2.runtime.tensorfold_qwen38 import stream_attention as sa

SRC = Path(__file__).resolve().parents[1] / "src" / "mlx2" / "runtime"
MODULES = (
    "tensorfold_qwen38/lane_attention.py",
    "tensorfold_qwen38/stream_attention.py",
    "int8_prefill.py",
    "models/qwen4_qsa_stage1.py",
    "models/deepseek_v41_omlx_nax_attention.py",
    "models/qwen4_qsa_nax.py",
)
# get_left_input_cooperative_tensor<...>(arg): the destination-reuse form.
_REUSE = re.compile(r"get_left_input_cooperative_tensor<[^>]*>\((?!\))")
_EMPTY = re.compile(r"get_left_input_cooperative_tensor<[^>]*>\(\)")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


_KERNELS = {
    "lane_attention.partial_direct": lambda: la._PARTIAL_DIRECT,
    "lane_attention.partial_direct_128": lambda: la._PARTIAL_DIRECT_128,
    "stream_attention.partial": lambda: sa.sources()["partial"],
}


@pytest.mark.parametrize("module", MODULES)
def test_every_destination_reuse_is_guarded(module):
    text = (SRC / module).read_text()
    if _REUSE.search(text):
        assert "is_compatible_as_left_input" in text, module
    else:
        # Only empty, self-filled input tensors: nothing to guard.
        assert _EMPTY.search(text), module


@pytest.mark.parametrize("name", sorted(_KERNELS))
def test_register_p_reuse_has_a_threadgroup_fallback(name):
    source = _KERNELS[name]()
    guard = "const bool p_in_registers = opO.template is_compatible_as_left_input<half, bfloat, float>(Pc);"
    assert source.count(guard) == 1, name
    # The conversion happens only on the compatible arm.
    reuse = "auto Pin = opO.template get_left_input_cooperative_tensor<half, bfloat, float>(Pc);"
    assert source.count(reuse) == 1, name
    assert source.index(guard) < source.index("    if (p_in_registers) {\n" + "      " + reuse)
    # Incompatible arm: P is stored to threadgroup memory and run from tP.
    store = source.index("    if (!p_in_registers) {\n")
    assert "myP[fm * TK + f * 16 + fn + i] = half(p[f * 8 + i]);" in source[store:]
    assert "opO.run(tP, bVlo, Olo);" in source
    assert source.count("simdgroup_barrier(mem_flags::mem_threadgroup);") == 2
    if name.endswith("_128"):
        assert "Ohi" not in source and "bVhi" not in source
    else:
        assert "opO.run(Pin, bVhi, Ohi);" in source and "opO.run(tP, bVhi, Ohi);" in source


@pytest.mark.parametrize("name", sorted(_KERNELS))
def test_compatible_arm_runs_the_previous_register_ops(name):
    """With the guard true the kernel issues the pre-guard register ops."""
    source = _KERNELS[name]()

    def register_arm(text):
        # Drop the incompatible arms and the guard, keep the compatible arm.
        text = re.sub(r"    if \(!p_in_registers\) \{\n.*?\n    \}\n", "", text, flags=re.S)
        text = re.sub(r"    \} else \{\n.*?\n    \}", "", text, flags=re.S)
        text = text.replace(
            "    // Register P needs the op's left-input layout to accept Pc's layout\n"
            "    // (MPP is_compatible_as_left_input); otherwise P goes through threadgroup\n"
            "    // memory with the same bits.\n", ""
        )
        text = re.sub(r"    const bool p_in_registers = [^\n]*\n", "", text)
        text = text.replace("    if (p_in_registers) {\n", "")
        return [line.strip() for line in text.splitlines() if line.strip()]

    previous = {
        "lane_attention.partial_direct": la._PARTIAL.replace(la._P_STORE, la._P_REGISTERS)
        .replace(la._PV_THREADGROUP, la._PV_REGISTERS),
        "lane_attention.partial_direct_128": la._single_half(
            la._PARTIAL.replace(la._P_STORE, la._P_REGISTERS)
            .replace(la._PV_THREADGROUP, la._PV_REGISTERS)
        ),
        "stream_attention.partial": sa._PARTIAL_REGISTERS,
    }[name]
    old = [line.strip() for line in previous.splitlines() if line.strip()]
    new = register_arm(source)
    # Same statements; the only move is the conversion, which now sits right
    # before its first use instead of before the independent O rescale loop.
    pin = "auto Pin = opO.template get_left_input_cooperative_tensor<half, bfloat, float>(Pc);"
    assert sorted(old) == sorted(new)
    assert [line for line in old if line != pin] == [line for line in new if line != pin]
    assert new.index(pin) == new.index("opO.run(Pin, bVlo, Olo);") - 1


def test_threadgroup_and_other_kernels_are_byte_identical():
    assert _sha(la._PARTIAL) == "514e31136acdf792e514b7b4cdc639fb2460f114b07dfca06af5f3e2681cfadb"
    assert _sha(la._PARTIAL_128) == "5cdc10dc76c20e031aaa8850caa802a7a1cbfd9808a391fba7d1cb89999996ba"
    assert _sha(la._MERGE) == "28235b8bb8b45b908fe08018ec884ae1f1a7e66cd3096a6ecff62517df5db7df"
    assert _sha(sa.sources()["tail"]) == "7b50c9a234bcf908e09748a51a69d13061e20bc9813e54106101eca8d7ca9460"
    assert _sha(sa.sources()["merge"]) == "82d19ceffef5aeb6f7e8a1e2d75fbb1161328f1261b82ed1ce9abbf9864eba33"
    # The pre-guard register kernel text is kept verbatim for review.
    assert _sha(sa._PARTIAL_REGISTERS) == "20c2be85d844eeb85c301a55c3a3b432810a612c01ca5abc6fe7796055fb260f"
