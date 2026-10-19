"""Static guard: MPP destination tensors take address-space-free operand types.

macOS 27.0.1's MetalPerformancePrimitives ``get_destination_cooperative_tensor``
only admits operand types that are tensors or *unqualified* cooperative
tensors.  ``decltype`` of a local cooperative tensor is ``thread``-qualified,
so a kernel that spells ``decltype(ct)`` no longer compiles ("Unable to build
metal library from source"); on the M5 this killed the server on the first
int8 Q8/Q4/Q5 in-place prefill and on Flash-Next's QSA stage-1 MPP score
producers (2026-10-18).  The fix strips the address space with
``metal::remove_addrspace_t``, as ``qwen4_qsa_nax`` already did.  Tensor
slices (``tA.slice(...)``) are tensor types and stay valid as they are.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_INPUT = re.compile(
    r"auto\s+(\w+)\s*=\s*\w+\s*\.\s*(?:template\s+)?"
    r"get_(?:left|right)_input_cooperative_tensor"
)


def _kernel_files():
    for path in sorted((ROOT / "src").rglob("*.py")):
        text = path.read_text(errors="replace")
        if "_input_cooperative_tensor" in text:
            yield path, text.replace("\\n", "\n")


def test_cooperative_operand_decltypes_strip_the_address_space():
    offenders = []
    files = 0
    for path, text in _kernel_files():
        files += 1
        for name in set(_INPUT.findall(text)):
            bare = re.findall(rf"(?<!remove_addrspace_t<)decltype\({name}\)", text)
            if bare:
                offenders.append((str(path.relative_to(ROOT)), name, len(bare)))
    assert files, "no MPP cooperative-tensor kernels found; the guard scans nothing"
    assert offenders == []


def test_guard_detects_a_thread_qualified_operand():
    sample = (
        "auto ct_a = op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();\n"
        "auto acc = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), int32_t>();\n"
    )
    names = set(_INPUT.findall(sample))
    assert names == {"ct_a"}
    assert re.findall(r"(?<!remove_addrspace_t<)decltype\(ct_a\)", sample)
