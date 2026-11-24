"""Offline compile check for ``mx.fast.metal_kernel`` sources (no GPU).

Rebuilds the translation unit ``mx.fast.metal_kernel`` would build (MLX
0.32's ``write_signature``: the header, a templated ``[[kernel]]`` signature
with inputs under 8 elements in the ``constant`` address space, the Metal
attributes the source names, the body, and the explicit instantiation) and
runs the offline Metal compiler (``xcrun -sdk macosx metal -c``) on it. This
compiles to AIR on the CPU; nothing runs on the GPU. It catches syntax, type
and overload errors in generated sources before the GPU lease does.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import mlx.core as mx

TYPES = {mx.bfloat16: "bfloat16_t", mx.float16: "float16_t", mx.float32: "float",
         mx.int32: "int32_t", mx.uint32: "uint32_t", mx.bool_: "bool"}
ATTRIBUTES = (
    ("simdgroup_index_in_threadgroup", "uint"), ("thread_index_in_simdgroup", "uint"),
    ("thread_index_in_threadgroup", "uint"), ("thread_position_in_grid", "uint3"),
    ("thread_position_in_threadgroup", "uint3"), ("threadgroup_position_in_grid", "uint3"),
    ("threadgroups_per_grid", "uint3"), ("threads_per_threadgroup", "uint3"),
)


def compiler():
    if shutil.which("xcrun") is None:
        return None
    probe = subprocess.run(["xcrun", "-sdk", "macosx", "-f", "metal"], capture_output=True,
                           text=True)
    if probe.returncode != 0:
        return None
    return probe.stdout.strip() or None


def translation_unit(name, inputs, outputs, source, header="", template=()):
    """``inputs``: ``[(name, dtype, n_elements)]``; ``outputs``: ``[(name, dtype)]``;
    ``template``: ``[(name, value)]`` with dtype, int or bool values."""
    lines = ['#include <metal_stdlib>', '#include "mlx/backend/metal/kernels/utils.h"',
             header]
    params, args = [], []
    for key, value in template:
        if isinstance(value, mx.Dtype):
            params.append(f"typename {key}")
            args.append(TYPES[value])
        elif isinstance(value, bool):
            params.append(f"bool {key}")
            args.append(str(int(value)))
        else:
            params.append(f"int {key}")
            args.append(str(int(value)))
    func = f"custom_kernel_{name}"
    signature = []
    index = 0
    for key, dtype, size in inputs:
        space = "constant" if size < 8 else "device"
        signature.append(f"  const {space} {TYPES[dtype]}* {key} [[buffer({index})]]")
        index += 1
    for key, dtype in outputs:
        signature.append(f"  device {TYPES[dtype]}* {key} [[buffer({index})]]")
        index += 1
    for attr, kind in ATTRIBUTES:
        if attr in source:
            signature.append(f"  {kind} {attr} [[{attr}]]")
    if params:
        lines.append(f"template <{', '.join(params)}>")
    lines.append(f"[[kernel]] void {func}(\n" + ",\n".join(signature) + ") {\n" + source + "\n}\n")
    if params:
        inst = f"{func}<{', '.join(args)}>"
        lines.append(f'template [[host_name("{func}_i")]] [[kernel]] decltype({inst}) {inst};')
    return "\n".join(lines)


def compile_unit(text, std="metal3.1", flags=()):
    """``None`` when it compiles, else the compiler's error text. ``std`` and
    ``flags`` let a test mirror MLX's JIT options (e.g. ``metal4.1`` and
    ``-fmetal-math-mode=safe``)."""
    root = Path(mx.__file__).resolve().parent / "include"
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "unit.metal"
        src.write_text(text)
        result = subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", f"-std={std}", *flags, "-I", str(root), "-c", str(src),
             "-o", str(Path(tmp) / "unit.air")], capture_output=True, text=True)
    if result.returncode == 0:
        return None
    return result.stderr[-4000:]
