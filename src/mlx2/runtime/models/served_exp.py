# SPDX-License-Identifier: MIT
"""Served-graph gates for fused kernels that copy one MLX sigmoid spelling.

Follow-up to the fused GDN gate (``qwen4_fused_gdn.served_silu_refusal``,
idea from jundot/omlx PR #4122). Several other fused kernels reproduce an MLX
sigmoid by copying its source spelling: the routed-decode SwiGLU and the HC
decode SiLU copy the JIT-compiled ``nn.silu`` (``metal::exp``), the QSA output
gate epilogue copies the eager ``mx.sigmoid`` unary (``metal::exp`` plus one
patched bf16 edge), and the shared-expert fold and the HC decode sigmoid copy
the eager unary in ``metal::precise::exp``. Which ``exp`` the served op
resolves to depends on how the MLX build compiles it (MLX #4461 moved the
sigmoid source to ``metal::precise::exp``; the prebuilt unary library is built
with -fno-fast-math), so a different build can silently break bit-exactness.

A :class:`ServedExpGate` compiles the kernel's own sigmoid expression once per
process and dtype in both spellings and compares each with the served op bit
for bit over every 16-bit encoding (float32: a widened-bf16 plus random-bits
sample; NaN matches NaN).  Callers may then compile their source with the
matching spelling.  A failing probe still turns the optional fused path off.
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Callable, Optional

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

EXP_SPELLINGS = ("metal::exp", "metal::precise::exp")

_SOURCE = """
  uint i = thread_position_in_grid.x;
  const T x = inp[i];
  T y;
  {
__BODY__
  }
  out[i] = y;
"""


def all_16bit_encodings(dtype) -> mx.array:
    from .qwen4_fused_gdn import all_16bit_encodings as encodings

    return encodings(dtype)


def probe_inputs(dtype) -> mx.array:
    """Every encoding of a 16-bit type; for float32 every bf16 value widened
    plus 2**20 fixed-seed random bit patterns (a sample, not exhaustive)."""
    if dtype in (mx.bfloat16, mx.float16):
        return all_16bit_encodings(dtype)
    if dtype == mx.float32:
        widened = all_16bit_encodings(mx.bfloat16).astype(mx.float32)
        bits = np.random.default_rng(4122).integers(0, 1 << 32, 1 << 20, dtype=np.uint32)
        sample = mx.array(bits).view(mx.float32)
        return mx.concatenate([widened, sample])
    raise ValueError(f"no served-exp probe inputs for {dtype}")


def is_device_fault(exc) -> bool:
    """A Metal out-of-memory or GPU-timeout step failure (serving recovers
    those by abandoning the lanes in the failed buffer)."""
    from ...serving import device_fault_kind

    return device_fault_kind(exc) is not None


def is_threadgroup_refusal(exc) -> bool:
    """MLX's ``metal_kernel`` refusing a threadgroup above the compiled
    pipeline's ``maxTotalThreadsPerThreadgroup`` (MLX 39400a0d4
    ``custom_kernel.cpp`` raises ``std::invalid_argument``, a Python
    ``ValueError``).  A kernel refusal, not a contract error: register-heavy
    pipelines on M1-M4 can allow fewer than 1024 threads (TensorFold
    #704/#715 measured 768-832)."""
    return isinstance(exc, ValueError) and (
        "maximum allowed threads per threadgroup" in str(exc)
    )


def same_bits(a: mx.array, b: mx.array) -> bool:
    from .qwen4_fused_gdn import same_bits as equal

    return equal(a, b)


def covering_rows(grid: mx.array, rows: int, width: int, seed: int) -> mx.array:
    """``[rows, width]`` copies of ``grid``'s values in seeded orders, every value present.

    When ``width`` holds the whole grid, each row covers it (row ``r`` is
    shuffled with key ``seed + r``); otherwise the rows cover it together
    (one shuffle of the flattened block, key ``seed``).  The tile is cut to
    size before it is shuffled: shuffling a longer tile and then truncating
    it repeats some values and drops others.
    """
    n = int(grid.size)
    if width >= n:
        base = mx.tile(grid, (-(-width // n),))[:width]
        return mx.stack([
            base[mx.random.permutation(width, key=mx.random.key(seed + r))] for r in range(rows)
        ])
    total = rows * width
    if total < n:
        raise ValueError(f"{rows} rows of {width} cannot cover {n} values")
    flat = mx.tile(grid, (-(-total // n),))[:total]
    return flat[mx.random.permutation(total, key=mx.random.key(seed))].reshape(rows, width)


def respell(text: str, kernel_exp: str, spelling: str) -> str:
    """``text`` with its ``kernel_exp`` calls spelled ``spelling``.

    ``metal::exp`` is not a substring of ``metal::precise::exp``, so either
    direction rewrites only the kernel's own spelling.
    """
    if kernel_exp not in EXP_SPELLINGS or spelling not in EXP_SPELLINGS:
        raise ValueError(f"unknown exp spelling {kernel_exp!r} -> {spelling!r}")
    if kernel_exp not in text:
        raise ValueError(f"probe text does not spell {kernel_exp}")
    return text.replace(kernel_exp, spelling)


def metal_helper(text: str, name: str) -> str:
    """One ``template <typename U> inline U name(U x) {...}`` helper of ``text``."""
    start = text.index(f"template <typename U>\ninline U {name}(")
    end = text.index("\n}\n", start) + 3
    return text[start:end]


def adapted_helper_header(gate, header: str, helper_name: str, dtype=mx.bfloat16) -> tuple[str, str]:
    """``header`` with only ``helper_name`` re-spelled to the served form.

    Other helpers in the same header (softplus, log1p) keep their own
    spelling.  Returns the header and a kernel-name suffix for the spelling.
    When no spelling matches, the header is returned unchanged: the gate's
    refusal keeps the gated use off, while ungated uses of the same kernel
    (which never call the helper) still compile.
    """
    spelling = gate.spelling(dtype)
    if spelling is None:
        return header, ""
    helper = metal_helper(header, helper_name)
    suffix = "_precise" if "precise" in spelling else "_fast"
    return header.replace(helper, gate.adapt(helper, dtype)), suffix


def kernel_dtype_refusal(gate, dtype, kernel_dtype=mx.bfloat16) -> Optional[str]:
    """Refuse ``dtype`` when no spelling reproduces it, or when it differs from
    the spelling the kernel was compiled with for ``kernel_dtype``."""
    refusal = gate.adaptation_refusal(dtype)
    if refusal is None and gate.spelling(dtype) != gate.spelling(kernel_dtype):
        refusal = f"served {gate.served_name} spelling differs between {dtype} and {kernel_dtype}"
    return refusal


class ServedExpGate:
    """Select the sigmoid spelling that reproduces a served operation.

    ``header``/``body`` are the kernel's own Metal text (``body`` reads ``x``
    of type ``T`` and assigns ``y``); ``served`` is the op the kernel stands in
    for. One probe per process and dtype; a probe failure refuses.
    """

    def __init__(
        self,
        name: str,
        *,
        served_name: str,
        served: Callable[[mx.array], mx.array],
        body: str,
        header: str = "",
        kernel_exp: str = "metal::exp",
    ):
        if kernel_exp not in EXP_SPELLINGS:
            raise ValueError(f"unknown exp spelling {kernel_exp!r}")
        if kernel_exp not in header + body:
            raise ValueError(f"{name}: probe text does not spell {kernel_exp}")
        self.name = name
        self.served_name = served_name
        self.served = served
        self.body = body
        self.header = header
        self.kernel_exp = kernel_exp
        self._spellings: dict = {}
        self._lock = Lock()

    def candidates(self) -> tuple:
        """The kernel's own spelling first, then the other one."""
        return (self.kernel_exp,) + tuple(s for s in EXP_SPELLINGS if s != self.kernel_exp)

    def probe(self, dtype) -> dict:
        """``{spelling: bool}``: which forms reproduce the served op (Metal)."""
        x = probe_inputs(dtype)
        served = self.served(x)
        matches = {}
        for spelling in self.candidates():
            kernel = mx.fast.metal_kernel(
                name=f"mlx2_served_exp_{self.name}_{'precise' if 'precise' in spelling else 'fast'}",
                input_names=["inp"],
                output_names=["out"],
                header=respell(self.header, self.kernel_exp, spelling)
                if self.kernel_exp in self.header else self.header,
                source=_SOURCE.replace(
                    "__BODY__",
                    respell(self.body, self.kernel_exp, spelling)
                    if self.kernel_exp in self.body else self.body,
                ),
                ensure_row_contiguous=True,
            )
            (y,) = kernel(
                inputs=[x],
                template=[("T", dtype)],
                grid=(x.size, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[x.shape],
                output_dtypes=[dtype],
            )
            matches[spelling] = same_bits(y, served)
        return matches

    def select(self, matches) -> Optional[str]:
        return next((s for s in self.candidates() if matches.get(s)), None)

    def spelling(self, dtype=mx.bfloat16) -> Optional[str]:
        """The served op's spelling for ``dtype``; None if none matches or the
        probe failed. Probed once per process and dtype."""
        key = str(dtype)
        if key in self._spellings:
            return self._spellings[key]
        with self._lock:
            if key not in self._spellings:
                try:
                    found = self.select(self.probe(dtype))
                except Exception as exc:  # noqa: BLE001 - fail closed
                    if is_device_fault(exc):
                        # The probe shared a command buffer with the step: let
                        # serving's device-fault recovery see it, and probe
                        # again next time rather than refusing for good.
                        raise
                    logger.info("Served exp probe %s failed: %s", self.name, exc)
                    found = None
                self._spellings[key] = found
            return self._spellings[key]

    def refusal(self, dtype=mx.bfloat16) -> Optional[str]:
        """Why the kernel's original form may not stand in for the served op."""
        spelling = self.spelling(dtype)
        if spelling == self.kernel_exp:
            return None
        if spelling is None:
            return f"served {self.served_name} matches no kernel form"
        return f"served {self.served_name} uses {spelling}"

    def adaptation_refusal(self, dtype=mx.bfloat16) -> Optional[str]:
        """Why no supported source form can stand in for the served op."""
        if self.spelling(dtype) is None:
            return f"served {self.served_name} matches no kernel form"
        return None

    def adapt(self, text: str, dtype=mx.bfloat16) -> str:
        """Rewrite this gate's expression to the probed served spelling."""
        spelling = self.spelling(dtype)
        if spelling is None:
            raise RuntimeError(f"served {self.served_name} matches no kernel form")
        if spelling == self.kernel_exp:
            return text
        return respell(text, self.kernel_exp, spelling)

    def reset(self) -> None:
        with self._lock:
            self._spellings.clear()
