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
process and dtype in both spellings, compares each with the served op bit for
bit over every 16-bit encoding (float32: a widened-bf16 plus random-bits
sample; NaN matches NaN), and refuses the kernel
unless its own spelling reproduced the served op. The kernels are never
rewritten to the other spelling; that would need its own GPU qualification.
A failing probe only turns a fused path off.
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


def same_bits(a: mx.array, b: mx.array) -> bool:
    from .qwen4_fused_gdn import same_bits as equal

    return equal(a, b)


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


class ServedExpGate:
    """Refuses a fused kernel whose sigmoid spelling the served op does not use.

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
                    logger.info("Served exp probe %s failed: %s", self.name, exc)
                    found = None
                self._spellings[key] = found
            return self._spellings[key]

    def refusal(self, dtype=mx.bfloat16) -> Optional[str]:
        """Why the kernel may not stand in for the served op, or None."""
        spelling = self.spelling(dtype)
        if spelling == self.kernel_exp:
            return None
        if spelling is None:
            return f"served {self.served_name} matches no kernel form"
        return f"served {self.served_name} uses {spelling}"

    def reset(self) -> None:
        with self._lock:
            self._spellings.clear()
