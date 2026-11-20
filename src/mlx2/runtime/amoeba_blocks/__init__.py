"""Gated block-sparse MLP/MoE kernels for aMoEba serving (experimental, opt-in, unqualified).

A tiled aMoEba model runs every MLP (or routed expert) as ``core + gated tiles``.  Masking closed
tiles saves nothing; these kernels skip them: a closed tile's gate/up rows and every original
``down`` quantization group that holds only closed neurons are neither read nor computed.

* :mod:`.layout`    offline packer (TensorFold-style row packing of gate/up, core first, tiles
                    contiguous) and block tables; tile graph with the ancestor closure folded in.
* :mod:`.kernels`   Metal sources: mpp (M5 tensor units, NAX) and simd (M1-M4) gate/up+SwiGLU
                    and down, and the fused tile-gate kernel; offline compilation for CPU checks.
* :mod:`.ops`       MLX launchers, work tiles (dense, decode pairs, sorted prefill routes).
* :mod:`.reference` float64 masked reference and the fp32 structural emulator (CPU tests).

States: implemented; not qualified; not selected by any route; never observed-used in serving.
See docs/research/2026-11-18-amoeba-block-sparse-kernels.md and provenance/amoeba-blocks.json.
"""

STATUS = {"implemented": True, "qualified": False, "selected": False, "observed_used": False}
