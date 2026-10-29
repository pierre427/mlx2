# SPDX-License-Identifier: MIT
"""Opt-in Source HC coefficient candidate, retaining raw FP32 boundaries.

Projection and RMS scaling remain in the Source model. This component fuses
only 24 mixes -> pre4/post4/comb4x4. No model or shared-family wiring occurs
here. Exact stock-GPU parity remains a separate required gate. See
provenance/deepseek-v41-source-hc.json for installed-source pins and license.
"""
from functools import lru_cache
import hashlib
import importlib.metadata
import math
from pathlib import Path


REDUCTIONS = ("sequential", "column_pairwise", "column_interleaved", "row_pairwise", "both_pairwise")
SOFTMAX_EXPS = ("metal::fast::exp", "metal::exp", "metal::precise::exp")
_SUM_MODES = {"sequential": (0, 0), "column_pairwise": (0, 1),
              "column_interleaved": (0, 2), "row_pairwise": (1, 0), "both_pairwise": (1, 1)}
INSTALLED_VERSION = "0.32.4"
INSTALLED_PINS = {
    "include/mlx/backend/metal/kernels/unary_ops.h": "af710925f066b029f67289a0bbe4edd0ba6c5e7f51c9df609abb111216fa963f",
    "include/mlx/backend/metal/kernels/softmax.h": "cb233fe42a7358f5f658c9a1ef52cd2ee9dd3786ddfad01793a976d58f4d4051",
    "include/mlx/backend/metal/kernels/defines.h": "a6c2c824e855078cd092fd9592457616f995755f0e865475c3f006a3f5d6bf1d",
    "include/mlx/backend/metal/kernels/reduction/reduce_row.h": "8045c77cb0dde4f97a77ddcdcfad825ccf855928d31369a1dc1832629a63da50",
    "include/mlx/backend/metal/kernels/reduction/reduce_col.h": "47d47a9a8b52e3dfd15b195f40cdb3623583e00f5ac11378cdc0a7fe49fe17cf",
    "include/mlx/backend/metal/kernels/reduction/ops.h": "78d06730fc9564a73944e7f1fe3897d25c8789b28a939bf418e1968db311da41",
    "include/mlx/backend/metal/kernels/binary_ops.h": "6dbe789faafbf7259f411c7fab7c0fcdd192df47a78f8f0bd408228f7a30e5c7",
    "lib/libmlx.dylib": "d8c9393d584e781393c3effccfccd8ba85ad0c70264edaf9f897dca9875375c6",
    "lib/mlx.metallib": "c6d7a3facb83568f6e4cb2cf4a8c736ee29302571a48343cb9a3a4d1cd3e3893",
}
_SCHEDULED = {}


def _package_root():
    import mlx.core as mx
    return Path(mx.__file__).parent


@lru_cache(maxsize=1)
def _verify_installed():
    if importlib.metadata.version("mlx") != INSTALLED_VERSION:
        raise ValueError("Source HC installed MLX version differs from pinned candidate")
    root = _package_root()
    for relative, expected in INSTALLED_PINS.items():
        try:
            digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        except OSError as error:
            raise ValueError("Source HC installed source missing: " + relative) from error
        if digest != expected:
            raise ValueError("Source HC installed source changed: " + relative)
    return tuple(INSTALLED_PINS.items())


def installed_receipt():
    """CPU-only source validation; no Metal work or parity assertion."""
    return {"version": INSTALLED_VERSION, "files_sha256": dict(_verify_installed())}


@lru_cache(maxsize=1)
def _sigmoid_gate():
    import mlx.core as mx
    from .served_exp import ServedExpGate
    return ServedExpGate(
        "v41_source_hc_sigmoid", served_name="eager mx.sigmoid(float32)", served=mx.sigmoid,
        body="float z=1.0f/(1.0f+metal::exp(metal::abs(x))); y=x<0.0f ? z : 1.0f-z;",
    )


def _device():
    import mlx.core as mx
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise ValueError("Source HC requires a caller-owned Metal GPU window")
    return mx


def sigmoid_spelling():
    mx = _device()
    spelling = _sigmoid_gate().spelling(mx.float32)
    if spelling is None:
        raise ValueError("Source HC cannot reproduce served eager F32 sigmoid")
    return spelling


def plan(mixes, scale, base, *, iterations=20, eps=1e-6,
         reduction="sequential", softmax_exp="metal::fast::exp"):
    import mlx.core as mx
    if (mixes.ndim not in (2, 3) or mixes.shape[-1] != 24
            or (mixes.ndim == 3 and mixes.shape[0] != 1)
            or not all(mixes.shape) or scale.shape != (3,) or base.shape != (24,)
            or any(x.dtype != mx.float32 for x in (mixes, scale, base))):
        raise ValueError("Source HC requires FP32 B1 mixes24, scale3 and base24")
    rows = math.prod(mixes.shape[:-1])
    if not 1 <= rows <= 1000:
        raise ValueError("Source HC rows must be in1..1000")
    if type(iterations) is not int or not 1 <= iterations <= 100:
        raise ValueError("Source HC iterations must be in1..100")
    if (isinstance(eps, bool) or not isinstance(eps, (int, float))
            or not math.isfinite(eps) or eps <= 0):
        raise ValueError("Source HC epsilon must be positive finite")
    if reduction not in REDUCTIONS or softmax_exp not in SOFTMAX_EXPS:
        raise ValueError("Source HC unknown explicit reduction or softmax-exp candidate")
    return rows


HEADER = r'''
#pragma clang fp contract(off)
inline float source_hc_sum4(float a, float b, float c, float d, int mode) {
    if(mode==1) {
        volatile float left=a+b, right=c+d;
        return left+right;
    }
    if(mode==2) {
        volatile float left=a+c, right=b+d;
        return left+right;
    }
    volatile float total=0.0f;
    total=a+total; total=b+total; total=c+total; total=d+total;
    return total;
}
inline float source_hc_sigmoid(float x) {
    const float z=1.0f/(1.0f+SIGMOID_EXP(metal::abs(x)));
    return x<0.0f ? z : 1.0f-z;
}
'''
SOURCE = r'''
const uint row=thread_position_in_grid.x;
if(row>=ROWS)return;
const float epsilon=eps_value[0];
float c[16];
for(uint i=0;i<4;i++) {
    const float pre_product=mixes[row*24+i]*scale[0];
    const float post_product=mixes[row*24+4+i]*scale[1];
    const float pre_logit=pre_product+base[i];
    const float post_logit=post_product+base[4+i];
    out[row*24+i]=source_hc_sigmoid(pre_logit)+epsilon;
    out[row*24+4+i]=2.0f*source_hc_sigmoid(post_logit);
}
for(uint r=0;r<4;r++) {
    float maximum=-FLT_MAX;
    for(uint s=0;s<4;s++) {
        const uint i=r*4+s;
        const float product=mixes[row*24+8+i]*scale[2];
        c[i]=product+base[8+i];
        maximum=maximum<c[i] ? c[i] : maximum;
    }
    float values[4];
    for(uint s=0;s<4;s++)values[s]=SOFTMAX_EXP(c[r*4+s]-maximum);
    // Installed softmax width4 assigns all four elements to one lane, sums
    // them in order, then multiplies by the reciprocal normalizer.
    const float total=source_hc_sum4(values[0],values[1],values[2],values[3],0);
    const float inverse=1.0f/total;
    for(uint s=0;s<4;s++)c[r*4+s]=values[s]*inverse+epsilon;
}
for(uint s=0;s<4;s++) {
    const float sum=source_hc_sum4(c[s],c[4+s],c[8+s],c[12+s],COL_SUM);
    const float denominator=sum+epsilon;
    for(uint r=0;r<4;r++)c[r*4+s]=c[r*4+s]/denominator;
}
for(uint iteration=1;iteration<ITERS;iteration++) {
    for(uint r=0;r<4;r++) {
        const uint i=r*4;
        const float sum=source_hc_sum4(c[i],c[i+1],c[i+2],c[i+3],ROW_SUM);
        const float denominator=sum+epsilon;
        for(uint s=0;s<4;s++)c[i+s]=c[i+s]/denominator;
    }
    for(uint s=0;s<4;s++) {
        const float sum=source_hc_sum4(c[s],c[4+s],c[8+s],c[12+s],COL_SUM);
        const float denominator=sum+epsilon;
        for(uint r=0;r<4;r++)c[r*4+s]=c[r*4+s]/denominator;
    }
}
for(uint i=0;i<16;i++)out[row*24+8+i]=c[i];
'''


def compilation_source(*, sigmoid_exp="metal::precise::exp", reduction="sequential",
                       softmax_exp="metal::fast::exp"):
    if sigmoid_exp not in ("metal::exp", "metal::precise::exp"):
        raise ValueError("Source HC unknown sigmoid spelling")
    if reduction not in REDUCTIONS or softmax_exp not in SOFTMAX_EXPS:
        raise ValueError("Source HC unknown explicit candidate")
    r, c = _SUM_MODES[reduction]
    return (HEADER.replace("SIGMOID_EXP", sigmoid_exp),
            SOURCE.replace("SOFTMAX_EXP", softmax_exp).replace("ROW_SUM", str(r)).replace("COL_SUM", str(c)))


@lru_cache(maxsize=30)
def _kernel(sigmoid_exp, reduction, softmax_exp):
    import mlx.core as mx
    header, source = compilation_source(sigmoid_exp=sigmoid_exp, reduction=reduction, softmax_exp=softmax_exp)
    tag = "_".join((sigmoid_exp, reduction, softmax_exp)).replace("::", "_")
    return mx.fast.metal_kernel(name="mlx2_v41_source_hc_"+tag,
        input_names=["mixes", "scale", "base", "eps_value"], output_names=["out"],
        header=header, source=source, ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"})


def split(mixes, scale, base, *, iterations=20, eps=1e-6,
          reduction="sequential", softmax_exp="metal::fast::exp"):
    """Schedule one candidate kernel; exact stock parity is not assumed.

    No CPU fallback, source projection change or BF16 round is permitted.
    Nonfinite arithmetic is outside this candidate's finite-input oracle.
    """
    rows = plan(mixes, scale, base, iterations=iterations, eps=eps,
                reduction=reduction, softmax_exp=softmax_exp)
    mx = _device()
    _verify_installed()
    spelling = sigmoid_spelling()
    value = _kernel(spelling, reduction, softmax_exp)(
        inputs=[mixes, scale, base, mx.array([eps], mx.float32)],
        template=[("ROWS", rows), ("ITERS", iterations)],
        grid=(rows, 1, 1), threadgroup=(min(32, rows), 1, 1),
        output_shapes=[mixes.shape], output_dtypes=[mx.float32])[0]
    tag = reduction+"/"+softmax_exp+"/"+spelling
    _SCHEDULED[tag] = _SCHEDULED.get(tag, 0)+1
    return value[..., :4], value[..., 4:8], value[..., 8:].reshape(*mixes.shape[:-1], 4, 4)


def route_status():
    return {"implemented": True, "qualified": False, "selected_by_default": False,
            "scheduled_calls": dict(_SCHEDULED), "gpu_completion_not_implied": True,
            "projection_and_rms": "unchanged external FP32 Source graph"}
