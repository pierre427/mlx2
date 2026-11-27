# SPDX-License-Identifier: MIT
"""Source-only WoA PP99 BF16 numerical profile; default selection is external.

Original BF16 resident coefficients and Source BF16-owned producer values are
unchanged. The reduction program differs; native pre-BF16 F32 is unexposed.
No GPU work/import-time probing; see source-woa-native-pp provenance.
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import importlib.metadata
import re

POLICY='native_bf16_grouped_pp_v1'
LIBRARY_SHA256='d8c9393d584e781393c3effccfccd8ba85ad0c70264edaf9f897dca9875375c6'
HEADER_PINS={
    'steel/gemm/mma.h':'9b37709de40c3acaa5411875e52c8d26e6de093261741f671fbdab9c39a852b7',
    'steel/gemm/nax.h':'fb10bcae44095e9fed5c81213d03527dd5fe5ac11be11ec7784bd357c6f317ec',
    'steel/gemm/gemm_nax.h':'8f8d11f3c29e30dbfcaa6799963edd5fb71b81f912c4f3494e294d3002e3d337',
}
# Per-build (libmlx sha256, header pins).  0.32.5 re-audit (2026-11-27,
# v0.32.4..v0.32.5): the route is one stock BF16 batched einsum (M99, N1024,
# K4096 per group) on the steel NAX GEMM.  steel/gemm/nax.h and gemm_nax.h are
# byte-identical; mma.h only gains a ``TB = T`` template parameter on
# BlockMMA::mma (BaseMMAFrag unchanged); einsum.cpp is unchanged and matmul.cpp
# changes only the transposed gemv_t tiles (mlx #4850, a vector path this M99
# GEMM never takes) and empty GatherMM/SegmentedMM outputs.  transforms.h's
# TransformAdd/Axpby change reaches only addmm epilogues, not this einsum.
# libmlx/metallib are rebuilt, so the stock numerical profile stays a GPU-gate
# claim (provenance source-woa-native-pp), not a CPU one.
PINS_BY_VERSION={
    '0.32.4':(LIBRARY_SHA256,HEADER_PINS),
    '0.32.5':('0b8158bf29ecac60049f9163dda9537aab10d6fd8aa3d7614c646c3e6f810750',{
        **HEADER_PINS,
        'steel/gemm/mma.h':'21439be1087ac840ac5936bda0e57bd982fb7b104ea5e5f82ba7be0d2eaee524',
    }),
}


def configuration_supported(args):
    return all(getattr(args,k,None)==v for k,v in(
        ('activation_dtype','bfloat16'),('head_dim',512),('num_attention_heads',64),('o_groups',8),('o_lora_rank',1024)))


def owning_alias(name):
    if type(name)is not str:return False
    match=re.fullmatch(r'blk\.(0|[1-9][0-9]?)\.attn_output_a\.weight',name)
    return match is not None and int(match[1])<40


@lru_cache(maxsize=1)
def implementation_identity():
    import mlx.core as mx
    pinned=PINS_BY_VERSION.get(importlib.metadata.version('mlx'))
    if pinned is None:raise RuntimeError('Source native WoA PP MLX version changed; re-audit required')
    library_sha256,header_pins=pinned
    root=Path(mx.__file__).resolve().parent
    if hashlib.sha256((root/'lib/libmlx.dylib').read_bytes()).hexdigest()!=library_sha256:raise RuntimeError('Source native WoA PP MLX build changed; re-audit required')
    for name,expected in header_pins.items():
        if hashlib.sha256((root/'include/mlx/backend/metal/kernels'/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('Source native WoA PP header changed; re-audit required')
    own=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return hashlib.sha256((POLICY+':'+library_sha256+':'+own+':'+repr(sorted(header_pins.items()))).encode()).hexdigest()


def eligible(name,x,weight,*,groups=8,check_device=True):
    import mlx.core as mx
    if not owning_alias(name):return False
    supported=(type(groups)is int and groups==8 and isinstance(x,mx.array)and x.shape==(1,99,8,4096)and x.dtype==mx.float32
        and isinstance(weight,mx.array)and weight.shape==(8192,4096)and weight.dtype==mx.bfloat16)
    if not supported:return False
    if not check_device:return True
    from ..runtime.gguf.source_dense_legacy import _device_supported
    return bool(_device_supported())


def maybe_project(name,x,weight,*,groups=8,counters):
    """Return None only for refusal; device/build faults propagate unchanged."""
    import mlx.core as mx
    if not owning_alias(name):return None
    if not eligible(name,x,weight,groups=groups):
        key='source_woa_native_pp_reference_fallback';counters[key]=counters.get(key,0)+1
        return None
    implementation_identity()
    value=mx.einsum('btgi,goi->btgo',x.astype(mx.bfloat16),weight.reshape(8,1024,4096))
    if value.dtype!=mx.bfloat16:raise RuntimeError('Source native WoA PP expected BF16 stock output')
    key='source_woa_native_pp';counters[key]=counters.get(key,0)+1
    return value.astype(mx.float32)
