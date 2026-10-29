# SPDX-License-Identifier: MIT
"""Explicit Source PP numerical profile, using unchanged installed stock QMM.

See provenance/deepseek-v41-source-fp8-native-pp.json. The native BF16 output
is not a raw F32 accumulator; the model's existing owner boundary still runs.
No GPU work occurs on import. No new kernel or coefficient format is installed.
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import importlib.metadata
import re

POLICY='native_bf16_qb_wob_v1'
LIBRARY_SHA256='d8c9393d584e781393c3effccfccd8ba85ad0c70264edaf9f897dca9875375c6'
HEADER_PINS={
    'fp_quantized.h':'fb4bf0919642717c5e36cfa7be0628e651844586cae1b405c27b2057a17dd1c8',
    'fp_quantized_nax.h':'8a712870dc17493ddf05daa706481d52867b29e9141d2ecb76dccf47f04c8b42',
}
SHAPES={'attn_q_b':(1280,32768),'attn_output_b':(8192,5120)}


@lru_cache(maxsize=1)
def implementation_identity():
    import mlx.core as mx
    if importlib.metadata.version('mlx')!='0.32.4':raise RuntimeError('Source native FP8 PP MLX version changed; re-audit required')
    root=Path(mx.__file__).resolve().parent
    if hashlib.sha256((root/'lib/libmlx.dylib').read_bytes()).hexdigest()!=LIBRARY_SHA256:
        raise RuntimeError('Source native FP8 PP MLX build changed; re-audit required')
    for name,expected in HEADER_PINS.items():
        if hashlib.sha256((root/'include/mlx/backend/metal/kernels'/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Source native FP8 PP header changed; re-audit required')
    own=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return hashlib.sha256((POLICY+':'+LIBRARY_SHA256+':'+own+':'+repr(sorted(HEADER_PINS.items()))).encode()).hexdigest()


def projection_kind(name):
    match=re.fullmatch(r'blk\.(0|[1-9][0-9]?)\.(attn_q_b|attn_output_b)\.weight',name)
    return match[2] if match is not None and int(match[1])<40 else None


def maybe_project(name,original,prepared,q,scales,*,counters):
    import mlx.core as mx
    kind=projection_kind(name)
    if kind is None:return None
    k,n=SHAPES[kind]
    from ..runtime.gguf.source_dense_legacy import _device_supported
    supported=(isinstance(original,mx.array) and original.shape==(1,99,k) and original.dtype==mx.float32
        and isinstance(prepared,mx.array) and prepared.shape==original.shape and prepared.dtype==mx.float32
        and isinstance(q,mx.array) and q.shape==(n,k//4) and q.dtype==mx.uint32
        and isinstance(scales,mx.array) and scales.shape==(n,k//32) and scales.dtype==mx.uint8)
    if not supported or not _device_supported():
        key='source_fp8_native_pp_reference_fallback'
        counters[key]=counters.get(key,0)+1
        return None
    implementation_identity()
    value=mx.quantized_matmul(prepared.astype(mx.bfloat16),q,scales=scales,mode='mxfp8')
    if value.dtype!=mx.bfloat16:raise RuntimeError('Source native FP8 PP expected BF16 stock output')
    key='source_fp8_native_pp';counters[key]=counters.get(key,0)+1
    key='source_fp8_native_pp_'+kind;counters[key]=counters.get(key,0)+1
    return value.astype(mx.float32)
