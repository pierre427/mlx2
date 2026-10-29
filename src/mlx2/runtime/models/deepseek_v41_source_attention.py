# SPDX-License-Identifier: MIT
"""Opt-in Source attention dispatch; see oMLX attention provenance.

The distinct numerical law never changes decode, the index selection, cache
storage, or the existing owning BF16 boundary. Unsupported calls use reference.
"""
from functools import lru_cache
from pathlib import Path
import hashlib

POLICY = 'omlx_nax_bf16_online_v1'
KERNEL_SHA256 = '9ecfb346f1df8d51f5a248570a9a4afd48395bfb898b1b8e7754fa3bd8c90391'


@lru_cache(maxsize=1)
def implementation_identity():
    from . import deepseek_v41_omlx_nax_attention as native
    actual = hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
    if actual != KERNEL_SHA256:
        raise RuntimeError('Source oMLX attention implementation changed; re-audit required')
    dispatch_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return hashlib.sha256((POLICY+':'+actual+':'+dispatch_sha).encode()).hexdigest()


def eligible(q, local, pooled, indices, valid, sinks, *, trusted_retained):
    import mlx.core as mx
    from ..gguf.source_dense_legacy import _device_supported
    try:
        n=q.shape[2];p=pooled.shape[1]
        return bool(trusted_retained and q.shape==(1,64,n,512) and 16<=n<=128
                    and 1<=local.shape[2]<=255 and local.shape==(1,1,local.shape[2],512)
                    and pooled.shape==(1,p,512) and 1<=p<=512
                    and indices.shape==(1,n,p) and valid.shape==indices.shape
                    and indices.dtype in (mx.int32,mx.uint32) and valid.dtype==mx.bool_
                    and sinks.shape==(64,) and all(x.dtype==mx.float32 for x in (q,local,pooled,sinks))
                    and _device_supported())
    except (TypeError,ValueError,AttributeError,IndexError):
        return False


def dispatch(q,local,mask,pooled,indices,valid,sinks,*,trusted_retained,reference,counters):
    if not eligible(q,local,pooled,indices,valid,sinks,trusted_retained=trusted_retained):
        key='source_omlx_nax_attention_reference_fallback'
        counters[key]=counters.get(key,0)+1
        return reference()
    implementation_identity()
    from . import deepseek_v41_omlx_nax_attention as native
    result=native.attention(q,local,mask,pooled,indices,valid,sinks,numerical_profile=native.PROFILE)
    key='source_omlx_nax_attention'
    counters[key]=counters.get(key,0)+1
    # Logical temporary staging only: never physical DRAM traffic/cache savings.
    key='source_omlx_nax_operand_staging_logical_bytes'
    counters[key]=counters.get(key,0)+2*(q.size+local.size+pooled.size)
    return result
