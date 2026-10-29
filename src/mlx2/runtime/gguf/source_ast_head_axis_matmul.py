# SPDX-License-Identifier: MIT
"""Native attention products with an explicit singleton M axis per head.

Both padded and unpadded variants are experiments requiring actual GPU raw
parity. No shader body, global cache, Python query loop or default selection.
Provenance: deepseek-v41-source-ast-head-axis-matmul.json.
"""
import hashlib
from pathlib import Path

PROFILE='source_ast_native_head_axis_candidate_v1'


def plan(batches,heads,k,n,*,transpose_b,padded=True):
    if (any(type(v)is not int for v in(batches,heads,k,n)) or not 1<=batches<=16
            or heads!=64 or not 1<=k<=512 or not 1<=n<=512
            or type(transpose_b)is not bool or type(padded)is not bool):
        raise ValueError('Source AST head-axis candidate geometry unsupported')
    stride=k+int(padded)
    return dict(profile=PROFILE,padded=padded,transpose_b=transpose_b,
        left_shape=(batches,heads,1,k),left_element_strides=(heads*stride,stride,k,1),
        coefficient_logical_shape=(batches,heads,k,n),
        coefficient_storage_shape=(batches,n,k)if transpose_b else(batches,k,n),
        coefficient_head_stride=0,coefficient_head_copies=0,native_matmul_calls=1,
        activation_padding_bytes=batches*heads*4 if padded else 0,
        output_shape=(batches,heads,n),qualified=False,exact_gpu_claim=False)


def prepare_operands(mx,left,right,*,transpose_b,padded=True):
    """Prepare array views on the caller's stream, including explicit CPU tests.

    Logical right is [G,K,N], with G=1 allowed for shared coefficients.
    No eval, CPU buffer inspection, shader construction or host routing.
    """
    if (left.ndim!=3 or right.ndim!=3 or left.dtype!=mx.float32 or right.dtype!=mx.float32
            or right.shape[0]not in(1,left.shape[0])or left.shape[-1]!=right.shape[-2]):
        raise ValueError('Source AST head-axis input geometry/dtype mismatch')
    batches,heads,k=left.shape;n=right.shape[-1]
    d=plan(batches,heads,k,n,transpose_b=transpose_b,padded=padded)
    if padded:
        storage=mx.concatenate((left,mx.zeros((batches,heads,1),mx.float32)),axis=-1)
        a=mx.as_strided(storage,shape=d['left_shape'],strides=d['left_element_strides'])
    else:
        a=mx.contiguous(left).reshape(d['left_shape'])
    if transpose_b:
        storage=mx.contiguous(mx.swapaxes(right,-1,-2))
        b=mx.swapaxes(storage[:,None],-1,-2)
    else:
        storage=mx.contiguous(right)
        b=storage[:,None]
    b=mx.broadcast_to(b,d['coefficient_logical_shape'])
    return a,b


def private_matmul(left,right,*,transpose_b,padded=True,check_live):
    """Schedule one native product; caller owns GPU and retains it until done."""
    import mlx.core as mx
    if not callable(check_live)or check_live()is not True:
        raise ValueError('Source AST head-axis candidate requires live ownership')
    if mx.default_device()!=mx.gpu:
        raise ValueError('Source AST head-axis candidate requires parent-owned GPU')
    a,b=prepare_operands(mx,left,right,transpose_b=transpose_b,padded=padded)
    return mx.matmul(a,b).reshape(left.shape[0],left.shape[1],right.shape[-1])


def implementation_identity():
    from .source_dense import _installed,LIBRARY_SHA256
    _installed()
    return hashlib.sha256(Path(__file__).read_bytes()+LIBRARY_SHA256.encode()).hexdigest()
