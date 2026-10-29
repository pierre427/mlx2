# SPDX-License-Identifier: MIT
"""Unwired native MXFP4/E8M0 TensorOps component; no work on import.

Half inputs are admitted only after exact finite F32 roundtrip checking. Native
dot arithmetic is a distinct unqualified profile, not the stock Source law.
Caller retains immutable inputs/expert leases until output completion.
Provenance: deepseek-v41-source-native-mx.json.
"""
from dataclasses import dataclass
from functools import lru_cache
import time

SHAPES = {'gate': (5120, 2304), 'up': (5120, 2304), 'down': (2304, 5120)}
ROWS = (1, 3, 5, 9, 16, 32, 34, 98)
TILE = (16, 32)
TILES = ((8,32),(16,32),(16,64),(32,32),(32,64))
PROFILE = 'native_half_mxfp4_e8m0_f32_v1'
_SEAL = object()
_PREPARATION_SCOPE = 'Explicit evaluated input-representation gate; not native dot parity or operator timing'

HEADER = r'''
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#if __METAL_VERSION__ < 410
#error "Native Source MX component requires Metal4.1/macOS27 TensorOps"
#endif
using namespace metal;
using namespace mpp::tensor_ops;
'''


def plan(projection, rows, *, tile=TILE):
    if (type(projection) is not str or projection not in SHAPES
            or type(rows) is not int or rows not in ROWS
            or type(tile) is not tuple or tile not in TILES or any(type(v)is not int for v in tile)):
        raise ValueError('Native MX component requires bounded projection/M/tile')
    k, n = SHAPES[projection]
    bm,bn=tile
    return dict(profile=PROFILE, projection=projection, rows=rows, k=k, n=n,
        geometry_profile=PROFILE+f':tile{bm}x{bn}',
        output_shape=(rows,n), tile=tile, threadgroup=(32,1,1),
        threadgroups=(n//bn,(rows+bm-1)//bm,1), grid=(n//bn*32,(rows+bm-1)//bm,1),
        input_dtype='float16_exact_from_float32', output_dtype='float32',
        weight_format='original_E2M1_E8M0_group32', scale_block_factors=(32,1),
        transpose_left=False, transpose_right=True, relaxed_precision=False,
        matmul_mode='multiply', accumulator='explicit_F32_positive_zero',
        full_weight_repack_bytes=0, full_weight_dequantization_bytes=0,
        activation_cast_bytes=rows*k*2, output_bytes=rows*n*4,
        padded_rows=(rows+bm-1)//bm*bm, language='metal4.1', deployment='macos27')


def kernel_spec(projection, rows, *, tile=TILE):
    p=plan(projection,rows,tile=tile);k,n=p['k'],p['n'];bm,bn=p['tile']
    body=f'''
using AExt = extents<int, {k}, {rows}>;
using WExt = extents<int, {k}, {n}>;
using CExt = extents<int, {n}, {rows}>;
using Scales = tensor_blockwise<tensor_plane_scales,
    device metal_fp8_ue8m0_format, 32, 1>;
// SDK TensorOps type dispatch does not strip element const. These descriptors
// are read-only operands; the kernel ABI stays const and no input store occurs.
tensor<device half, AExt, tensor_inline>
    a(const_cast<device half*>(x_half), AExt{{}});
Scales scales(const_cast<device uchar*>(backing + layout_offsets[1]));
tensor<device metal_fp4_e2m1_format, WExt, tensor_inline, Scales>
    w(const_cast<device uchar*>(backing + layout_offsets[0]), WExt{{}}, scales);
tensor<device float, CExt, tensor_inline> c(out, CExt{{}});
auto ta = a.slice(0, int(threadgroup_position_in_grid.y * {bm}));
auto tw = w.slice(0, int(threadgroup_position_in_grid.x * {bn}));
auto tc = c.slice(int(threadgroup_position_in_grid.x * {bn}),
                  int(threadgroup_position_in_grid.y * {bm}));
constexpr auto descriptor = matmul2d_descriptor(
    {bm}, {bn}, {k}, false, true, false, matmul2d_descriptor::mode::multiply);
matmul2d<descriptor, execution_simdgroup> op;
auto acc = op.get_destination_cooperative_tensor<decltype(ta),decltype(tw),float>();
for (auto it=acc.begin(); it!=acc.end(); ++it) *it=0.0f;
op.run(ta, tw, acc);
acc.store(tc);
'''
    return dict(p,header=HEADER,source=body,input_names=['x_half','backing','layout_offsets'],
                output_names=['out'],compile_options={'math_mode':'safe'},
                ensure_row_contiguous=False)


def compilation_source(projection, rows, *, tile=TILE):
    """CPU-only standalone source of the exact JIT header/body."""
    s=kernel_spec(projection,rows,tile=tile)
    return s['header']+r'''
kernel void source_native_mx_component(
    const device half* x_half [[buffer(0)]],
    const device uchar* backing [[buffer(1)]],
    constant uint* layout_offsets [[buffer(2)]],
    device float* out [[buffer(3)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
'''+s['source']+'\n}\n'


@dataclass(frozen=True)
class _InputAdmission:
    raw: object
    half: object
    shape: tuple
    preparation_gate_seconds: float
    prepared_bytes: int


@dataclass(frozen=True)
class PreparedInput:
    raw: object
    half: object
    receipt: dict
    _seal: object
    _admission: object = None


def _receipt(shape,seconds,prepared_bytes):
    return dict(profile=PROFILE,shape=list(shape),source_dtype='float32',prepared_dtype='float16',
        finite=True,float_half_float_bits_exact=True,signed_zero_preserved=True,
        preparation_gate_seconds=seconds,prepared_bytes=prepared_bytes,
        contiguous_half_explicit=True,scope=_PREPARATION_SCOPE)


def _receipt_matches(receipt,expected):
    # Refuse array-valued or ambiguous metadata before Python equality could
    # trigger a tensor truth/readback. This is not a tensor-data checksum.
    if (type(receipt)is not dict or receipt.keys()!=expected.keys()
            or any(type(receipt[k])is not type(v)for k,v in expected.items())
            or any(type(v)is not int for v in receipt['shape'])):
        return False
    return receipt==expected


def prepare_input(x):
    """Explicit diagnostic gate; evaluates two scalar predicates before timing math."""
    import mlx.core as mx
    if (not isinstance(x,mx.array) or x.dtype!=mx.float32 or x.ndim!=2
            or x.shape[0] not in ROWS or x.shape[1] not in (2304,5120)):
        raise ValueError('Native MX preparation requires bounded F32 input')
    # MLX Python arrays expose no public strides accessor. Materialize this
    # activation-only operand explicitly; coefficient backing is never copied.
    start=time.perf_counter();half=mx.contiguous(x.astype(mx.float16))
    restored=half.astype(mx.float32)
    finite=mx.all(mx.isfinite(x))
    exact=mx.all(x.view(mx.uint32)==restored.view(mx.uint32))
    mx.eval(half,finite,exact)
    finite_ok,exact_ok=bool(finite.item()),bool(exact.item())
    if not finite_ok or not exact_ok:
        raise ValueError('Native MX half input changes F32 values or is nonfinite')
    elapsed=time.perf_counter()-start
    admission=_InputAdmission(x,half,tuple(x.shape),elapsed,half.nbytes)
    return PreparedInput(x,half,_receipt(admission.shape,elapsed,half.nbytes),_SEAL,admission)


def _validate_prepared(prepared):
    """Metadata-only capability guard. Caller keeps admitted input immutable."""
    import mlx.core as mx
    if (not isinstance(prepared,PreparedInput) or prepared._seal is not _SEAL
            or not isinstance(prepared._admission,_InputAdmission)):
        raise ValueError('Native MX requires explicitly admitted PreparedInput')
    a=prepared._admission
    if (prepared.raw is not a.raw or prepared.half is not a.half
            or not isinstance(prepared.raw,mx.array) or prepared.raw.dtype!=mx.float32
            or not isinstance(prepared.half,mx.array) or prepared.half.dtype!=mx.float16
            or tuple(prepared.raw.shape)!=a.shape or tuple(prepared.half.shape)!=a.shape
            or prepared.half.nbytes!=a.prepared_bytes
            or not _receipt_matches(prepared.receipt,_receipt(a.shape,a.preparation_gate_seconds,a.prepared_bytes))):
        raise ValueError('Native MX admission input/metadata/receipt changed')
    return a


def _bindings(prepared,expert,projection,*,tile=TILE):
    import mlx.core as mx
    from ...adapters.deepseek_v41_source_buffers import PackedExpert,canonical_layout
    _validate_prepared(prepared)
    p=plan(projection,prepared.half.shape[0],tile=tile)
    if prepared.half.shape!=(p['rows'],p['k']) or prepared.raw.shape!=prepared.half.shape:
        raise ValueError('Native MX input does not match projection K')
    if (not isinstance(expert,PackedExpert) or expert.layout!=canonical_layout('original')
            or not isinstance(expert.backing,mx.array) or expert.backing.dtype!=mx.uint8
            or expert.backing.shape!=(expert.layout.nbytes,)):
        raise ValueError('Native MX requires borrowed canonical original expert backing')
    q,s=expert.layout.tensor(projection+'.qweight'),expert.layout.tensor(projection+'.scales')
    if (q.shape!=(p['n'],p['k']//8) or s.shape!=(p['n'],p['k']//32)
            or q.offset%128 or s.offset%128):
        raise ValueError('Native MX original code/scale geometry or base alignment changed')
    return p,(q.offset,s.offset)


@lru_cache(maxsize=120)
def _kernel(projection,rows,tile):
    import mlx.core as mx
    s=kernel_spec(projection,rows,tile=tile)
    suffix=''if tile==TILE else f'_t{tile[0]}x{tile[1]}'
    return mx.fast.metal_kernel(name=f'mlx2_native_mx_{projection}_m{rows}{suffix}_v1',
        input_names=s['input_names'],output_names=s['output_names'],source=s['source'],
        header=s['header'],ensure_row_contiguous=False,compile_options=s['compile_options'])


def _dispatch(half,expert,p,offsets):
    import mlx.core as mx
    if mx.default_device()!=mx.gpu:
        raise ValueError('Native MX component requires caller-owned GPU')
    return _kernel(p['projection'],p['rows'],p['tile'])(
        inputs=[half,expert.backing,mx.array(offsets,mx.uint32)],
        grid=p['grid'],threadgroup=p['threadgroup'],output_shapes=[p['output_shape']],
        output_dtypes=[mx.float32])[0]


def project(prepared,expert,*,projection='gate',tile=TILE):
    """Schedule raw F32 only. Caller applies/times existing BF16 owner separately."""
    p,offsets=_bindings(prepared,expert,projection,tile=tile)
    return _dispatch(prepared.half,expert,p,offsets)


def project_from_admitted_raw(admission,expert,*,projection='gate',tile=TILE,rounded=True):
    """Fresh cast+native+optional owner graph after one immutable-input admission.

    This does not evaluate, repeat predicates, read scalar flags or accept a
    replacement raw tensor. Caller keeps admission.raw immutable and owns the
    final completion boundary. The earlier checked-stage metric stays separate.
    """
    import mlx.core as mx
    if type(rounded)is not bool:
        raise ValueError('Native MX admitted stage rounded flag must be boolean')
    p,offsets=_bindings(admission,expert,projection,tile=tile)
    half=mx.contiguous(admission.raw.astype(mx.float16))
    raw=_dispatch(half,expert,p,offsets)
    return raw.astype(mx.bfloat16).astype(mx.float32)if rounded else raw
