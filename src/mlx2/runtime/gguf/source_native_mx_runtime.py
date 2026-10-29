# SPDX-License-Identifier: MIT
"""Opt-in Source original-expert native primitive, dynamic bounded prefill M.

One GU geometry and one DOWN geometry; no row-specialized JIT variants. Loader
attests original coefficient bytes once on CPU. Bound tokens retain metadata,
never arrays. Activation conversion explicitly consumes HALF values: it is not
a runtime roundtrip/finite proof. Adapter owns shape fallback and BF16 owners.
Provenance: provenance/deepseek-v41-source-native-mx-runtime.json.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import re

from .source_native_mx import HEADER

PROFILE = 'source_original_pp_half_dynamic_m_v1'
SHAPES = {'gate': (5120, 2304), 'up': (5120, 2304), 'down': (2304, 5120)}
TILE = (16, 32)
MIN_ROWS, MAX_ROWS = 16, 99
_SEAL = object()


@dataclass(frozen=True)
class CoefficientAttestation:
    eligible: bool
    reason: str
    source_fingerprint: str
    source_seals_sha256: str
    payload_sha256: str
    layout_sha256: str
    layer: int
    expert: int
    scale_domains: tuple
    _seal: object

    def receipt(self):
        return dict(profile=PROFILE, eligible=self.eligible, reason=self.reason,
            source_fingerprint=self.source_fingerprint, source_seals_sha256=self.source_seals_sha256,
            payload_sha256=self.payload_sha256, layout_sha256=self.layout_sha256,
            layer=self.layer, expert=self.expert, scale_domains=[list(v) for v in self.scale_domains],
            coefficient_gate='all original E8M0 scale bytes104..140 implies every E2M1 coefficient finite/losslessHALF',
            activation_consumption='float32 to contiguous float16; rounding is part of explicit profile',
            qualification=False)


@dataclass(frozen=True)
class BoundExpert:
    attestation: CoefficientAttestation
    expert_object_id: int
    backing_object_id: int
    tensor_object_ids: tuple
    _seal: object


def _layout_record(layout):
    return [dict(name=t.name, offset=t.offset, dtype=t.dtype, shape=list(t.shape)) for t in layout.tensors]


@lru_cache(maxsize=1)
def canonical_original_layout():
    """Native-only immutable CPU descriptor; retain no experts or arrays.

    Keep the ordinary buffer loader's construction/validation untouched. The
    cached ExpertLayout and its nested TensorSlice/tuple fields are frozen;
    every native call still validates the live expert and all typed views.
    """
    from ...adapters.deepseek_v41_source_buffers import canonical_layout
    return canonical_layout('original')


def attest_host(host, *, source_fingerprint, source_seals_sha256, layer, expert):
    """CPU-only loader witness; caller uploads this exact immutable host payload.

    Scales104..140 correspond to powers2^-23..2^13. With E2M1 magnitudes
    0,.5,1,1.5,2,3,4,6, all products are finite exact half values, including
    subnormal2^-24. This conservative gate may reject harmless zero-code groups;
    it never claims arbitrary E8M0-byte or native reduction parity.
    """
    import numpy as np
    from ...adapters.deepseek_v41_source_buffers import HostExpert, canonical_layout
    if (not isinstance(host, HostExpert) or host.layout != canonical_layout('original')
            or type(layer) is not int or not 0 <= layer < 40
            or type(expert) is not int or not 0 <= expert < 384
            or any(type(v) is not str or not re.fullmatch('[0-9a-f]{64}', v)
                for v in (source_fingerprint, source_seals_sha256))):
        raise ValueError('Native runtime attestation requires sealed canonical original source')
    domains = tuple((p, int(np.min(host.views()[p+'.scales'])),
        int(np.max(host.views()[p+'.scales']))) for p in ('gate', 'up', 'down'))
    eligible = all(104 <= low <= high <= 140 for _, low, high in domains)
    layout_sha = hashlib.sha256(json.dumps(_layout_record(host.layout),sort_keys=True,separators=(',', ':')).encode()).hexdigest()
    return CoefficientAttestation(eligible,
        'eligible_conservative_original_scale_domain' if eligible else 'original_scale_domain_outside104..140',
        source_fingerprint, source_seals_sha256, hashlib.sha256(host.data).hexdigest(),
        layout_sha, layer, expert, domains, _SEAL)


def _attestation(a):
    if (not isinstance(a, CoefficientAttestation) or a._seal is not _SEAL
            or type(a.eligible) is not bool or type(a.scale_domains) is not tuple):
        raise ValueError('Native runtime requires CPU loader coefficient attestation')
    if (any(type(v) is not str or not re.fullmatch('[0-9a-f]{64}', v)
            for v in (a.source_fingerprint,a.source_seals_sha256,a.payload_sha256,a.layout_sha256))
            or type(a.layer) is not int or not 0 <= a.layer < 40
            or type(a.expert) is not int or not 0 <= a.expert < 384
            or len(a.scale_domains) != 3
            or tuple(v[0] for v in a.scale_domains if type(v) is tuple and len(v)==3) != ('gate','up','down')
            or any(type(low) is not int or type(high) is not int or not 0 <= low <= high <= 255
                for _,low,high in a.scale_domains)
            or a.eligible != all(104 <= low <= high <= 140 for _,low,high in a.scale_domains)):
        raise ValueError('Native runtime coefficient attestation metadata changed')
    return a


def _expert_metadata(expert):
    import mlx.core as mx
    from ...adapters.deepseek_v41_source_buffers import PackedExpert
    if (not isinstance(expert, PackedExpert) or expert.layout != canonical_original_layout()
            or not isinstance(expert.backing,mx.array) or expert.backing.dtype != mx.uint8
            or expert.backing.shape != (expert.layout.nbytes,)
            or set(expert) != {t.name for t in expert.layout.tensors}):
        raise ValueError('Native runtime requires canonical original PackedExpert')
    dtypes = {'<u4':mx.uint32, '|u1':mx.uint8}
    if any(not isinstance(expert[t.name], mx.array) or expert[t.name].shape != t.shape
            or expert[t.name].dtype != dtypes[t.dtype] or t.offset%128 for t in expert.layout.tensors):
        raise ValueError('Native runtime original typed-view geometry/alignment changed')
    return tuple((t.name,id(expert[t.name])) for t in expert.layout.tensors)


def bind_expert(expert, attestation):
    """Metadata-only after exact-host materialization/fence; retains no arrays."""
    _attestation(attestation)
    ids = _expert_metadata(expert)
    layout_sha = hashlib.sha256(json.dumps(_layout_record(expert.layout),sort_keys=True,separators=(',', ':')).encode()).hexdigest()
    if layout_sha != attestation.layout_sha256:
        raise ValueError('Native runtime loader attestation layout mismatch')
    return BoundExpert(attestation,id(expert),id(expert.backing),ids,_SEAL)


def plan(projection, rows):
    if type(projection) is not str or projection not in SHAPES or type(rows) is not int or not 1 <= rows <= MAX_ROWS:
        raise ValueError('Native runtime requires bounded projection/M1..99 metadata')
    k,n = SHAPES[projection]
    return dict(profile=PROFILE,rows=rows,k=k,n=n,tile=TILE,dispatch_eligible=rows>=MIN_ROWS,
        output_shape=(rows,n),grid=(n//32*32,(rows+15)//16,1),threadgroup=(32,1,1),
        kernel_geometry=(k,n),activation_consumption='float16_from_float32_rounding_allowed',
        weight_format='original_E2M1_E8M0_group32',raw_output='float32',
        transpose_left=False,transpose_right=True,relaxed_precision=False,
        full_weight_repack_bytes=0,full_weight_dequantization_bytes=0,qualification=False)


def kernel_spec(projection):
    p=plan(projection,MIN_ROWS);k,n=p['k'],p['n']
    source=f'''
const int rows=int(layout_offsets_rows[2]);
using AExt=extents<int,{k},dynamic_extent>;
using WExt=extents<int,{k},{n}>;
using CExt=extents<int,{n},dynamic_extent>;
using Scales=tensor_blockwise<tensor_plane_scales,
    device metal_fp8_ue8m0_format,32,1>;
tensor<device half,AExt,tensor_inline>
    a(const_cast<device half*>(x_half),AExt{{rows}});
Scales scales(const_cast<device uchar*>(backing+layout_offsets_rows[1]));
tensor<device metal_fp4_e2m1_format,WExt,tensor_inline,Scales>
    w(const_cast<device uchar*>(backing+layout_offsets_rows[0]),WExt{{}},scales);
tensor<device float,CExt,tensor_inline> c(out,CExt{{rows}});
auto ta=a.slice(0,int(threadgroup_position_in_grid.y*16));
auto tw=w.slice(0,int(threadgroup_position_in_grid.x*32));
auto tc=c.slice(int(threadgroup_position_in_grid.x*32),
                int(threadgroup_position_in_grid.y*16));
constexpr auto descriptor=matmul2d_descriptor(
    16,32,{k},false,true,false,matmul2d_descriptor::mode::multiply);
matmul2d<descriptor,execution_simdgroup> op;
auto acc=op.get_destination_cooperative_tensor<decltype(ta),decltype(tw),float>();
for(auto it=acc.begin();it!=acc.end();++it) *it=0.0f;
op.run(ta,tw,acc);
acc.store(tc);
'''
    return dict(header=HEADER,source=source,input_names=['x_half','backing','layout_offsets_rows'],
        output_names=['out'],ensure_row_contiguous=False,compile_options={'math_mode':'safe'})


def compilation_source(projection):
    s=kernel_spec(projection)
    return s['header']+'''\nkernel void source_native_mx_dynamic(
const device half* x_half [[buffer(0)]],const device uchar* backing [[buffer(1)]],
constant uint* layout_offsets_rows [[buffer(2)]],device float* out [[buffer(3)]],
uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {\n'''+s['source']+'\n}\n'


@lru_cache(maxsize=2)
def _kernel(k,n):
    import mlx.core as mx
    projection='down' if (k,n)==SHAPES['down'] else 'gate'
    if (k,n) not in SHAPES.values():
        raise ValueError('Native runtime unsupported kernel geometry')
    return mx.fast.metal_kernel(name=f'mlx2_source_native_mx_dynamic_k{k}_n{n}_v1',**kernel_spec(projection))


def prepare_half(x):
    """Fresh explicit contiguous activation cast; no scalar/CPU checks."""
    import mlx.core as mx
    if (not isinstance(x,mx.array) or x.dtype != mx.float32 or x.ndim != 2
            or not MIN_ROWS <= x.shape[0] <= MAX_ROWS or x.shape[1] not in (2304,5120)):
        raise ValueError('Native runtime half preparation requires F32M16..99/K2304or5120')
    return mx.contiguous(x.astype(mx.float16))


def project_half(half, expert, bound, *, projection='gate'):
    """Borrow original backing; metadata-only checks, no eval/item/readbacks."""
    import mlx.core as mx
    if (not isinstance(half,mx.array) or half.dtype != mx.float16 or half.ndim != 2
            or not isinstance(bound,BoundExpert) or bound._seal is not _SEAL):
        raise ValueError('Native runtime requires HALF input and bound original expert')
    p=plan(projection,half.shape[0]);a=_attestation(bound.attestation)
    if (not p['dispatch_eligible'] or half.shape[1] != p['k'] or a.eligible is not True
            or bound.expert_object_id != id(expert) or bound.backing_object_id != id(expert.backing)
            or bound.tensor_object_ids != _expert_metadata(expert)):
        raise ValueError('Native runtime shape/coefficient/binding capability refused')
    q,s=expert.layout.tensor(projection+'.qweight'),expert.layout.tensor(projection+'.scales')
    # No public Python strides metadata exists. This is a no-copy identity for
    # prepare_half's contiguous operand and guards explicit standalone callers.
    half=mx.contiguous(half)
    metadata=mx.array([q.offset,s.offset,p['rows']],mx.uint32)
    return _kernel(p['k'],p['n'])(inputs=[half,expert.backing,metadata],grid=p['grid'],
        threadgroup=p['threadgroup'],output_shapes=[p['output_shape']],output_dtypes=[mx.float32])[0]
