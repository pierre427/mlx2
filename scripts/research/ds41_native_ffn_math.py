# SPDX-License-Identifier: MIT
"""Unwired complete original-expert FFN graph; caller owns GPU and lifetimes.

Admission is evaluated diagnostic work. Subsequent fresh graph construction has
no evaluation, scalar predicates, CPU weight reads or arbitrary input override.
Native dot reductions are a separate numerical profile. Provenance:
provenance/deepseek-v41-native-ffn-math.json.
"""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
import copy
import time

PROFILE = 'original_native_half_mxfp4_full_ffn_v1'
_SEAL = object()


@dataclass(frozen=True)
class StageAdmission:
    mx: object
    arrays: object
    case: object
    expert: object
    receipt: dict
    _seal: object
    _state: object


@dataclass(frozen=True)
class _StageState:
    mx: object
    arrays: object
    case: object
    expert: object
    backing: object
    tensors: tuple
    tile: tuple
    rows: int
    zero: object
    masks: tuple
    offsets: tuple
    kernels: tuple
    middle_kernel: object
    assembly_kernel: object
    inverse: object
    receipt: dict



def assembly_kernel_spec():
    """Address-only selected-part substitution in Source's ordered F32 fold."""
    return dict(input_names=['parts', 'chosen', 'ranks'], output_names=['output'],
        header='#pragma clang fp contract(off)\n', source=r'''
const uint i=thread_position_in_grid.x;
if(i>=ROWS*5120) return;
const uint row=i/5120, column=i%5120;
float total=0.0f;
#pragma clang loop unroll(disable)
for(uint slot=0;slot<6;++slot) {
    const float value=slot==ranks[row] ? chosen[i] : parts[(row*6+slot)*5120+column];
    total=total+value;
}
output[i]=total;
''', ensure_row_contiguous=True, compile_options={'math_mode':'safe'})


@lru_cache(maxsize=1)
def _assembly_kernel():
    import mlx.core as mx
    return mx.fast.metal_kernel(name='mlx2_source_native_ffn_selected_ordered_fold_v1',
        **assembly_kernel_spec())

def _owned(mx, x):
    return x.astype(mx.bfloat16).astype(mx.float32)


def _guard(a):
    if not isinstance(a, StageAdmission) or a._seal is not _SEAL or not isinstance(a._state, _StageState):
        raise ValueError('Native FFN requires one-time StageAdmission')
    s = a._state
    if (a.mx is not s.mx or a.arrays is not s.arrays or a.case is not s.case or a.expert is not s.expert
            or a.expert.backing is not s.backing or a.receipt is not s.receipt
            or any(a.expert[name] is not value for name, value in s.tensors)):
        raise ValueError('Native FFN admitted operand identity changed')
    return s


def _native(mx, s, half, index):
    from mlx2.runtime.gguf import source_native_mx as native
    projection = ('gate', 'up', 'down')[index]
    p = native.plan(projection, s.rows, tile=s.tile)
    return s.kernels[index](inputs=[half, s.backing, s.offsets[index]],
        grid=p['grid'], threadgroup=p['threadgroup'],
        output_shapes=[p['output_shape']], output_dtypes=[mx.float32])[0]


def _middle(mx, s, gate, up):
    # Exact existing Source fused middle source and served-exp witness, resolved
    # once during admission so a timed graph cannot initiate an exp probe.
    return s.middle_kernel(inputs=[gate, up, s.arrays['route_scores'],
        s.arrays['_local_slots'], s.inverse],
        template=[('ELEMENTS', s.rows*2304), ('WIDTH', 2304)],
        grid=(s.rows*2304, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(s.rows, 2304)], output_dtypes=[mx.float32])[0]


def _graph(a, *, native, frontiers, canary):
    if type(frontiers) is not bool or type(canary) is not bool:
        raise ValueError('Native FFN diagnostic/canary flags must be bool')
    s = _guard(a); mx = a.mx
    shared = s.zero if canary else s.arrays['shared_gu_input']
    selected = shared[s.arrays['route_rows']]
    def stock(x, projection):
        return mx.quantized_matmul(x, s.expert[projection+'.qweight'],
            scales=s.expert[projection+'.scales'], mode='mxfp4')
    half = mx.contiguous(selected.astype(mx.float16)) if native else None
    gate = _native(mx, s, half, 0) if native else stock(selected, 'gate')
    up = _native(mx, s, half, 1) if native else stock(selected, 'up')
    middle = _middle(mx, s, gate, up)
    down = _native(mx, s, mx.contiguous(middle.astype(mx.float16)), 2) if native else stock(middle, 'down')
    owned = _owned(mx, down)
    # Same single sequential F32 fold as Source ordered_combine. Only input
    # addressing differs: select this expert's new owner at its admitted rank.
    total = s.assembly_kernel(inputs=[s.arrays['assembly_parts'], owned,
        s.arrays['assembly_rank']], template=[('ROWS', s.rows)],
        grid=(s.rows*5120, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(s.rows, 5120)], output_dtypes=[mx.float32])[0]
    if not frontiers:
        return total
    # These arrays are diagnostic reification, never part of timed assembly.
    parts = [mx.where(s.masks[slot], owned, s.arrays['assembly_parts'][:, slot, :])
        for slot in range(6)]
    import mlx.nn as nn
    g, u = _owned(mx, gate), _owned(mx, up)
    weighted = (nn.silu(mx.minimum(g, 10.))*mx.clip(u, -10., 10.))*s.arrays['route_scores'][:, None]
    return dict(shared_gu_input=shared, gu_input=selected, gate_raw=gate, up_raw=up,
        gate_owned=g, up_owned=u, weighted_raw=weighted, weighted_owned=_owned(mx, weighted),
        middle=middle, middle_reference=middle, down_input=middle, down_raw=down,
        down_owned=owned, assembly_parts=mx.stack(parts, axis=1), assembly_output=total)


def reference_graph(admission, *, frontiers=False, canary=False):
    """Fresh stock F32-MXFP4 stage, exact existing middle and assembly order."""
    return _graph(admission, native=False, frontiers=frontiers, canary=canary)


def native_graph(admission, *, frontiers=False, canary=False):
    """Fresh gather/cast/GU/middle/cast/DOWN/owner/assembly; no readbacks."""
    return _graph(admission, native=True, frontiers=frontiers, canary=canary)


def admit(mx, np, frame, expert, *, tile=(16, 32)):
    """One-time actual-frame/actual-coefficient and native-created-input gates.

    Actual backing and typed views are inspected on CPU, outside timing. Caller
    separately verifies checkpoint/file seals and materializer provenance. The
    caller retains immutable capture and expert lease through graph completion.
    """
    from ds41_native_ffn_capture import validate_arrays
    from ds41_native_mx_components import coefficient_gate, half_roundtrip
    from mlx2.adapters.deepseek_v41_source_buffers import PackedExpert, canonical_layout
    from mlx2.runtime.gguf import source_native_mx as native, source_middle_fusion as middle
    if (type(frame) is not dict or set(frame) != {'case', 'arrays'}
            or not isinstance(expert, PackedExpert) or expert.layout != canonical_layout('original')
            or not isinstance(expert.backing, mx.array) or expert.backing.dtype != mx.uint8
            or expert.backing.shape != (expert.layout.nbytes,)
            or mx.default_device() != mx.gpu):
        raise ValueError('Native FFN requires actual frame and canonical original expert on owned GPU')
    case, cpu = frame['case'], frame['arrays']
    validate_arrays(case, cpu)
    rows = case['rows']
    for projection in ('gate', 'up', 'down'):
        native.plan(projection, rows, tile=tile)
    start = time.perf_counter()
    payload = np.array(expert.backing)
    coefficients = {}
    for projection in ('gate', 'up', 'down'):
        views = {}
        for field in ('qweight', 'scales'):
            name = projection+'.'+field; t = expert.layout.tensor(name)
            expected = payload[t.offset:t.offset+t.nbytes].view(t.dtype).reshape(t.shape)
            actual = np.array(expert[name])
            if actual.shape != t.shape or actual.dtype != expected.dtype or actual.tobytes() != expected.tobytes():
                raise ValueError('Native FFN typed view is not exact canonical backing: '+name)
            views[field] = expected
        coefficients[projection] = coefficient_gate(np, views['qweight'], views['scales'])
        if coefficients[projection]['eligible'] is not True:
            raise ValueError('Native FFN coefficient representation refused: '+projection)
    gu_gate = half_roundtrip(np, cpu['gu_input'])
    if gu_gate['finite'] is not True or gu_gate['bitwise_exact'] is not True:
        raise ValueError('Native FFN actual gathered input changes in half')
    # Fresh immutable uploads; no external mutable arrays/mapping remain held.
    uploaded = {key: mx.array(value) for key, value in cpu.items()}
    uploaded['_local_slots'] = mx.array(np.arange(rows, dtype=np.uint32))
    uploaded = MappingProxyType(uploaded)
    frozen_case = MappingProxyType(copy.deepcopy(case))
    masks = tuple(mx.array((cpu['assembly_rank'] == slot)[:, None]) for slot in range(6))
    offsets = tuple(mx.array((expert.layout.tensor(p+'.qweight').offset,
        expert.layout.tensor(p+'.scales').offset), mx.uint32) for p in ('gate', 'up', 'down'))
    spelling = middle.exp_spelling()
    receipt = dict(profile=PROFILE, native_profile=native.PROFILE, rows=rows, tile=list(tile),
        eligible=False, coefficient_gates=coefficients, gu_half_roundtrip=gu_gate,
        source_kind='original', shared_layer_qat_included=False, cpu_routing_included=False,
        gather_cast_gu_middle_down_owner_assembly_included=True,
        untimed_actual_backing_readback_bytes=int(payload.nbytes),
        untimed_typed_view_readback_bytes=int(payload.nbytes), gpu_weight_repack_bytes=0,
        existing_middle_exp_spelling=spelling, qualification=False, selected=False,
        numerical_scope='Native dot reduction distinct; existing Source middle/owners/assembly order unchanged')
    state = _StageState(mx, uploaded, frozen_case, expert, expert.backing,
        tuple((t.name, expert[t.name]) for t in expert.layout.tensors), tile, rows,
        mx.zeros((99, 5120), dtype=mx.float32), masks, offsets,
        tuple(native._kernel(p, rows, tile) for p in ('gate', 'up', 'down')),
        middle._kernel(spelling), _assembly_kernel(), mx.array([1/448], mx.float32), receipt)
    admission = StageAdmission(mx, uploaded, frozen_case, expert, receipt, _SEAL, state)
    down_gates = {}
    for canary in (False, True):
        graph = native_graph(admission, frontiers=True, canary=canary)
        mx.eval(*graph.values())
        down_gate = half_roundtrip(np, np.array(graph['middle']))
        if down_gate['finite'] is not True or down_gate['bitwise_exact'] is not True:
            raise ValueError('Native FFN native-generated DOWN input changes in half')
        down_gates['zero' if canary else 'actual'] = down_gate
        for name in ('gate_raw', 'up_raw', 'middle', 'down_raw', 'assembly_output'):
            if not np.isfinite(np.array(graph[name])).all():
                raise ValueError('Native FFN admission nonfinite native frontier: '+name)
    receipt.update(eligible=True, native_generated_down_half_roundtrip=down_gates,
        admission_seconds=time.perf_counter()-start,
        timing_scope='Evaluated admission/CPU coefficient inspection excluded from pure fresh graph timing')
    return admission
