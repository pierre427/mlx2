# SPDX-License-Identifier: MIT
"""Default-off AST DOWN geometry: private M1 dots, 1/4/8 SIMD groups per TG.

Only independent four-output owners share a threadgroup. Dot helper bodies,
32-lane reduction, per-occurrence BF16 store, route checks and combine order
remain unchanged. The caller retains the bank lease until consumers complete.
No import-time device work and no automatic dispatch/qualification.
Provenance: deepseek-v41-source-ast-down-groups.json.
"""
from functools import lru_cache
import hashlib
from pathlib import Path

PROFILE = "source_ast_down_simdgroups_candidate_v1"
SIMDGROUPS = (1, 4, 8)
OCCURRENCES = 96
HIDDEN = 5120
INTERMEDIATE = 2304


def plan(simdgroups=8):
    if type(simdgroups) is not int or simdgroups not in SIMDGROUPS:
        raise ValueError("Source AST DOWN requires1/4/8 independent SIMD groups")
    rows = 4*simdgroups
    return dict(profile=PROFILE, simdgroups=simdgroups, lanes=32,
        outputs_per_SIMDgroup=4, output_rows_per_threadgroup=rows,
        grid=(HIDDEN//rows*simdgroups*32, OCCURRENCES, 1),
        threadgroup=(simdgroups*32, 1, 1),
        threadgroups_per_layer=HIDDEN//rows*OCCURRENCES,
        total_SIMDgroups_per_layer=1280*OCCURRENCES,
        threadgroups_per_40_layers=HIDDEN//rows*OCCURRENCES*40,
        shared_partial_bytes=rows*4, dot_rows=1, output_shape=(OCCURRENCES, HIDDEN),
        existing_control=simdgroups==1, expert_row_panel_reuse=False,
        changed_arithmetic_expressions=False, exact_gpu_parity=False,
        qualified=False, performance_claim=False)


def kernel_spec(key, *, simdgroups=8):
    """CPU source builder; reject changed anchors before constructing a shader."""
    descriptor = plan(simdgroups)
    from .source_ast_batched_moe import down_spec as original
    spec = original(key)
    if simdgroups == 1:
        return dict(spec, descriptor=descriptor)
    source = spec['source']
    old = 'const uint tile=threadgroup_position_in_grid.x,lane=thread_index_in_simdgroup;'
    new = ('const uint physical_tile=threadgroup_position_in_grid.x;\n'
           'const uint sg=simdgroup_index_in_threadgroup;\n'
           f'const uint tile=physical_tile*{simdgroups}u+sg,lane=thread_index_in_simdgroup;')
    if source.count(old) != 1:
        raise RuntimeError("Source AST DOWN pinned tile geometry changed")
    source = source.replace(old, new)
    old = 'threadgroup float partials[4];'
    new = (f'threadgroup float shared_partials[{4*simdgroups}];\n'
           'threadgroup float* partials=shared_partials+sg*4;')
    if source.count(old) != 1 or source.count('threadgroup_barrier(mem_flags::mem_threadgroup);') != 1:
        raise RuntimeError("Source AST DOWN pinned private scratch/barrier changed")
    source = source.replace(old, new)
    # Every early return before the barrier depends only on occurrence or its
    # row status. Both are uniform across all SIMD groups of this threadgroup.
    return dict(spec, source=source, descriptor=descriptor)


@lru_cache(maxsize=128)
def _kernel(key, simdgroups):
    import mlx.core as mx
    spec = kernel_spec(key, simdgroups=simdgroups)
    suffix = hashlib.sha256(repr((key, simdgroups)).encode()).hexdigest()[:18]
    return mx.fast.metal_kernel(name='mlx2_source_ast_down_groups_'+suffix,
        input_names=spec['input_names'], output_names=spec['output_names'],
        header=spec['header'], source=spec['source'], ensure_row_contiguous=False)


def down(middle, ids, groups, ordered, status, lease, *, simdgroups=8, check_live):
    """Schedule only DOWN; no eval/fence, no status readback or route mutation."""
    import mlx.core as mx
    from .source_resident_bank_gather import layout_key
    from .source_ast_batched_moe import MAX_BANKS
    descriptor = plan(simdgroups)
    if not callable(check_live) or check_live() is not True:
        raise ValueError("Source AST DOWN groups requires live caller ownership")
    if mx.default_device() != mx.gpu:
        raise ValueError("Source AST DOWN groups requires parent-owned GPU")
    if (middle.shape != (96, 2304) or middle.dtype != mx.float32
            or ids.shape != (96,) or ids.dtype != mx.uint32
            or groups.shape != (384, 8) or groups.dtype != mx.uint32
            or ordered.shape != (96,) or ordered.dtype != mx.uint32
            or status.shape != (16,) or status.dtype != mx.uint32):
        raise ValueError("Source AST DOWN groups input geometry mismatch")
    key = layout_key(lease)
    if len(key) > MAX_BANKS or any(b.dtype != mx.uint32 or a.dtype != mx.uint32
                                 for b, a in zip(lease.banks, lease.anchors)):
        raise ValueError("Source AST DOWN groups bank binding mismatch")
    # All spans and actual aliases belong to the existing bank publisher. This
    # consumer borrows anchors; caller retains the full immutable bank lease.
    return _kernel(key, simdgroups)(inputs=[middle, ids, groups, ordered, status, *lease.anchors],
        grid=descriptor['grid'], threadgroup=descriptor['threadgroup'],
        output_shapes=[descriptor['output_shape']], output_dtypes=[mx.float32])[0]


def decode_tile(x, ids, scores, lease, *, layer, expected_generation, check_live,
                exp_spelling=None, down_simdgroups=8):
    """Explicit experiment wrapper: existing route/GU/status/combine, new DOWN.

    It schedules the same six logical stages and returns the same owner roots.
    No production caller or numerical profile selects it automatically.
    """
    import mlx.core as mx
    from . import source_ast_batched_moe as ordinary
    from .source_resident_bank_gather import layout_key
    plan(down_simdgroups)
    if not callable(check_live) or check_live() is not True:
        raise ValueError("Source AST DOWN experiment requires live ownership")
    if (x.shape != (16, 5120) or x.dtype != mx.float32 or ids.shape != (16, 6)
            or ids.dtype != mx.uint32 or scores.shape != ids.shape or scores.dtype != mx.float32
            or type(layer) is not int or not 0 <= layer < 40
            or type(expected_generation) is not int or not 0 < expected_generation < 1<<32):
        raise ValueError("Source AST DOWN experiment geometry mismatch")
    if mx.default_device() != mx.gpu:
        raise ValueError("Source AST DOWN experiment requires parent-owned GPU")
    key = layout_key(lease)
    if (len(key) > ordinary.MAX_BANKS or lease.lut.shape != (40, 384, 4) or lease.lut.dtype != mx.uint32
            or any(b.dtype != mx.uint32 or a.dtype != mx.uint32 for b,a in zip(lease.banks,lease.anchors))):
        raise ValueError("Source AST DOWN experiment bank metadata mismatch")
    if exp_spelling is None:
        from .source_middle_fusion import exp_spelling as current
        exp_spelling = current()
    ids=ids.reshape(-1);scores=scores.reshape(-1)
    groups,ordered,ranks=ordinary._kernel('route',key,layer,expected_generation)(inputs=[ids,lease.lut],
        grid=(384,1,1),threadgroup=(128,1,1),output_shapes=[(384,8),(96,),(96,)],output_dtypes=[mx.uint32]*3)
    tasks,count,route_status=ordinary._kernel('tasks',())(inputs=[groups,ids,scores],grid=(1,1,1),threadgroup=(1,1,1),
        output_shapes=[(96,2),(1,),(16,)],output_dtypes=[mx.uint32]*3)
    middle,finite=ordinary._kernel('gu',key,spelling=exp_spelling)(inputs=[x,scores,groups,ordered,tasks,count,mx.array([1/448],mx.float32),*lease.anchors],
        grid=(72*256,1,96),threadgroup=(256,1,1),output_shapes=[(96,2304),(96,72)],output_dtypes=[mx.float32,mx.uint32])
    status=ordinary._kernel('status',())(inputs=[route_status,ranks,finite],grid=(16,1,1),threadgroup=(32,1,1),
        output_shapes=[(16,)],output_dtypes=[mx.uint32])[0]
    parts=down(middle,ids,groups,ordered,status,lease,simdgroups=down_simdgroups,check_live=check_live)
    output=ordinary._kernel('combine',())(inputs=[parts,ids,ranks,status],grid=(5120,16,1),threadgroup=(256,1,1),
        output_shapes=[(16,5120)],output_dtypes=[mx.float32])[0]
    return ordinary.TileResult(output,status,middle,groups,
        (x,ids,scores,groups,ordered,ranks,tasks,count,route_status,middle,finite,status,parts,output))


@lru_cache(maxsize=1)
def implementation_identity():
    names=('source_ast_batched_down_groups.py','source_ast_batched_moe.py','source_down_fusion.py','source_resident_bank_gather.py')
    return hashlib.sha256(b''.join(hashlib.sha256(Path(__file__).with_name(n).read_bytes()).digest() for n in names)).hexdigest()
