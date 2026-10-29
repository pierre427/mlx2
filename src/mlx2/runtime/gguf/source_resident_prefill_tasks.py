# SPDX-License-Identifier: MIT
"""Explicit bounded PP task striding, preserving each original task body.

Provenance: deepseek-v41-source-resident-prefill.json. GU uses eight task
stripes and DOWN two. A stripe walks the GPU task count without host readback.
An end-of-task barrier protects reused threadgroup scratch; no arithmetic,
row grouping, coefficient packing, partial fold or score position is changed.
"""
from functools import lru_cache
import hashlib
from pathlib import Path

POLICIES=('padded_v1','persistent_8_2_v1')
POLICY='persistent_8_2_v1'


def validate_policy(policy):
    if policy not in POLICIES:raise ValueError('Invalid resident PP task policy')
    return policy


def implementation_identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def task_grid(policy,which,maximum):
    validate_policy(policy)
    if type(maximum)is not int or maximum<=0:raise ValueError('Invalid PP task maximum')
    if which not in ('gu','down','gu_wide','down_wide'):raise ValueError('Unknown PP task stage')
    return maximum if policy=='padded_v1'else min(maximum,8 if which in ('gu','gu_wide')else 2)


def persistent_spec(which,key,spelling='metal::exp'):
    from . import source_resident_prefill as narrow
    if which in ('gu','down'):
        original=(narrow.gu_spec(key,spelling)if which=='gu'else narrow.down_spec(key))
        prefix='const uint task=threadgroup_position_in_grid.z;if(task>=task_count[0])return;'
        variable,limit='task','task_count[0]'
    else:raise ValueError('Unknown persistent PP stage')
    return _loop(original,prefix,variable,limit,which)


def wide_persistent_spec(which,key,rows):
    from .source_resident_prefill_wide import projection_spec
    if which not in ('gu_wide','down_wide'):raise ValueError('Unknown wide persistent PP stage')
    original=projection_spec(key,rows,down=which=='down_wide')
    limit='counts[2]'if which=='down_wide'else'counts[1]'
    prefix=f'const uint job=threadgroup_position_in_grid.z;if(job>={limit})return;'
    return _loop(original,prefix,'job',limit,which)


def _loop(original,prefix,variable,limit,which):
    source=original['source']
    if source.count(prefix)!=1:raise RuntimeError('Pinned PP task prefix changed')
    stride=8 if which in ('gu','gu_wide')else 2
    wrapper=(f'for(uint {variable}=threadgroup_position_in_grid.z;{variable}<{limit};'
             f'{variable}+={stride}u){{')
    # Preserve the complete original body byte for byte after its task guard.
    # Each loop iteration has independent accumulators. All threads in a stripe
    # have the same job/count, and finish scratch reads before its next task.
    return dict(original,source=source.replace(prefix,wrapper)+
        '\nthreadgroup_barrier(mem_flags::mem_threadgroup);\n}\n')


@lru_cache(maxsize=128)
def kernel(which,key,rows,spelling):
    import mlx.core as mx
    spec=(wide_persistent_spec(which,key,rows)if which.endswith('_wide')else persistent_spec(which,key,spelling))
    spec={name:spec[name]for name in ('input_names','output_names','header','source')}
    digest=hashlib.sha256(repr((POLICY,which,key,rows,spelling)).encode()).hexdigest()[:16]
    return mx.fast.metal_kernel(name='mlx2_source_pp_tasks_'+which+'_'+digest,
        ensure_row_contiguous=False,**spec)
