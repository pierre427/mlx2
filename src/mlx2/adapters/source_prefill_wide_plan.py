# SPDX-License-Identifier: MIT
"""Pure CPU whole-expert Source wide-original tile planning.

This module creates metadata only. It does not select or run a kernel. Every
expert keeps its complete M, original mapped row order and stock K partition.
See provenance/deepseek-v41-source-wide-prefill-plan.json.
"""
from dataclasses import dataclass

import numpy as np

from .deepseek_v41_source_prefill import Job

K, N, BM, BK, BN, WM, WN = 5120, 2304, 32, 32, 32, 2, 2
MAX_EXPERTS = 15


@dataclass(frozen=True)
class WideBatch:
    job_indices: tuple
    row_counts: tuple
    row_offsets: tuple
    partitions: int
    k_partition: int
    m_tiles: int
    rows: np.ndarray
    locations: np.ndarray
    partial_shape: tuple
    partition_stride: int
    partial_bytes: int
    gathered_input_bytes: int
    threadgroups: tuple
    stock_threadgroup: tuple
    stock_threadgrid: tuple
    planned_input_binding_upper_bound: int


@dataclass(frozen=True)
class WidePlan:
    batches: tuple
    wide_job_indices: tuple
    eligible_job_indices: tuple
    unhandled_wide_job_indices: tuple
    stats: dict


def stock_partitions(rows):
    if type(rows) is not int or not 13 <= rows <= 1024:
        raise ValueError('Source wide planner requires whole-expert M13..1024')
    p=min(K//32,max(1,512//(((rows+31)//32)*(N//32))))
    while p>1 and K%(p*32):p-=1
    return p


def _array(values):
    a=np.array(values,dtype=np.uint32,copy=True,order='C');a.flags.writeable=False
    return a


def plan_wide_jobs(jobs, batch_size=MAX_EXPERTS):
    """Consume admitted CPU Job objects; group only original M13..96.

    ``job_indices`` always index the original ascending-expert Job sequence.
    One whole expert may have several32-row tiles, but its M and K reduction
    are not changed. The partial arena is partition-major, expert-row compact.
    The binding count is a planned upper bound, not an implemented kernel ABI.
    """
    if type(batch_size) is not int or not 1 <= batch_size <= MAX_EXPERTS:
        raise ValueError('Source wide planner batch_size must be1..15')
    jobs=tuple(jobs)
    if any(not isinstance(j,Job) for j in jobs):
        raise ValueError('Source wide planner requires admitted CPU Job objects')
    if any(type(j.expert_id)is not int or not 0<=j.expert_id<384 or j.kind not in ('original','q1')for j in jobs):
        raise ValueError('Source wide planner expert identity/format mismatch')
    if any(a.expert_id>=b.expert_id for a,b in zip(jobs,jobs[1:])):
        raise ValueError('Source wide planner jobs must retain ascending unique expert order')
    wide=[];eligible=[];unhandled=[];buckets={}
    for index,job in enumerate(jobs):
        if type(job.count)is not int or job.count<1:
            raise ValueError('Source wide planner local row count must be positive integer')
        if job.kind!='original' or job.count<=12:continue
        if (type(job.count)is not int or not isinstance(job.locations,np.ndarray)
                or not isinstance(job.rows,np.ndarray) or job.locations.dtype!=np.uint32 or job.rows.dtype!=np.uint32
                or job.locations.shape!=(job.count,) or job.rows.shape!=(job.count,)
                or not np.array_equal(job.rows,job.locations//6)
                or np.any(job.locations[1:]<=job.locations[:-1])
                or np.any(job.locations>=6144) or np.any(job.rows>=1024)):
            raise ValueError('Source wide planner local occurrence geometry mismatch')
        wide.append(index)
        if job.count>96:
            unhandled.append(index);continue
        p=stock_partitions(job.count)
        if p not in (2,5):raise ValueError('Source wide planner unproved partition law')
        eligible.append(index);buckets.setdefault((p,(job.count+31)//32),[]).append(index)
    batches=[]
    for (p,tiles),indices in sorted(buckets.items()):
        for start in range(0,len(indices),batch_size):
            selected=tuple(indices[start:start+batch_size]);local=[jobs[i]for i in selected]
            counts=tuple(j.count for j in local);offsets=[0]
            for count in counts:offsets.append(offsets[-1]+count)
            total=offsets[-1];groups=(N//BN,tiles,p*len(selected));tg=(32,2,2)
            batches.append(WideBatch(selected,counts,tuple(offsets),p,K//p,tiles,
                _array(np.concatenate([j.rows for j in local])),
                _array(np.concatenate([j.locations for j in local])),
                (p,total,N),total*N,p*total*N*4,total*K*4,
                groups,tg,tuple(a*b for a,b in zip(groups,tg)),len(selected)+5))
    active=sum(jobs[i].count for i in eligible)
    padded=sum(len(b.job_indices)*b.m_tiles*BM for b in batches)
    all_wide_rows=sum(jobs[i].count for i in wide)
    stats={'all_expert_jobs':len(jobs),'original_wide_jobs':len(wide),
        'original_wide_route_rows':all_wide_rows,'eligible_jobs':len(eligible),
        'eligible_route_rows':active,'unhandled_wide_jobs':len(unhandled),
        'unhandled_wide_route_rows':all_wide_rows-active,
        'one_projection_batches':len(batches),'gate_plus_up_partial_dispatches':2*len(batches),
        'gate_plus_up_partial_and_reduce_dispatches':4*len(batches),
        'scheduled_32row_slots':padded,'active_row_slots':active,'masked_row_slots':padded-active,
        'row_slot_utilization':active/padded if padded else None,
        'max_expert_backings':max((len(b.job_indices)for b in batches),default=0),
        'max_planned_input_bindings':max((b.planned_input_binding_upper_bound for b in batches),default=0),
        'max_one_projection_partial_bytes':max((b.partial_bytes for b in batches),default=0),
        'max_gathered_input_bytes':max((b.gathered_input_bytes for b in batches),default=0),
        'one_projection_partial_bytes_summed_over_batches':sum(b.partial_bytes for b in batches),
        'gathered_input_bytes_summed_over_batches':sum(b.gathered_input_bytes for b in batches),
        'threadgroup_scratch_bytes':2*BM*(BK+4)*4,
        'coefficient_repack_bytes':0,
        'scope':'CPU arithmetic bounds only; dispatches assume separate one-projection partial/reduce kernels; not observed GPU traffic or speed'}
    return WidePlan(tuple(batches),tuple(wide),tuple(eligible),tuple(unhandled),stats)


def partial_offset(batch, expert_slot, partition, local_row, column):
    """F32 element offset in the proposed compact partition-major arena."""
    for value in (expert_slot,partition,local_row,column):
        if type(value)is not int:raise ValueError('Source wide partial coordinate must be integer')
    if (not 0<=expert_slot<len(batch.row_counts) or not 0<=partition<batch.partitions
            or not 0<=local_row<batch.row_counts[expert_slot] or not 0<=column<N):
        raise ValueError('Source wide partial coordinate outside arena')
    return partition*batch.partition_stride+(batch.row_offsets[expert_slot]+local_row)*N+column
