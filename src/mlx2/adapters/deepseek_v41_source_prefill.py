# SPDX-License-Identifier: MIT
"""Pure CPU route planning for explicit Source prefill worklists.

No device or coefficient arrays are created. Jobs retain ordinary ascending
expert order; batches may interleave it, so consumers restore by job_indices.
See provenance/deepseek-v41-source-prefill-plan.json.
"""
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Job:
    expert_id: int
    locations: np.ndarray
    rows: np.ndarray
    count: int
    kind: str


@dataclass(frozen=True)
class Batch:
    kind: str
    job_indices: tuple
    row_counts: tuple
    offsets: tuple
    rows: np.ndarray
    locations: np.ndarray


@dataclass(frozen=True)
class Plan:
    jobs: tuple
    batches: tuple
    fallback_job_indices: tuple
    stats: dict
    values: np.ndarray | None = None
    order: np.ndarray | None = None
    inverse: np.ndarray | None = None
    combine_offsets: np.ndarray | None = None
    expert_ids: np.ndarray | None = None
    starts: np.ndarray | None = None
    counts: np.ndarray | None = None


def _readonly(values):
    result=np.array(values,dtype=np.uint32,copy=True,order='C')
    result.flags.writeable=False
    return result


def _owned_readonly(values):
    """Freeze a newly constructed private U32 array without copying it again."""
    result=np.asarray(values,dtype=np.uint32,order='C')
    result.flags.writeable=False
    return result


def plan_jobs(values, kinds, batch_size=15, *, reuse=False):
    """Plan CPU uint32[T,6] routes without reading a device or weight object.

    ``kinds`` maps every called expert ID to ``original`` or ``q1``. Duplicate
    route occurrences remain distinct and stable. ``jobs`` are ascending ID;
    each batch concatenates complete jobs with matching original row laws.
    Fallback jobs keep the same representation for the ordinary provider.
    """
    if (not isinstance(values,np.ndarray) or values.dtype != np.uint32
            or values.ndim != 2 or values.shape[1] != 6 or not 2 <= values.shape[0] <= 1024
            or np.any(values>383)):
        raise ValueError('Source prefill planner requires CPU uint32[T,6],T2..1024,expert IDs0..383')
    if type(reuse) is not bool:
        raise ValueError('Source prefill planner reuse flag must be boolean')
    if not isinstance(kinds,Mapping) and not (reuse and callable(kinds)):
        raise ValueError('Source prefill planner requires CPU expert-kind mapping')
    if type(batch_size) is not int or not 1 <= batch_size <= 15:
        raise ValueError('Source prefill planner batch_size must be1..15')
    frame_values=frame_inverse=frame_offsets=frame_rows=None
    if reuse:
        # Own the frame input once; jobs borrow immutable slices of these two
        # flat owner arrays. Nothing here is cached across layers or forwards.
        values=frame_values=_readonly(values)
        flat=values.reshape(-1)
        order=_owned_readonly(np.argsort(flat,kind='stable'))
        sorted_ids=flat[order]
        starts=_owned_readonly(np.r_[0,np.flatnonzero(sorted_ids[1:]!=sorted_ids[:-1])+1])
        unique=_owned_readonly(sorted_ids[starts])
        counts=_owned_readonly(np.diff(np.r_[starts,order.size]))
        frame_rows=_owned_readonly(order//6)
        frame_inverse=np.empty(order.size,np.uint32)
        frame_inverse[order]=np.arange(order.size,dtype=np.uint32)
        frame_inverse.flags.writeable=False
        # Global sorted positions already encode ascending expert IDs and
        # stable duplicate-route order; no second per-token argsort is needed.
        frame_offsets=_owned_readonly(np.sort(frame_inverse.reshape(values.shape),axis=1))
    else:
        flat=values.reshape(-1)
        order=np.argsort(flat,kind='stable')
        sorted_ids=flat[order]
        unique,starts,counts=np.unique(sorted_ids,return_index=True,return_counts=True)
    histogram_counts=counts if reuse else None
    jobs=[]
    buckets={};fallback=[];fallback_reasons={}
    for expert,start,count in zip(unique,starts,counts):
        expert=int(expert);start=int(start);count=int(count)
        kind=kinds(expert) if callable(kinds) and not isinstance(kinds,Mapping) else kinds.get(expert)
        if kind not in ('original','q1'):
            raise ValueError(f'Source prefill planner missing/invalid format for expert{expert}')
        locations=order[start:start+count] if reuse else _readonly(order[start:start+count])
        rows=frame_rows[start:start+count] if reuse else _readonly(locations//6)
        index=len(jobs);jobs.append(Job(expert,locations,rows,count,kind))
        if (kind=='original' and count>12) or (kind=='q1' and count>1024):
            fallback.append(index)
            fallback_reasons[expert]='original_row_band_above12' if kind=='original' else 'q1_occurrences_above1024'
        else:
            buckets.setdefault((kind,count if kind=='original' else 0),[]).append(index)
    batches=[]
    for key,indices in sorted(buckets.items()):
        if key[0]=='q1':
            # Q1 has no cross-row arithmetic. Group similar local counts to
            # reduce guarded inactive grid.y planes, then restore by job index.
            indices=sorted(indices,key=lambda i:(jobs[i].count,i))
        for start in range(0,len(indices),batch_size):
            selected=tuple(indices[start:start+batch_size]);local=tuple(jobs[i]for i in selected)
            counts=tuple(job.count for job in local);offsets=[0]
            for count in counts:offsets.append(offsets[-1]+count)
            freeze=_owned_readonly if reuse else _readonly
            batches.append(Batch(key[0],selected,counts,tuple(offsets),
                freeze(np.concatenate([job.rows for job in local])),
                freeze(np.concatenate([job.locations for job in local]))))
    eligible=sum(sum(batch.row_counts)for batch in batches)
    array_bytes=sum(job.locations.nbytes+job.rows.nbytes for job in jobs)+sum(
        batch.rows.nbytes+batch.locations.nbytes for batch in batches)
    q1_batches=[b for b in batches if b.kind=='q1']
    q1_scheduled=sum(len(b.job_indices)*max(b.row_counts)for b in q1_batches)
    q1_active=sum(sum(b.row_counts)for b in q1_batches)
    stats={'token_rows':int(values.shape[0]),'route_occurrences':int(values.size),
        'expert_jobs':len(jobs),'batched_jobs':sum(len(b.job_indices)for b in batches),
        'fallback_jobs':len(fallback),'fallback_reasons':fallback_reasons,
        'gu_worklist_launches':len(batches),'ordinary_gu_jobs_replaced':sum(len(b.job_indices)for b in batches),
        'batched_route_occurrences':eligible,'fallback_route_occurrences':int(values.size)-eligible,
        'maximum_expert_bindings':max((len(b.job_indices)for b in batches),default=0),
        'maximum_input_bindings':max((len(b.job_indices)+5 for b in batches),default=0),
        'numpy_metadata_nbytes':int(array_bytes),
        'maximum_batch_route_metadata_nbytes':max((b.rows.nbytes+b.locations.nbytes+4*len(b.offsets)for b in batches),default=0),
        'maximum_batch_middle_nbytes':max((sum(b.row_counts)*2304*4 for b in batches),default=0),
        'logical_input_gather_bytes_avoided':eligible*5120*4,
        'q1_scheduled_row_planes':q1_scheduled,'q1_active_row_planes':q1_active,
        'q1_inactive_row_planes':q1_scheduled-q1_active,
        'coefficient_repack_bytes':0}
    if reuse:
        return Plan(tuple(jobs),tuple(batches),tuple(fallback),stats,frame_values,
                    order,frame_inverse,frame_offsets,unique,starts,histogram_counts)
    return Plan(tuple(jobs),tuple(batches),tuple(fallback),stats)
