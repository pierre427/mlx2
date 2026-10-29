"""Whole-M/K-partition and compact arena CPU contracts; no GPU oracle."""
from dataclasses import replace

import numpy as np
import pytest

from mlx2.adapters.deepseek_v41_source_prefill import Job, plan_jobs
from mlx2.adapters.source_prefill_wide_plan import plan_wide_jobs, stock_partitions, partial_offset


def jobs_for(counts,kinds=None):
    jobs=[];cursor=0
    for e,count in enumerate(counts):
        locations=np.arange(cursor,cursor+count,dtype=np.uint32);locations.flags.writeable=False
        rows=locations//6;rows.flags.writeable=False
        jobs.append(Job(e,locations,rows,count,kinds[e]if kinds else'original'));cursor+=count
    return tuple(jobs)


@pytest.mark.parametrize('rows,p,tiles',[(13,5,1),(16,5,1),(31,5,1),(32,5,1),(33,2,2),(64,2,2),(65,2,3),(96,2,3),(97,1,4)])
def test_stock_partition_law_and_whole_row_band(rows,p,tiles):
    assert stock_partitions(rows)==p
    result=plan_wide_jobs(jobs_for([rows]))
    if rows==97:
        assert not result.batches and result.unhandled_wide_job_indices==(0,)
        return
    b=result.batches[0]
    assert b.row_counts==(rows,) and b.partitions==p and b.k_partition==5120//p
    assert b.m_tiles==tiles and b.partial_shape==(p,rows,2304)
    assert b.threadgroups==(72,tiles,p) and b.stock_threadgroup==(32,2,2)
    assert b.stock_threadgrid==(72*32,tiles*2,p*2)
    assert b.partition_stride==rows*2304 and b.partial_bytes==p*rows*2304*4
    assert b.gathered_input_bytes==rows*5120*4
    assert result.stats['threadgroup_scratch_bytes']==9216


def test_batches_share_only_compatible_p_and_row_tiles_and_restore_by_job_index():
    counts=[13,33,96,32,65,64,97,12,7]
    jobs=jobs_for(counts,['original']*8+['q1']);p=plan_wide_jobs(jobs)
    assert p.wide_job_indices==tuple(range(7))
    assert p.eligible_job_indices==tuple(range(6))
    assert p.unhandled_wide_job_indices==(6,)
    assert [b.job_indices for b in p.batches]==[(1,5),(2,4),(0,3)]
    for b in p.batches:
        expected=[jobs[i]for i in b.job_indices]
        assert b.row_counts==tuple(j.count for j in expected)
        assert np.array_equal(b.rows,np.concatenate([j.rows for j in expected]))
        assert np.array_equal(b.locations,np.concatenate([j.locations for j in expected]))
        assert all((j.count+31)//32==b.m_tiles for j in expected)
        assert b.partial_shape[1]==sum(j.count for j in expected)
        assert not b.rows.flags.writeable and not b.locations.flags.writeable
        assert not any(np.shares_memory(b.rows,j.rows)for j in expected)


def test_more_than15_backings_chunk_without_global_shape_padding_or_row_splits():
    jobs=jobs_for([13+(i%20)for i in range(31)])
    p=plan_wide_jobs(jobs)
    assert [len(b.job_indices)for b in p.batches]==[15,15,1]
    assert sorted(i for b in p.batches for i in b.job_indices)==list(range(31))
    assert p.stats['max_expert_backings']==15 and p.stats['max_planned_input_bindings']==20
    assert p.stats['scheduled_32row_slots']==31*32
    assert p.stats['active_row_slots']==sum(j.count for j in jobs)
    assert p.stats['gate_plus_up_partial_and_reduce_dispatches']==12
    assert p.stats['coefficient_repack_bytes']==0


def test_compact_partial_ownership_disjoint_and_same_as_virtual_stock_plane():
    p=plan_wide_jobs(jobs_for([13,31,32]));b=p.batches[0];seen=set()
    for expert,count in enumerate(b.row_counts):
        for part in range(b.partitions):
            for row in range(count):
                for col in (0,1,2302,2303):
                    offset=partial_offset(b,expert,part,row,col)
                    assert offset==part*b.partition_stride+(b.row_offsets[expert]+row)*2304+col
                    assert offset not in seen;seen.add(offset)
                    # The per-expert stock output index changes only its base
                    # and partition stride, never localrow/column ownership.
                    stock=part*count*2304+row*2304+col
                    assert offset-stock==part*(b.partition_stride-count*2304)+b.row_offsets[expert]*2304
    assert max(seen)<b.partial_bytes//4


def test_duplicate_route_occurrences_keep_exact_whole_expert_local_order():
    values=np.array([0]*13+[1]*5,np.uint32).reshape(3,6)
    source=plan_jobs(values,{0:'original',1:'q1'})
    wide=plan_wide_jobs(source.jobs);b=wide.batches[0]
    assert b.row_counts==(13,) and b.locations.tolist()==list(range(13))
    assert b.rows.tolist()==[0]*6+[1]*6+[2]


@pytest.mark.parametrize('bad',[0,16,True,2.5])
def test_invalid_batch_ceiling_refused(bad):
    with pytest.raises(ValueError):plan_wide_jobs(jobs_for([13]),bad)


@pytest.mark.parametrize('mutation',['kind','id','count','location_dtype','row_map','location_order','location_bound','not_array'])
def test_malformed_admitted_job_metadata_refused(mutation):
    job=jobs_for([13])[0]
    if mutation=='kind':job=replace(job,kind='other')
    elif mutation=='id':job=replace(job,expert_id=384)
    elif mutation=='count':job=replace(job,count=True)
    elif mutation=='location_dtype':job=replace(job,locations=job.locations.astype(np.int32))
    elif mutation=='row_map':job=replace(job,rows=job.rows+1)
    elif mutation=='location_order':job=replace(job,locations=job.locations[::-1],rows=job.rows[::-1])
    elif mutation=='not_array':job=replace(job,rows=job.rows.tolist())
    else:job=replace(job,locations=job.locations+6144,rows=job.rows+1024)
    with pytest.raises(ValueError):plan_wide_jobs([job])


def test_job_order_duplicate_ids_and_nonjobs_refused():
    jobs=jobs_for([13,14])
    for bad in ([jobs[1],jobs[0]],[jobs[0],jobs[0]],[{}]):
        with pytest.raises(ValueError):plan_wide_jobs(bad)


@pytest.mark.parametrize('coord',[(-1,0,0,0),(0,5,0,0),(0,0,13,0),(0,0,0,2304),(True,0,0,0)])
def test_partial_coordinate_refusal(coord):
    b=plan_wide_jobs(jobs_for([13])).batches[0]
    with pytest.raises(ValueError):partial_offset(b,*coord)


def test_empty_or_nonwide_plan_remains_explicit_and_bounded():
    for jobs in ((),jobs_for([1,12,3],['original','original','q1'])):
        p=plan_wide_jobs(jobs)
        assert not p.batches and not p.wide_job_indices
        assert p.stats['max_one_projection_partial_bytes']==0
        assert p.stats['row_slot_utilization']is None


def test_masked_rows_never_write_the_following_expert_segment():
    b=plan_wide_jobs(jobs_for([33,63,64])).batches[0]
    owned=set()
    for slot,m in enumerate(b.row_counts):
        for part in range(b.partitions):
            for tile in range(b.m_tiles):
                for row in range(tile*32,(tile+1)*32):
                    if row>=m:continue
                    index=partial_offset(b,slot,part,row,0)//2304
                    assert index not in owned;owned.add(index)
    assert owned==set(range(b.partitions*sum(b.row_counts)))
