"""CPU-only stable route coverage and bounded expert worklist planning."""
import numpy as np
import pytest

from mlx2.adapters.deepseek_v41_source_prefill import plan_jobs


@pytest.mark.parametrize('tokens',[2,7,99,1024])
@pytest.mark.parametrize('batch_size',[1,3,15])
def test_every_occurrence_owned_once_with_stable_expert_and_local_order(tokens,batch_size):
    values=np.random.default_rng(41025).integers(0,384,(tokens,6),dtype=np.uint32)
    kinds={i:'original'if i%2 else'q1'for i in range(384)}
    p=plan_jobs(values,kinds,batch_size)
    assert [j.expert_id for j in p.jobs]==sorted(set(values.ravel().tolist()))
    all_locations=np.concatenate([j.locations for j in p.jobs])
    assert np.array_equal(np.sort(all_locations),np.arange(values.size,dtype=np.uint32))
    assert len(set(all_locations.tolist()))==values.size
    for job in p.jobs:
        expected=np.flatnonzero(values.ravel()==job.expert_id).astype(np.uint32)
        assert np.array_equal(job.locations,expected)
        assert np.array_equal(job.rows,expected//6)
        assert job.count==len(expected) and job.kind==kinds[job.expert_id]
        for a in (job.locations,job.rows):
            assert a.dtype==np.uint32 and a.flags.c_contiguous and not a.flags.writeable
            assert not np.shares_memory(a,values)
    seen=[]
    for batch in p.batches:
        seen.extend(batch.job_indices)
        assert 1<=len(batch.job_indices)<=batch_size
        jobs=[p.jobs[i]for i in batch.job_indices]
        assert batch.row_counts==tuple(j.count for j in jobs)
        assert batch.kind==jobs[0].kind and all(j.kind==batch.kind for j in jobs)
        if batch.kind=='original':assert len(set(batch.row_counts))==1 and batch.row_counts[0]<=12
        assert batch.offsets==tuple(np.cumsum([0,*batch.row_counts]).tolist())
        assert np.array_equal(batch.locations,np.concatenate([j.locations for j in jobs]))
        assert np.array_equal(batch.rows,np.concatenate([j.rows for j in jobs]))
        assert not batch.locations.flags.writeable and not batch.rows.flags.writeable
    assert sorted(seen+list(p.fallback_job_indices))==list(range(len(p.jobs)))
    assert len(set(seen))==len(seen)
    s=p.stats
    assert s['maximum_input_bindings']<=20 and s['maximum_expert_bindings']<=batch_size
    assert s['numpy_metadata_nbytes']<=16*values.size
    assert s['batched_route_occurrences']+s['fallback_route_occurrences']==values.size
    assert s['maximum_batch_middle_nbytes']<=values.size*2304*4
    assert s['coefficient_repack_bytes']==0
    assert s['logical_input_gather_bytes_avoided']==s['batched_route_occurrences']*5120*4


def test_variable_count_q1_jobs_share_one_batch_and_duplicate_occurrences_remain():
    values=np.array([0]+[1]*3+[2]*7+[3],np.uint32).reshape(2,6)
    p=plan_jobs(values,{e:'q1'for e in range(4)})
    assert len(p.batches)==1 and p.batches[0].row_counts==(1,1,3,7)
    assert p.batches[0].offsets==(0,1,2,5,12)
    # Duplicate expert2 calls at one token remain distinct route occurrences.
    assert p.jobs[2].rows.tolist()==[0,0,1,1,1,1,1]


def test_original_row_laws_bucket_separately_and_wide_fallback_is_explicit():
    values=np.array([0]*13+[1]*3+[2]*2,np.uint32).reshape(3,6)
    p=plan_jobs(values,{0:'original',1:'original',2:'original'})
    assert p.fallback_job_indices==(0,)
    assert p.stats['fallback_reasons']=={0:'original_row_band_above12'}
    assert [b.row_counts for b in p.batches]==[(2,),(3,)]
    assert [b.job_indices for b in p.batches]==[(2,),(1,)]


def test_duplicate_heavy_q1_remains_lossless_with_explicit_overcap_fallback():
    values=np.zeros((171,6),np.uint32)
    p=plan_jobs(values,{0:'q1'})
    assert len(p.jobs)==1 and p.jobs[0].count==1026
    assert p.jobs[0].locations.tolist()==list(range(1026))
    assert not p.batches and p.fallback_job_indices==(0,)
    assert p.stats['fallback_reasons']=={0:'q1_occurrences_above1024'}
    assert p.stats['maximum_input_bindings']==0


def test_plan_does_not_alias_input_and_replanning_is_deterministic():
    values=np.arange(18,dtype=np.uint32).reshape(3,6)
    first=plan_jobs(values,{e:'q1'for e in range(18)},batch_size=3)
    second=plan_jobs(values,{e:'q1'for e in range(18)},batch_size=3)
    assert first.stats==second.stats
    assert [b.job_indices for b in first.batches]==[b.job_indices for b in second.batches]
    values.fill(7)
    assert [j.expert_id for j in first.jobs]==list(range(18))
    assert np.concatenate([j.locations for j in first.jobs]).tolist()==list(range(18))


@pytest.mark.parametrize('bad',[np.zeros((1,6),np.uint32),np.zeros((1025,6),np.uint32),
 np.zeros((2,5),np.uint32),np.zeros((2,6),np.int32),np.full((2,6),384,np.uint32),
 [[0]*6]*2,object()])
def test_invalid_cpu_route_input_refused(bad):
    with pytest.raises(ValueError,match='CPU uint32'):plan_jobs(bad,{0:'q1'})


@pytest.mark.parametrize('kinds',[{}, {0:'bogus'}, {0:None},[(0,'q1')]])
def test_missing_or_incompatible_kind_mapping_refused(kinds):
    with pytest.raises(ValueError):plan_jobs(np.zeros((2,6),np.uint32),kinds)


@pytest.mark.parametrize('batch_size',[0,16,True,1.5])
def test_invalid_batch_bound_refused(batch_size):
    with pytest.raises(ValueError,match='batch_size'):plan_jobs(np.zeros((2,6),np.uint32),{0:'q1'},batch_size)


def test_q1_count_bucketing_reduces_inactive_planes_without_changing_job_rows():
    counts=[1,99]*15
    values=np.concatenate([np.full(n,e,np.uint32)for e,n in enumerate(counts)]).reshape(250,6)
    p=plan_jobs(values,{e:'q1'for e in range(30)})
    assert p.batches[0].job_indices==tuple(range(0,30,2))
    assert p.batches[1].job_indices==tuple(range(1,30,2))
    assert p.batches[0].row_counts==(1,)*15 and p.batches[1].row_counts==(99,)*15
    assert p.stats['q1_scheduled_row_planes']==1500
    assert p.stats['q1_active_row_planes']==1500
    assert p.stats['q1_inactive_row_planes']==0
    baseline_scheduled=sum(len(counts[i:i+15])*max(counts[i:i+15])for i in (0,15))
    assert baseline_scheduled==2970
    for batch in p.batches:
        for i,a,b in zip(batch.job_indices,batch.offsets[:-1],batch.offsets[1:]):
            expected=np.flatnonzero(values.ravel()==i).astype(np.uint32)
            assert np.array_equal(batch.locations[a:b],expected)
            assert np.array_equal(batch.rows[a:b],expected//6)
