"""Synthetic CPU packet identities and ownership; no model/GPU claim."""
from dataclasses import replace

import numpy as np
import pytest

from mlx2.adapters.deepseek_v41_source_prefill import plan_jobs
from mlx2.adapters.deepseek_v41_source_row_packet import (
    prepare_row_packet, implementation_identity, DeviceRowPacket)


def _plan(tokens=7, seed=4):
    values = np.random.default_rng(seed).integers(0, 384, (tokens, 6), dtype=np.uint32)
    return plan_jobs(values, {i: 'q1' if i % 2 else 'original' for i in range(384)}, reuse=True)


def _packet(plan=None, **kwargs):
    return prepare_row_packet(plan or _plan(), frame_key='request1:forward2', generation=3,
                              layer=4, **kwargs)


@pytest.mark.parametrize('tokens', [2, 7, 99, 1024])
@pytest.mark.parametrize('seed', [0, 17, 41])
def test_one_immutable_backing_has_lossless_views_and_cpu_oracle(tokens, seed):
    plan = _plan(tokens, seed)
    packet = _packet(plan)
    assert not packet.backing.flags.writeable
    with pytest.raises(ValueError):
        packet.backing.flags.writeable = True
    assert packet.view('values').tolist() == plan.values.tolist()
    assert packet.view('order').tolist() == plan.order.tolist()
    assert packet.view('combine_offsets').tolist() == plan.combine_offsets.tolist()
    for name in packet.fields:
        a = packet.view(name)
        assert not a.flags.writeable
        if a.size:
            assert np.shares_memory(a, packet.backing)
    # Independent route oracle walks original tokens and route slots.
    entries = [(int(e), t, s) for t, row in enumerate(plan.values) for s, e in enumerate(row)]
    entries.sort(key=lambda item: item[0])
    assert packet.view('rows').tolist() == [t for _, t, _ in entries]
    assert packet.view('route_slots').tolist() == [s for _, _, s in entries]
    for j, desc in zip(plan.jobs, packet.view('jobs')):
        expert, kind, start, count, virtual_m = map(int, desc)
        assert (expert, kind, count, virtual_m) == (j.expert_id, int(j.kind == 'q1'), j.count, j.count)
        assert packet.view('order')[start:start + count].tolist() == j.locations.tolist()
    assert packet.receipt()['coefficient_repack_bytes'] == 0
    assert packet.receipt()['activation_repack_bytes'] == 0


def test_u64_positions_and_generation_do_not_truncate():
    p = prepare_row_packet(_plan(), frame_key='x', generation=2**63 + 4, layer=39,
                           position_start=2**32 - 3)
    positions = p.view('position_low').astype(np.uint64) + (p.view('position_high').astype(np.uint64) << 32)
    assert positions.tolist() == list(range(2**32 - 3, 2**32 + 4))
    assert p.view('header')[2:4].tolist() == [4, 2**31]


class Backend:
    uint32 = np.uint32

    def __init__(self):
        self.uploads = 0

    def array(self, value, *, dtype):
        self.uploads += 1
        return np.array(value, dtype=dtype, copy=True)


def test_single_upload_all_provider_batch_job_views_share_it():
    p = _packet(_plan(99))
    backend = Backend()
    device = p.upload(backend)
    for i, batch in enumerate(p.plan.batches):
        rows, locations = device.batch(i)
        again_rows, again_locations = device.batch(i)
        assert again_rows is rows and again_locations is locations
        assert np.array_equal(rows, batch.rows)
        assert np.array_equal(locations, batch.locations)
        assert np.shares_memory(rows, device.root)
    for i, job in enumerate(p.plan.jobs):
        rows, locations = device.job(i)
        assert rows.tolist() == job.rows.tolist()
        assert locations.tolist() == job.locations.tolist()
    assert backend.uploads == 1


def test_independent_outputs_retained_through_successful_fence():
    d = _packet().upload(Backend())
    a, b = object(), object()
    d.retain(a)
    d.retain(b, a)
    assert d.completion_roots()[0] is d.root
    assert d.completion_roots()[1:] == (a, b)
    called = []
    d.complete(lambda *roots: called.append(roots))
    assert called[0][1:] == (a, b)
    assert d.root is None
    with pytest.raises(ValueError, match='retired'):
        d.view('rows')


def test_failed_fence_keeps_root_and_every_consumer_then_can_retry():
    d = _packet().upload(Backend())
    root = d.root
    a, b = object(), object()
    d.retain(a, b)
    def fail(*args):
        raise RuntimeError('pending GPU failure simulated on CPU')
    with pytest.raises(RuntimeError):
        d.complete(fail)
    assert d.root is root
    with pytest.raises(ValueError, match='pending'):
        d.close_unused()
    seen = []
    d.complete(lambda *roots: seen.extend(roots))
    assert seen[1:] == [a, b]


def test_unused_can_close_but_empty_completion_is_refused():
    d = _packet().upload(Backend())
    with pytest.raises(ValueError, match='no submitted'):
        d.complete(lambda *x: None)
    d.close_unused()
    with pytest.raises(ValueError, match='retired'):
        d.retain(object())


@pytest.mark.parametrize('field,value', [('frame_key','other'), ('generation',4), ('layer',5)])
def test_stale_binding_refused(field, value):
    p = _packet()
    args = dict(frame_key=p.frame_key, generation=p.generation, layer=p.layer)
    args[field] = value
    with pytest.raises(ValueError, match='stale'):
        p.validate(p.plan, **args)


def test_different_plan_object_cannot_reuse_packet_even_with_identical_routes():
    p = _packet()
    with pytest.raises(ValueError, match='owner'):
        p.validate(_plan(), frame_key=p.frame_key, generation=p.generation, layer=p.layer)


@pytest.mark.parametrize('mutation', ['order', 'inverse', 'combine', 'jobs', 'batch', 'fallback'])
def test_corrupt_authoritative_metadata_rejected(mutation):
    p = _plan()
    if mutation == 'order':
        a = p.order.copy(); a[:2] = a[:2][::-1]; a.flags.writeable = False
        p = replace(p, order=a)
    elif mutation == 'inverse': p = replace(p, inverse=np.zeros_like(p.inverse))
    elif mutation == 'combine': p = replace(p, combine_offsets=np.zeros_like(p.combine_offsets))
    elif mutation == 'jobs': p = replace(p, jobs=(replace(p.jobs[0], count=999), *p.jobs[1:]))
    elif mutation == 'batch': p = replace(p, batches=(replace(p.batches[0], offsets=(0,)), *p.batches[1:]))
    elif mutation == 'fallback': p = replace(p, fallback_job_indices=(0,))
    with pytest.raises(ValueError): _packet(p)


@pytest.mark.parametrize('kwargs', [dict(generation=True), dict(generation=-1), dict(generation=2**64),
                                  dict(layer=40), dict(layer=False), dict(frame_key=''),
                                  dict(frame_key='x'*257), dict(position_start=2**64),
                                  dict(max_words=0), dict(max_words=True), dict(max_words=8)])
def test_bad_bounds_and_capacity_fail_before_backend_upload(kwargs):
    args = dict(frame_key='x', generation=1, layer=0)
    args.update(kwargs)
    with pytest.raises(ValueError): prepare_row_packet(_plan(), **args)


def test_non_reused_plan_refuses_and_input_mutation_does_not_change_packet():
    values = np.arange(12, dtype=np.uint32).reshape(2,6)
    kinds = {i:'q1' for i in range(12)}
    with pytest.raises(ValueError): _packet(plan_jobs(values, kinds))
    plan = plan_jobs(values, kinds, reuse=True)
    packet = _packet(plan)
    values.fill(7)
    assert packet.view('values').ravel().tolist() == list(range(12))


def test_duplicate_expert_occurrences_remain_distinct_routes():
    values = np.array([[3,1,3,2,1,3], [1,2,3,1,2,3]], np.uint32)
    p = _packet(plan_jobs(values, {i:'q1' for i in (1,2,3)}, reuse=True))
    assert sorted(p.view('order').tolist()) == list(range(12))
    assert p.view('route_slots').size == 12
    assert p.view('jobs')[:,3].tolist() == [4,3,5]


def test_profile_identity_is_a_cached_source_digest_without_backend_access():
    implementation_identity.cache_clear()
    first = implementation_identity()
    assert len(first) == 64 and int(first, 16) >= 0
    assert implementation_identity() == first
    assert implementation_identity.cache_info().hits == 1


@pytest.mark.parametrize('name,value', [('frame_key','other'), ('generation',8),
                                      ('layer',9), ('plan',None), ('backing',None)])
def test_factory_packet_attributes_cannot_be_replaced(name,value):
    with pytest.raises(AttributeError,match='immutable'):
        setattr(_packet(),name,value)


def test_device_packet_factory_and_owner_are_not_publicly_replaceable():
    p=_packet()
    with pytest.raises(ValueError,match='factory uploaded'):
        DeviceRowPacket(p,np.zeros_like(p.backing))
    d=p.upload(Backend())
    with pytest.raises(AttributeError):d.root=np.zeros_like(p.backing)
    with pytest.raises(AttributeError):d.packet=None


@pytest.mark.parametrize('name',['values','order','inverse','combine_offsets'])
def test_modified_plan_even_after_refreezing_is_not_the_same_semantic_packet(name):
    p=_packet()
    a=getattr(p.plan,name)
    a.flags.writeable=True
    a.ravel()[0]=np.uint32((int(a.ravel()[0])+1)%384)
    a.flags.writeable=False
    binding=dict(frame_key=p.frame_key,generation=p.generation,layer=p.layer)
    # The cheaper identity-only check is explicit, and cannot certify data.
    p.validate(p.plan,contents=False,**binding)
    with pytest.raises(ValueError,match='metadata changed'):
        p.validate(p.plan,**binding)


def test_modified_batch_metadata_cannot_reuse_old_uploaded_rows():
    p=_packet()
    a=p.plan.batches[0].rows
    a.flags.writeable=True;a[0]=999;a.flags.writeable=False
    with pytest.raises(ValueError,match='batch metadata changed'):
        p.validate_contents()


@pytest.mark.parametrize('bad',[None,np.zeros((1,),np.uint32),np.zeros((1,),np.float32)])
def test_backend_upload_must_preserve_the_exact_packet_geometry(bad):
    class BadBackend(Backend):
        def array(self,*args,**kwargs):return bad
    with pytest.raises(ValueError,match='upload geometry'):
        _packet().upload(BadBackend())
