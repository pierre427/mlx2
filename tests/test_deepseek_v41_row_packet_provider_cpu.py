"""Actual Source provider/packet/GU/combine orchestration with CPU dot tags.

Only projection kernels are replaced; real planning, one upload, GU offsets,
ordered-combine wrapper and provider completion windows execute on CPU.
No checkpoint, Metal compiler or GPU is used.
"""
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from test_deepseek_v41_source_prefill_integration_cpu import case, expected
from mlx2.adapters import deepseek_v41_source_math as assembly
from mlx2.adapters import deepseek_v41_source_row_packet as packets
from mlx2.adapters import deepseek_v41_mlx_weights as native
from mlx2.runtime.gguf import source_gate_up_prefill as gu


def _provider(monkeypatch, ids, *, selected=True, owned=True, kind='mixed', begin=True):
    real_gu, real_combine = gu.pp_middle, assembly.ordered_combine
    c=case(monkeypatch,ids,owned=owned,kind=kind)
    c.w.source_frame_reuse_policy='prefill_v1'
    c.w.source_row_packet_policy='row_packet_v1' if selected else 'reference'
    c.store.observe_routes=lambda prefix,values,**kw:c.observed.append((prefix,values.copy(),kw))
    if begin:c.w.begin_source_frame(101,len(c.ids))
    trace=SimpleNamespace(uploads=[],u32_uploads=[],gu=[],combine=[],evals=[],packet_validations=[])
    old_array, old_eval=mx.array,mx.eval
    def array(value,*args,**kwargs):
        dtype=kwargs.get('dtype',args[0] if args else None)
        if dtype is not None and dtype==mx.uint32:trace.u32_uploads.append(value)
        return old_array(value,*args,**kwargs)
    monkeypatch.setattr(mx,'array',array)
    def evaluate(*outputs):
        trace.evals.append(outputs)
        return old_eval(*outputs)
    monkeypatch.setattr(mx,'eval',evaluate)
    old_upload=packets.RowPacket.upload
    def upload(packet,backend):
        result=old_upload(packet,backend)
        trace.uploads.append(result)
        return result
    monkeypatch.setattr(packets.RowPacket,'upload',upload)
    old_validate=packets.RowPacket.validate
    def validate(packet,*args,**kwargs):
        trace.packet_validations.append(kwargs.get('contents',True))
        return old_validate(packet,*args,**kwargs)
    monkeypatch.setattr(packets.RowPacket,'validate',validate)
    monkeypatch.setattr(packets.DeviceRowPacket,'complete',
                        lambda *a,**kw:pytest.fail('provider must not add packet completion fence'))
    def gu_kernel(key,exp_spelling,return_projections):
        assert not return_projections
        def project(*,inputs,**kwargs):
            # Actual pp_middle prepared these inputs and used its offset branch.
            activation,rows,scores,offsets,inverse,*backings=inputs
            eids=[next(e for e,obj in c.bank.items() if obj.backing is b)for b in backings]
            counts=np.diff(np.array(offsets)).tolist()
            trace.gu.append((rows,scores,offsets,tuple(eids),tuple(counts)))
            pieces=[]
            stops=np.array(offsets).tolist()
            for e,a,b in zip(eids,stops[:-1],stops[1:]):
                pieces.append(mx.repeat(((e+1)*scores[a:b])[:,None],2304,axis=1))
            return [mx.concatenate(pieces,axis=0)]
        return project
    monkeypatch.setattr(gu,'_kernel',gu_kernel)
    monkeypatch.setattr(gu,'pp_middle',real_gu)
    def combine_kernel(direct_count):
        assert direct_count==0
        def combine(*,inputs,template,**kwargs):
            packed,offsets=inputs
            trace.combine.append(offsets)
            shape=dict(template)
            result=[]
            for row in range(shape['ROWS']):
                total=mx.zeros((shape['WIDTH'],),mx.float32)
                for location in np.array(offsets)[row]:total=total+packed[int(location)]
                result.append(total)
            return [mx.stack(result)]
        return combine
    monkeypatch.setattr(assembly,'_combine_kernel',combine_kernel)
    monkeypatch.setattr(assembly,'ordered_combine',real_combine)
    return c,trace


@pytest.mark.parametrize('owned',[True,False])
@pytest.mark.parametrize('ids,kind',[
    ([[9,0,7,2,6,1],[9,0,7,6,5,1],[9,0,7,6,5,3]],'mixed'),
    (np.arange(36,dtype=np.uint32).reshape(6,6)[:,::-1].copy(),'q1'),
    (np.array([0]*13+[1]*3+[2]*2,np.uint32).reshape(3,6),'mixed'),
])
def test_actual_provider_one_upload_all_maps_and_same_reference_output(monkeypatch,owned,ids,kind):
    results=[];controls=[]
    for selected in (False,True):
        with monkeypatch.context() as patch, mx.stream(mx.cpu):
            c,trace=_provider(patch,ids,selected=selected,owned=owned,kind=kind)
            output=c.w.moe(c.x,c.ids,c.scores,'blk.0')
            result=np.array(output)
            assert result.tobytes()==expected(c).tobytes()
            assert len(trace.evals)==int(owned)
            assert len(c.fences)==int(owned)
            if selected:
                assert len(trace.uploads)==len(trace.u32_uploads)==1
                device=trace.uploads[0]
                assert trace.u32_uploads[0] is device.packet.backing
                for i,(rows,scores,offsets,eids,counts) in enumerate(trace.gu):
                    assert rows is device.view(f'batch{i}.rows')
                    assert offsets is device.view(f'batch{i}.offsets')
                    assert np.array_equal(np.array(rows),device.packet.view(f'batch{i}.rows'))
                    assert np.array_equal(np.array(offsets),device.packet.view(f'batch{i}.offsets'))
                    locations=device.packet.view(f'batch{i}.locations')
                    assert np.array_equal(np.array(scores),np.array(c.scores).ravel()[locations])
                assert np.array_equal(np.array(trace.combine[0]),device.packet.view('combine_offsets'))
                assert trace.combine[0] is device.view('combine_offsets')
                assert c.w.scheduled_expert_calls['source_row_packet_uploads']==1
                assert trace.packet_validations
                assert device.root is not None  # Existing graph completion owns immutable input lifetime.
            else:
                assert not trace.uploads
            results.append(result)
            controls.append((len(trace.evals),len(c.fences)))
            c.w.end_source_frame()
    assert results[0].tobytes()==results[1].tobytes()
    assert controls[0]==controls[1]


@pytest.mark.parametrize('ended',[False,True])
def test_missing_or_ended_forward_refuses_before_acquire(monkeypatch,ended):
    with mx.stream(mx.cpu):
        c,trace=_provider(monkeypatch,[[0,1,2,3,4,5]]*2,begin=ended)
        if ended:c.w.end_source_frame()
        with pytest.raises(ValueError,match='forward frame'):
            c.w.moe(c.x,c.ids,c.scores,'blk.0')
        assert not c.acquired and not trace.uploads and not trace.gu


def test_nested_frame_and_wrong_row_count_refused(monkeypatch):
    with mx.stream(mx.cpu):
        c,trace=_provider(monkeypatch,[[0,1,2,3,4,5]]*2)
        with pytest.raises(ValueError,match='already active'):c.w.begin_source_frame(200,2)
        c.w.end_source_frame();c.w.begin_source_frame(200,3)
        with pytest.raises(ValueError,match='forward frame'):c.w.moe(c.x,c.ids,c.scores,'blk.0')
        assert not c.acquired and not trace.uploads


@pytest.mark.parametrize('change_at',['acquire','gu'])
def test_generation_change_mid_provider_refuses_before_combine(monkeypatch,change_at):
    with mx.stream(mx.cpu):
        c,trace=_provider(monkeypatch,[[0,1,2,3,4,5]]*2)
        if change_at=='acquire':
            old=c.store.acquire
            def acquire(prefix,e):
                result=old(prefix,e)
                if e==max(c.bank):c.w._source_current_frame=None
                return result
            c.store.acquire=acquire
        else:
            old=gu.pp_middle
            def middle(*args,**kwargs):
                result=old(*args,**kwargs)
                c.w._source_current_frame=None
                return result
            monkeypatch.setattr(gu,'pp_middle',middle)
        with pytest.raises(ValueError,match='stale model forward'):
            c.w.moe(c.x,c.ids,c.scores,'blk.0')
        assert not trace.combine and not trace.evals
        assert len(c.fences)==1 and not c.store._active and not c.store._leased


@pytest.mark.parametrize('owned',[True,False])
@pytest.mark.parametrize('failure',['gu','combine'])
def test_provider_failure_uses_existing_completion_and_retains_external_leases(monkeypatch,owned,failure):
    with mx.stream(mx.cpu):
        c,trace=_provider(monkeypatch,[[0,1,2,3,4,5]]*2,owned=owned)
        def fail(*args,**kwargs):
            assert c.store._active and c.store._leased
            c.store._pending_host['sentinel']=object()
            c.store._pending_uploads['sentinel']={}
            raise RuntimeError('injected '+failure)
        if failure=='gu':monkeypatch.setattr(gu,'pp_middle',fail)
        else:monkeypatch.setattr(assembly,'ordered_combine',fail)
        with pytest.raises(RuntimeError,match='injected'):c.w.moe(c.x,c.ids,c.scores,'blk.0')
        assert len(trace.uploads)==1 and trace.uploads[0].root is not None
        assert not trace.evals
        if owned:
            assert len(c.fences)==1 and not c.store._leased and not c.store._active
            assert not c.store._pending_host and not c.store._pending_uploads
        else:
            assert not c.fences and c.store._leased and c.store._active
            assert c.store._pending_host and c.store._pending_uploads


def test_failed_existing_fence_quarantines_all_expert_owners(monkeypatch):
    with mx.stream(mx.cpu):
        c,trace=_provider(monkeypatch,[[0,1,2,3,4,5]]*2)
        def bad_fence():
            raise RuntimeError('injected completion failure')
        c.store._fence=bad_fence
        try:
            with pytest.raises(RuntimeError,match='completion failure'):
                c.w.moe(c.x,c.ids,c.scores,'blk.0')
            assert c.store._active and len(c.store._leased)==len(c.bank)
            assert id(c.store) in native._UPLOAD_QUARANTINE
            assert trace.uploads[0].root is not None
        finally:
            # CPU fixture has no asynchronous consumers. Explicit completion
            # precedes release of the real quarantine ownership scaffold.
            mx.synchronize();c.store._leased.clear();c.store._pending_host.clear()
            c.store._pending_uploads.clear();c.store._active=False
            native._release_upload(c.store)


def test_GU_refuses_lookalike_row_map_outside_the_factory_packet(monkeypatch):
    with mx.stream(mx.cpu):
        c,trace=_provider(monkeypatch,[[0,1,2,3,4,5]]*2,owned=False)
        c.w.moe(c.x,c.ids,c.scores,'blk.0')
        device=trace.uploads[0];batch=device.packet.plan.batches[0]
        experts=[c.bank[device.packet.plan.jobs[j].expert_id]for j in batch.job_indices]
        rows=mx.array(np.array(device.view('batch0.rows')),mx.uint32)
        scores=c.scores.reshape(-1)[device.view('batch0.locations')]
        calls=len(trace.gu)
        with pytest.raises(ValueError,match='row packet'):
            gu.pp_middle(c.x,experts,rows,scores,batch.row_counts,row_packet=device,batch_index=0)
        assert len(trace.gu)==calls
