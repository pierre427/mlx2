"""Untimed parent-owned replay of the actual layer0 local attention chain.

No model or DSpark capture. Primitive fixtures are already scaled; actual
unscaled node1 Q anchors the ordinary owning attend reference independently.
"""
from pathlib import Path
import hashlib
import json
import os

_QUARANTINE=[]

def _write(path,value):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('x')as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)

def _digest(path,expected):
    raw=Path(path).read_bytes()
    if type(expected)is not str or len(expected)!=64 or hashlib.sha256(raw).hexdigest()!=expected:
        raise ValueError('Local pipeline evidence digest differs')
    return raw

def _compare(a,b):
    import numpy as np
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('Local pipeline stage shape/dtype differs')
    return dict(equal=a.tobytes()==b.tobytes(),shape=list(a.shape),dtype=str(a.dtype),
        finite=bool(np.isfinite(a).all()and np.isfinite(b).all()),
        different_elements=int(np.count_nonzero(np.any(a.view(np.uint8).reshape(-1,a.dtype.itemsize)!=b.view(np.uint8).reshape(-1,b.dtype.itemsize),axis=1))),
        maximum_absolute_error=0. if not a.size else float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))),
        candidate_sha256=hashlib.sha256(a.tobytes()).hexdigest(),reference_sha256=hashlib.sha256(b.tobytes()).hexdigest())

def run(adapter,unit,check_live,*,fixture_path,fixture_sha,stage_path,stage_sha):
    if not callable(check_live):raise ValueError('Local pipeline replay needs caller-owned live guard')
    import mlx.core as mx
    if mx.default_device()!=mx.gpu:raise ValueError('Local pipeline replay requires parent-owned GPU')
    import numpy as np
    from mlx2.adapters import deepseek_v41_ast_batched_attention_exact as attention
    from mlx2.runtime.models.deepseek_v4 import attend
    from mlx2.runtime.gguf import source_ast_batched_splitk as pinned
    from mlx2.runtime.models.deepseek_v41_source_hc import installed_receipt
    from ds41_resident_prefill_model_gate import _checkpoint
    def live():
        if check_live()is False:raise RuntimeError('Local pipeline replay ownership lost')
        return True
    live();unit=Path(unit);path=unit/'ast-exact-local-pipeline.json'
    if path.exists()or path.with_name(path.name+'.tmp').exists():raise FileExistsError('Local pipeline evidence exists')
    spec=json.loads((unit/'spec.json').read_text())
    result=dict(schema='mlx2.ds41-ast-local-pipeline.v1',checkpoint=_checkpoint(adapter),
        source_revision=spec.get('source_revision'),source_snapshot_sha256=spec.get('source_snapshot_sha256'),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),completed=False,
        qualified=False,rate_eligible=False,APCv2_publication=False,model_capture_performed=False,
        input_scope='Actual saved G3 preScaled QK operands; node1 ordinary reference uses actual unscaled Q; no inverse-scaling reconstruction')
    _write(path,result);roots=[];drained=True
    try:
        fixture_path=Path(fixture_path).resolve();stage_path=Path(stage_path).resolve()
        fixture=json.loads(_digest(fixture_path,fixture_sha));stage=json.loads(_digest(stage_path,stage_sha))
        if(fixture.get('checkpoint')!=result['checkpoint']or stage.get('checkpoint')!=result['checkpoint']
           or fixture.get('schema')!='mlx2.ds41-ast-attention-matmul-fixture.v1'
           or stage.get('completed')is not True or stage.get('node')!=1):
            raise ValueError('Local pipeline checkpoint/capture binding differs')
        # Require current arithmetic sources to equal the actual stage producer.
        manifest=json.loads((stage_path.parent/'source/desktop-source.json').read_text())
        if manifest['source_sha256']!=stage['source_snapshot_sha256']:raise ValueError('Stage producer source seal differs')
        modules=(attention,)
        for module in modules:
            relative='src/'+str(Path(module.__file__).resolve()).split('/src/',1)[1]
            expected=manifest['files'][relative]
            _digest(stage_path.parent/'source'/relative,expected);_digest(module.__file__,expected)
        installed=installed_receipt()
        if os.environ.get('MLX_ENABLE_TF32')!='0':raise ValueError('Local pipeline replay requires TF32 disabled')
        fa=Path(fixture['archive']).resolve();sa=Path(stage['raw_arrays_path']).resolve()
        if fa.parent!=fixture_path.parent or sa.parent!=stage_path.parent:raise ValueError('Archive owner differs')
        if fa.stat().st_size>130<<20 or sa.stat().st_size>65<<20:raise ValueError('Replay archive exceeds bound')
        _digest(fa,fixture['archive_sha256']);_digest(sa,stage['raw_arrays_sha256'])
        row=next(r for r in stage['stages']if r['layer']==0 and r['kind']=='attention_core')
        if row['name']!='local11_selected0':raise ValueError('Replay requires actual first local-only core')
        key='stage'+str(row['sequence'])
        with np.load(fa,allow_pickle=False)as arrays:
            left=arrays['case0_left'].copy();right=arrays['case0_right'].copy()
        with np.load(sa,allow_pickle=False)as arrays:
            query=arrays[key+'_input'].copy();gold_candidate=arrays[key+'_candidate'].copy();gold_oracle=arrays[key+'_oracle'].copy()
        if(left.shape!=(3,64,512)or right.shape!=(3,512,11)or query.shape!=(1,64,1,512)
           or gold_candidate.shape!=query.shape or gold_oracle.shape!=query.shape
           or any(a.dtype!=np.float32 or not np.isfinite(a).all()for a in(left,right,query,gold_candidate,gold_oracle))):
            raise ValueError('Replay actual array geometry/dtype/finite contract differs')
        scaled=query[:,:,0]*np.float32(512**-.5)
        if scaled.tobytes()!=left[:1].tobytes():raise ValueError('Actual unscaled Q does not reproduce saved scaled node1')
        q=mx.array(query);a=mx.array(left);b=mx.array(right)
        local=mx.contiguous(mx.swapaxes(b[:1],-1,-2))[:,None]
        sinks=adapter.model.layers[0].attn.sinks
        if sinks.shape!=(64,):raise ValueError('Layer0 sink geometry differs')
        mask=mx.ones((1,1,1,11),mx.bool_)
        # Literal owning attend chain, preserving its actual tensor geometry.
        oq=q.astype(mx.float32)*512**-.5
        ols=oq@mx.swapaxes(local.astype(mx.float32),-1,-2)
        masked=mx.where(mask,ols,-mx.inf)
        olse=mx.logsumexp(masked,axis=-1,keepdims=True)
        onorm=mx.logaddexp(olse,sinks[None,:,None,None].astype(mx.float32))
        op=mx.exp(masked-onorm);opv=op@local.astype(mx.float32)
        ordinary=attend(q,local,mask,sinks=sinks)
        # Exact candidate local class, supplied its saved production preScaled
        # inputs. There is no row loop and no fabricated unscaled other Qs.
        with attention.owned_stock_splitk(live):
            cls=attention._private_matmul(a,b,transpose_b=True,role='local')[:,:,None]
            clse=mx.logsumexp(cls,axis=-1,keepdims=True)
            cnorm=mx.logaddexp(clse,sinks[None,:,None,None].astype(mx.float32))
            cp=mx.exp(cls-cnorm)
            cpv=attention._private_matmul(cp.squeeze(2),mx.swapaxes(b,-1,-2),transpose_b=False,role='local')[:,:,None]
        # Independent candidate: actual ordinary M64 law on every local dot.
        pls=pinned.private_matmul(a,b,transpose_b=True,check_live=live)[:,:,None]
        plse=mx.logsumexp(pls,axis=-1,keepdims=True)
        pnorm=mx.logaddexp(plse,sinks[None,:,None,None].astype(mx.float32))
        pp=mx.exp(pls-pnorm)
        ppv=pinned.private_matmul(pp.squeeze(2),mx.swapaxes(b,-1,-2),transpose_b=False,check_live=live)[:,:,None]
        pinned_stages=(a[:1,:,None],pls[:1],pls[:1],plse[:1],pnorm[:1],pp[:1],ppv[:1])
        candidates=(a[:1,:,None],cls[:1],cls[:1],clse[:1],cnorm[:1],cp[:1],cpv[:1])
        controls=(oq,ols,masked,olse,onorm,op,opv)
        names=('scaled_query','raw_local_scores','masked_local_scores','local_logsumexp','sink_normalizer','probabilities','local_PV')
        roots=[q,a,b,sinks,candidates,pinned_stages,controls,ordinary];drained=False;mx.eval(roots);drained=True;live()
        validity=dict(mirror_vs_owning_attend=_compare(opv,ordinary),
            owning_attend_vs_stored_stage_oracle=_compare(ordinary,gold_oracle),
            candidate_vs_stored_stage_candidate=_compare(cpv[:1],gold_candidate))
        result['replay_validity']=validity
        rows=[dict(stage=name,comparison=_compare(candidate,control))for name,candidate,control in zip(names,candidates,controls)]
        result.update(stages=rows,first_divergence=next((r for r in rows if not r['comparison']['equal']),None),
            fixture=dict(path=str(fixture_path),sha256=fixture_sha,archive_sha256=fixture['archive_sha256']),
            actual_stage=dict(path=str(stage_path),sha256=stage_sha,archive_sha256=stage['raw_arrays_sha256']),
            installed=installed,arithmetic_identity=attention.implementation_identity())
        pinned_rows=[dict(stage=name,comparison=_compare(candidate,control))for name,candidate,control in zip(names,pinned_stages,controls)]
        result['pinned_M64_pipeline']=dict(stages=pinned_rows,all_stages_exact=all(r['comparison']['equal']for r in pinned_rows),
            first_divergence=next((r for r in pinned_rows if not r['comparison']['equal']),None),
            implementation_identity=pinned.implementation_identity(),selected_or_defaulted=False)
        raw={}
        for name,candidate,control in zip(names,candidates,controls):
            raw[name+'_candidate']=np.asarray(candidate);raw[name+'_ordinary']=np.asarray(control)
        for name,value in zip(names,pinned_stages):raw[name+'_pinned_M64']=np.asarray(value)
        archive=unit/'ast-exact-local-pipeline-raw.npz'
        with archive.open('xb')as stream:np.savez(stream,**raw)
        result.update(raw_arrays_path=str(archive),raw_arrays_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        _write(path,result)
        if not all(r['equal']and r['finite']for r in validity.values()):
            raise ValueError('Local pipeline replay failed mandatory owning/stored raw identity validation')
        from ds41_ast_attention_matmul_components import run_replay
        role_gate=run_replay(adapter,unit,live,fixture_path,fixture_sha,
            candidate_families=('pinned_stock_v2','head_axis_padded_v1','head_axis_unpadded_v1'),
            control_law='actual_local_reshape_selected_einsum_v1')
        result['all32_actual_role_replay']=dict(completed=role_gate['completed'],passed=role_gate['passed'],
            passing_families=role_gate['passing_families'],evidence=str(unit/'ast-attention-matmul-components.json'))
        result.update(completed=True,status='completed',all_pipeline_stages_exact=all(r['comparison']['equal']for r in rows),
            interpretation='First divergence localizes this actual pipeline only; no automatic runtime fix/admission or performance claim')
        return result
    except BaseException as error:
        result.update(status='failed',error=dict(type=type(error).__name__,message=str(error)));raise
    finally:
        if not drained:
            try:mx.synchronize()
            except BaseException:_QUARANTINE.append(roots)
        _write(path,result)
