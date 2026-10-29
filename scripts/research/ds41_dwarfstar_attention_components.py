"""Pinned original DS4.1 DwarfStar kernels; GPU ownership is caller-owned."""
import hashlib
import ctypes
import os
from contextlib import contextmanager
from pathlib import Path

LIMIT=6<<30
_PREPARED_QUARANTINE=[]


def peer_repetitions(report):
    options=report.get('peer_options',{})
    if not isinstance(options,dict)or set(options)-{'repeats'}:raise ValueError('Unknown peerattention options')
    value=options.get('repeats',3)
    if type(value)is not int or not 3<=value<=7:raise ValueError('Peerattention repetitions must be integer3..7')
    return value


def preparation_mode(report):
    options=report.get('dwarfstar_options',{})
    if not isinstance(options,dict)or set(options)-{'prepare_mode'}:raise ValueError('Unknown DwarfStar options')
    mode=options.get('prepare_mode','reference')
    if type(mode)is not str or mode not in ('reference','fused_decode'):raise ValueError('Unknown DwarfStar preparation mode')
    return mode


def prepared_bytes(np,reference,actual):
    a=np.ascontiguousarray(reference);b=np.ascontiguousarray(actual)
    same=a.shape==b.shape and a.dtype==b.dtype
    result=dict(exact=bool(same and a.tobytes()==b.tobytes()),reference_shape=list(a.shape),candidate_shape=list(b.shape),
        reference_dtype=str(a.dtype),candidate_dtype=str(b.dtype),reference_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
        candidate_sha256=hashlib.sha256(b.tobytes()).hexdigest(),scope='Preparedbufferbytes inclsignedzero/rounding/padding, no numeric tolerance')
    if same:
        changed=np.flatnonzero(a.view(np.uint8).ravel()!=b.view(np.uint8).ravel());result['changed_bytes']=int(len(changed))
        if len(changed):result['first_changed_byte']=int(changed[0])
    return result


def reference_functions(attend,args,gathered,length):
    functions={'reference':lambda:attend(*args,gathered=gathered)}
    if length>1:
        if (gathered is not None or args[4].shape[-1]!=args[3].shape[1]or args[3].shape[1]>512):
            raise ValueError('Productionbroadcast control requires eligibleallretained PPinputs')
        functions['production_broadcast']=lambda:attend(*args,gathered=None,trusted_broadcast=True)
    return functions


def roundtrip_half(np,value):
    from ds41_attention_components import raw_details
    with np.errstate(over='ignore',invalid='ignore'):actual=value.astype(np.float16).astype(np.float32)
    result=raw_details(np,value,actual);result['all_finite']=bool(np.isfinite(actual).all());return result


@contextmanager
def single_thread_blas(np):
    backend=np.__config__.CONFIG['Build Dependencies']['blas']['name']
    if backend!='accelerate':
        try:from threadpoolctl import threadpool_limits
        except ImportError:
            if any(os.environ.get(k)!='1'for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')):
                raise RuntimeError('NonAccelerate F64oracle needs startup single-thread BLAS environment')
            yield;return
        with threadpool_limits(limits=1,user_api='blas'):yield
        return
    api=ctypes.CDLL('/System/Library/Frameworks/Accelerate.framework/Accelerate')
    api.BLASGetThreading.argtypes=[];api.BLASGetThreading.restype=ctypes.c_uint
    api.BLASSetThreading.argtypes=[ctypes.c_uint];api.BLASSetThreading.restype=ctypes.c_int
    previous=api.BLASGetThreading()
    if api.BLASSetThreading(1)!=0:raise RuntimeError('Cannot set single-thread Accelerate BLAS oracle')
    try:yield
    finally:
        if api.BLASSetThreading(previous)!=0:raise RuntimeError('Cannot restore Accelerate BLAS threading')


def numpy_reference(np,arrays,*,staged=False,source_prescale=False,head_block=4):
    """Common original-input F64 truth by default, with BLAS threads limited1."""
    with single_thread_blas(np):
        return _numpy_reference(np,arrays,staged=staged,source_prescale=source_prescale,head_block=head_block)


def _numpy_reference(np,arrays,*,staged=False,source_prescale=False,head_block=4):
    """Blocked F64 masked/sink equations, not inference or quality acceptance.

    Default uses original inputs and scale entirely in F64. Source-prescale
    and staged-half variants separately isolate input and arithmetic effects.
    """
    q=arrays['q'];local=arrays['local'][0,0];pool=arrays['pooled'][0];ids=arrays['indices']
    if not np.array_equal(ids,np.broadcast_to(np.arange(len(pool)),ids.shape)):
        raise ValueError('DwarfStar minimum shim needs chronological allretained IDs')
    values=np.concatenate((local,pool),0)
    if staged:values=values.astype(np.float16).astype(np.float32)
    values=values.astype(np.float64);visible=np.concatenate((arrays['local_mask'][0,0],arrays['valid'][0]),1)
    result=np.zeros(q.shape,np.float64);scale=np.float32(1)/np.sqrt(np.float32(512))
    for first in range(0,64,head_block):
        last=min(64,first+head_block);query=q[0,first:last]
        if staged:query=query.astype(np.float16).astype(np.float64)
        elif source_prescale:query=(query*scale).astype(np.float64)
        else:query=query.astype(np.float64)*(512**-.5)
        scores=(query.reshape(-1,512)@values.T).reshape(last-first,q.shape[2],len(values))
        if staged:scores*=float(scale)
        scores=np.where(visible[None],scores,-np.inf);sinks=arrays['sinks'][first:last].astype(np.float64)
        maximum=np.maximum(scores.max(-1,initial=-np.inf),sinks[:,None]);weights=np.exp(scores-maximum[...,None])
        denom=weights.sum(-1)+np.exp(sinks[:,None]-maximum)
        result[0,first:last]=(weights.reshape(-1,len(values))@values).reshape(last-first,q.shape[2],512)/denom[...,None]
    return result


def metrics(np,reference,actual):
    from ds41_attention_components import raw_details
    # Word-level debugging explicitly rounds F64 truth to F32; numerical
    # max/RMSE/L2 below retain the unrounded F64 truth throughout.
    value=raw_details(np,reference.astype(np.float32),actual.astype(np.float32));value['all_finite']=bool(np.isfinite(actual).all())
    value['word_diagnostic_scope']='F32words; F64truth rounded onlyforworddebug'if reference.dtype==np.float64 else'OriginalF32referencewords'
    if reference.dtype==np.float64:value['F64_truth_sha256']=hashlib.sha256(np.ascontiguousarray(reference).tobytes()).hexdigest()
    if value['all_finite']and np.isfinite(reference).all():
        delta=actual.astype(np.float64)-reference.astype(np.float64)
        value['rmse']=float(np.sqrt(np.mean(delta*delta)))
        value['truth_max_abs_error']=float(np.max(np.abs(delta),initial=0.))
        value['relative_L2']=float(np.linalg.norm(delta.ravel())/max(np.linalg.norm(reference.astype(np.float64).ravel()),1e-30))
    return value


def fence_prepared(mx,owners):
    try:mx.synchronize()
    except BaseException:_PREPARED_QUARANTINE.extend(owners);raise
    owners.clear()


def run_components(mx,np,capture_dir,out,report,compare,atomic):
    if mx.default_device()!=mx.gpu:raise RuntimeError('DwarfStar attention needs caller-owned GPU')
    if 'dwarfstar_attention'in report:raise ValueError('Refusing to overwrite DwarfStar attention evidence')
    repeats=peer_repetitions(report);selected_mode=preparation_mode(report)
    from ds41_attention_components import capture_module,restored_inputs,raw_details,timed_interleaved
    from ds41_head_group_components import synthetic_stability
    from mlx2.runtime.models.deepseek_v41 import selected_attention
    from mlx2.runtime.models.deepseek_v41_fused import rounded
    from mlx2.runtime.models import deepseek_v41_dwarfstar_attention as dwarf
    parser,path=capture_module(capture_dir);manifest,cases=parser.load_capture(path)
    if len(cases)!=6 or {(c['layer'],c['phase'])for c,a in cases}!={(l,p)for l in (2,20,21)for p in ('pp99','decode1')}:
        raise ValueError('DwarfStar replay needs all six actual cases')
    if any(report.get(k)!=manifest[k]for k in ('checkpoint','source_revision')):raise ValueError('DwarfStar capture identity mismatch')
    section=report['dwarfstar_attention']=dict(schema='mlx2.ds41-dwarfstar-attention.v1',checkpoint=manifest['checkpoint'],
        source_revision=manifest['source_revision'],capture_source_snapshot_sha256=manifest['source_snapshot_sha256'],
        component_source_snapshot_sha256=report.get('source_snapshot_sha256'),capture_path=str(path),capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_repository='https://github.com/antirez/ds4',upstream_revision='ee7e8405232c8971000cc7ba2b945b00bf6d20d3',
        source_kernel='ActualDS4.1:nonvecQ8/C64/8SG PP;vectorC32/32splits/1SG+originalreduce decode',
        numerical_profile='dwarfstar_f16_flash_attention_v1',original_template_arithmetic_preserved=True,
        stock_Source_reduction_law=False,quality_verdict='not evaluated',qualified=False,performance_qualified=False,candidate_used_by_model=False,
        original_weight_quantization_unchanged=True,cache_layout_unchanged=True,maximum_allocator_bytes=LIMIT,
        compile_math_mode='fast,matchingupstreamdefaultintent;notbinary/codegenidentityclaim',
        scope='Sixactualallretainedcaptures;sharedF16bank/maskshim; no generalindexed/longcontext claim',
        timing_scope='Exactstockcontrols thenfinite/deterministicactualoutputs; F64errors/stabilitythresholdsreportedseparately, notqualityacceptance',
        F64_BLAS_threads=1,repetitions=repeats,requested_prepare_mode=selected_mode,
        cases=[],synthetic_stability=[],complete=False,harness_completed=False,required_reference_controls_exact=False,
        required_prepared_buffers_exact=False,verdicts={})
    out=Path(out)
    def write():atomic(out/'results.json',report)
    def observe():
        peak=int(mx.get_peak_memory());section['peak_mlx_bytes']=max(section.get('peak_mlx_bytes',0),peak)
        report['component_peak_mlx_bytes']=max(report.get('component_peak_mlx_bytes',0),peak)
        if peak>LIMIT:write();raise MemoryError('DwarfStar attention exceeds6GiB')
    write();observe()
    for case,arrays in cases:
        if case['length']!=(99 if case['phase']=='pp99'else 1)or case['frontier']!=(0 if case['phase']=='pp99'else 100):
            raise ValueError('DwarfStar actualcapture phase/frontier mismatch')
        args,gathered,layout=restored_inputs(mx,np,case,arrays)
        references=reference_functions(selected_attention,args,gathered,case['length'])
        raw=references['reference']();owner=rounded(raw);mx.eval(raw,owner)
        controls=dict(raw=raw_details(np,arrays['reference_raw'],np.array(raw)),BF16=raw_details(np,arrays['reference_bf16'],np.array(owner)))
        if 'production_broadcast'in references:
            broadcast=references['production_broadcast']();broadcast_owner=rounded(broadcast);mx.eval(broadcast,broadcast_owner)
            controls['production_broadcast_raw']=raw_details(np,arrays['reference_raw'],np.array(broadcast))
            controls['production_broadcast_BF16']=raw_details(np,arrays['reference_bf16'],np.array(broadcast_owner))
            del broadcast,broadcast_owner
        section['cases'].append(dict(capture=case,input_layout=layout,reference_controls=controls,
            production_broadcast_control_selected='production_broadcast'in references))
        if not all(v['exact']for v in controls.values()):write();raise ValueError('DwarfStar stockcapture controls must beexact')
        mx.synchronize();args=None;del raw,owner;observe();write();mx.clear_cache()
    section['required_reference_controls_exact']=True;write()
    # Establish all six exact preparation controls before any attention timing.
    for index,(case,arrays)in enumerate(cases):
        args,gathered,layout=restored_inputs(mx,np,case,arrays);owned=[]
        try:
            old=dwarf.prepare(*args,trusted_all_retained=True,prepare_mode='reference');owned.append(old)
            new=dwarf.prepare(*args,trusted_all_retained=True,prepare_mode=selected_mode);owned.append(new)
            buffers=('kv','mask','blocks','params');mx.eval(*(getattr(value,key)for value in (old,new)for key in buffers))
            checks={key:prepared_bytes(np,np.array(getattr(old,key)),np.array(getattr(new,key)))for key in buffers}
            exact=all(v['exact']for v in checks.values());identity=new.q is args[0]and new.sinks is args[6];same_plan=old.plan==new.plan
            section['cases'][index]['preparation_controls']=dict(checks=checks,all_bytes_exact=exact,
                requested_prepare_mode=selected_mode,actual_prepare_mode=new.prepare_mode,
                Q_and_sink_identity_preserved=identity,attention_plan_unchanged=same_plan)
            if not(exact and identity and same_plan):write();raise ValueError('DwarfStar fusedpreparation bytes/ownership/plan differ fromreference')
            observe();write()
        finally:fence_prepared(mx,owned)
        args=None;del old,new;mx.clear_cache()
    section['required_prepared_buffers_exact']=True;write();stability=True
    for name,values in synthetic_stability(np):
        host=dict(zip(('q','local','local_mask','pooled','indices','valid','sinks'),values));expected=numpy_reference(np,host,staged=True)
        inputs=[mx.array(v)for v in values];mx.eval(*inputs);owned=[]
        try:
            prepared=dwarf.prepare(*inputs,trusted_all_retained=True,prepare_mode=selected_mode);owned.append(prepared)
            value=dwarf.attention(prepared);mx.eval(value);actual=np.array(value)
            check=metrics(np,expected,actual);passed=check['all_finite']and bool(np.allclose(expected,actual,rtol=1e-4,atol=1e-5))
            section['synthetic_stability'].append(dict(name=name,scope='Syntheticoperatornotmodelquality',checks=check,
                operator_stability_passed=passed,tolerance='rtol1e-4/atol1e-5 stagedF64equations'))
            stability&=passed;observe();write()
        finally:fence_prepared(mx,owned)
        prepared=None;del value,inputs;mx.clear_cache()
    for index,(case,arrays)in enumerate(cases):
        args,gathered,layout=restored_inputs(mx,np,case,arrays);row=section['cases'][index];owned=[]
        row['stage_F16_roundtrips']={key:roundtrip_half(np,arrays[key])for key in ('q','local','pooled')}
        common_oracle=numpy_reference(np,arrays);source_oracle=numpy_reference(np,arrays,source_prescale=True);staged_oracle=numpy_reference(np,arrays,staged=True)
        row['F64_oracles']=dict(common_original_inputs_vs_stock=metrics(np,common_oracle,arrays['reference_raw']),
            source_prescale_vs_stock=metrics(np,source_oracle,arrays['reference_raw']),stage_precision_effect=metrics(np,common_oracle,staged_oracle),
            scope='IndependentblockedF64equations;notqualityorbitwiseoracle')
        try:
            prepared=dwarf.prepare(*args,trusted_all_retained=True,prepare_mode=selected_mode);owned.append(prepared)
            old_prepared=dwarf.prepare(*args,trusted_all_retained=True,prepare_mode='reference');owned.append(old_prepared)
            mx.eval(prepared.kv,prepared.mask,prepared.blocks,prepared.params)
            value=dwarf.attention(prepared);old_value=dwarf.attention(old_prepared);second=dwarf.attention(prepared);owner=rounded(value)
            mx.eval(value,old_value,second,owner);actual=np.array(value)
            checks=dict(raw=metrics(np,arrays['reference_raw'],actual),BF16=metrics(np,arrays['reference_bf16'],np.array(owner)),
                repeat_raw=raw_details(np,actual,np.array(second)),common_original_input_F64=metrics(np,common_oracle,actual),
                staged_F64=metrics(np,staged_oracle,actual),source_prescale_F64=metrics(np,source_oracle,actual),
                preparation_reference_attention_raw=raw_details(np,np.array(old_value),actual))
            if not checks['preparation_reference_attention_raw']['exact']:write();raise ValueError('Preparationmode changedoriginalDwarfStarattention output')
            admitted=checks['raw']['all_finite']and checks['BF16']['all_finite']and checks['repeat_raw']['exact']
            row.update(plan=prepared.plan,checks=checks,actual_prepare_mode=prepared.prepare_mode,operator_stability_passed=stability,
                timing_admitted=admitted,quality_verdict='not evaluated',observed_scope='boundedactualoperandcomponentonly')
            if admitted:
                references=reference_functions(selected_attention,args,gathered,case['length'])
                def total():
                    fresh=dwarf.prepare(*args,trusted_all_retained=True,prepare_mode=selected_mode);owned.append(fresh);return dwarf.attention(fresh)
                def reference_total():
                    fresh=dwarf.prepare(*args,trusted_all_retained=True,prepare_mode='reference');owned.append(fresh);return dwarf.attention(fresh)
                def prep_only():
                    fresh=dwarf.prepare(*args,trusted_all_retained=True,prepare_mode=selected_mode);owned.append(fresh);return(fresh.kv,fresh.mask,fresh.blocks,fresh.params)
                def reference_prep():
                    fresh=dwarf.prepare(*args,trusted_all_retained=True,prepare_mode='reference');owned.append(fresh);return(fresh.kv,fresh.mask,fresh.blocks,fresh.params)
                row['timings']=timed_interleaved(mx,np,{**references,'dwarfstar_core':lambda:dwarf.attention(prepared),
                    'dwarfstar_reference_stage_total':reference_total,'dwarfstar_total_including_stage':total},repeats=repeats)
                row['preparation_timings']=timed_interleaved(mx,np,{'reference_F16_bank_mask_args':reference_prep,'selected_F16_bank_mask_args':prep_only},repeats=repeats)
            else:row['timings']={}
            observe();write()
        finally:fence_prepared(mx,owned)
        prepared=None;args=None;del old_prepared,old_value,value,second,owner;mx.clear_cache()
    section['verdicts']=dict(all6_operator_valid=all(c['timing_admitted']for c in section['cases']),all6_prepared_buffers_exact=section['required_prepared_buffers_exact'],synthetic_operator_stability=stability,
        all6_stage_F16_exact=all(all(v['exact']for v in c['stage_F16_roundtrips'].values())for c in section['cases']))
    section['complete']=section['harness_completed']=True;write();return section
