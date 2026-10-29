"""Bounded Source PP99 WoA components; caller owns the 6GiB GPU unit.

Original resident BF16 coefficients stay authoritative. The ordinary timed
closure performs its F32 coefficient cast on EVERY call. Native BF16 output
has no exposed pre-round F32 accumulator. No model/route work on import.
"""
import hashlib
import json
from pathlib import Path
import time

LIMIT=6<<30
NATIVE_PROFILE='native_bf16_grouped_pp_v1'


def sequence_details(np,expected,actual):
    """Diagnostic metadata for B,T,G,D or B,T,D (query axis1).

    Shared attention raw_details uses B,H,T,D. Preserve its bit counts,
    hashes and first coordinate; only replace the sequence-row annotation.
    """
    from ds41_attention_components import raw_details
    if expected.ndim not in(3,4):raise ValueError('WoA sequence diagnostics require B,T[,G],D')
    result=raw_details(np,expected,actual)
    changed=(np.ascontiguousarray(expected).view(np.uint32)!=np.ascontiguousarray(actual).view(np.uint32))
    counts=np.count_nonzero(changed,axis=tuple(i for i in range(changed.ndim)if i!=1))
    result.update(query_row_axis=1,query_row_changed_counts=list(map(int,counts[:128])),query_rows_truncated=len(counts)>128)
    return result


def component_options(report):
    value=report.get('woa_options',{})
    if not isinstance(value,dict)or set(value)-{'numerical_profile','repeats'}:raise ValueError('Unknown WoA options')
    profile=value.get('numerical_profile','reference_exact_only')
    if type(profile)is not str or profile not in('reference_exact_only',NATIVE_PROFILE):raise ValueError('Unknown WoA numerical profile')
    repeats=value.get('repeats',7 if profile==NATIVE_PROFILE else 3)
    if type(repeats)is not int or not 3<=repeats<=7:raise ValueError('WoA repeats must be3..7')
    return profile,repeats


def round_bf16_f32(np,value):
    if value.dtype!=np.float32 or not np.isfinite(value).all():raise ValueError('WoA BF16 audit requires finite F32')
    bits=np.ascontiguousarray(value).view(np.uint32)
    rounded=((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
    if not np.isfinite(rounded).all():raise ValueError('WoA BF16 overflow')
    return rounded


def fp64_reference(np,x,q,scales,*,groups=8,block_rows=128,resident_rows=None):
    """Blocked common oracle for ORIGINAL resident BF16 coefficients.

    Original code/scale decoding is followed by the official resident BF16
    conversion, not a substitution of unrounded effective FP8 weights.
    """
    from ds41_attention_projection_components import decode_effective_f32
    from ds41_dwarfstar_attention_components import single_thread_blas
    if(type(groups)is not int or groups<1 or type(block_rows)is not int or not 1<=block_rows<=128
            or x.dtype!=np.float32 or x.ndim!=4 or x.shape[0]!=1 or not x.shape[1]or x.shape[2]!=groups
            or not np.isfinite(x).all()or q.dtype!=np.uint32 or q.ndim!=2 or not q.shape[0]or q.shape[0]%groups
            or x.shape[-1]!=q.shape[1]*4):raise ValueError('WoA FP64 input/group/coefficient geometry')
    codes=np.ascontiguousarray(q).view(np.uint8)
    if scales.dtype!=np.uint8 or scales.shape!=(q.shape[0],codes.shape[1]//32):raise ValueError('WoA FP64 scale geometry')
    rows,width=x.shape[1],x.shape[-1];outputs=q.shape[0]//groups
    truth=np.empty((1,rows,groups,outputs),np.float64);changed=0;total=0;verified_blocks=0
    with single_thread_blas(np):
        for group in range(groups):
            x64=x[0,:,group].astype(np.float64)
            for first in range(0,outputs,block_rows):
                last=min(first+block_rows,outputs);lo=group*outputs+first;hi=group*outputs+last
                effective=decode_effective_f32(np,codes[lo:hi],scales[lo:hi]);weight=round_bf16_f32(np,effective)
                changed+=int(np.count_nonzero(effective.view(np.uint32)!=weight.view(np.uint32)));total+=int(weight.size)
                if resident_rows is not None:
                    actual=resident_rows(lo,hi)
                    if (actual.dtype!=np.float32 or actual.shape!=weight.shape
                            or actual.tobytes()!=weight.tobytes()):raise ValueError('WoA original resident BF16 coefficient replay differs')
                    weight=actual;verified_blocks+=1
                truth[0,:,group,first:last]=x64@weight.astype(np.float64).T
    return truth,dict(oracle_coefficients='Original codes/scales decoded, then owning resident BF16 conversion',
        effective_FP8_F32_to_resident_BF16_changed_words=changed,total_coefficient_words=total,
        effective_FP8_F32_BF16_roundtrip_exact=changed==0,maximum_decoded_rows=block_rows,
        BLAS_threads=1,actual_resident_coefficient_blocks_verified=verified_blocks,
        truth_sha256=hashlib.sha256(truth.tobytes()).hexdigest())


def ordinary_grouped(mx,x,weight):
    """Production-real PP99 Source expression: cast inside every invocation."""
    return mx.einsum('btgi,goi->btgo',x.astype(mx.float32),weight.astype(mx.float32).reshape(8,1024,4096))


def native_grouped(mx,x,weight):
    """Explicit changed numerical profile, input/output casts inside call."""
    return mx.einsum('btgi,goi->btgo',x.astype(mx.bfloat16),weight.reshape(8,1024,4096)).astype(mx.float32)


def native_timing_gate(np,first,second,*,profile,owner_exact):
    if profile not in('reference_exact_only',NATIVE_PROFILE)or type(owner_exact)is not bool:raise ValueError('WoA timing profile/exactness contract')
    if first.dtype!=np.float32 or second.dtype!=np.float32 or first.shape!=second.shape:raise ValueError('WoA timing output shape/dtype')
    finite=bool(np.isfinite(first).all()and np.isfinite(second).all());stable=first.tobytes()==second.tobytes()
    return dict(finite=finite,self_repeat_exact=stable,
        timing_admitted=finite and stable and(owner_exact or profile==NATIVE_PROFILE),
        changed_numerical_profile=not owner_exact)


def allocator_probe(mx,function):
    """Allocator-owned bytes only, not DRAM traffic; no retained result."""
    mx.synchronize();mx.clear_cache()
    if not callable(getattr(mx,'reset_peak_memory',None)):raise ValueError('WoA allocator probe requires reset_peak_memory')
    before=int(mx.get_active_memory());mx.reset_peak_memory()
    value=function();mx.eval(value);peak=int(mx.get_peak_memory());active=int(mx.get_active_memory())
    del value;mx.synchronize();after=int(mx.get_active_memory());mx.clear_cache()
    return dict(resident_before_bytes=before,peak_active_bytes=peak,peak_increment_bytes=peak-before,
        active_after_eval_bytes=active,active_after_output_release_bytes=after,
        scope='MLX allocator bytes with prepared inputs resident; not bus/DRAM traffic or GPU-active time')


def run_components(mx,np,captureDir,out,report,compare,atomic):
    """Three actual-derived PP99 cases, unchanged products/consumer controls."""
    if mx.default_device()!=mx.gpu:raise RuntimeError('WoA PP components require caller-owned GPU')
    if 'woa_pp_components'in report:raise ValueError('Refusing WoA evidence overwrite')
    profile,repeats=component_options(report)
    cross_dir=report.get('output_chain_projection_capture_path')
    if not cross_dir:raise ValueError('WoA actual prepared-WoB projection capture required')
    from ds41_attention_components import capture_module,raw_details
    from ds41_attention_projection_components import capture_module as projection_module,coefficient_payload,verify_coefficients,upload_coefficients,fp64_error,profile_timings
    from ds41_output_chain_components import case_inventory,grouped_layout,_dequant_wo_a
    from mlx2.adapters.deepseek_v41_source_artifact import inspect_artifact,read_tensors
    from mlx2.runtime.models.deepseek_v4 import Rotary
    from mlx2.runtime.models.deepseek_v41_fused import rounded,rotate,rotate_qat
    from mlx2.runtime.models.deepseek_v41_quant import fused_quantize
    parser,path=capture_module(captureDir);manifest,cases=parser.load_capture(path);case_inventory(cases)
    capture_spec=json.loads((path.parent/'spec.json').read_text())
    cross_parser,cross_path,cross_spec=projection_module(cross_dir);cross_manifest,cross_cases=cross_parser.load_capture(cross_path)
    for source in(manifest,cross_manifest):
        if any(report.get(k)!=source.get(k)for k in('checkpoint','source_revision')):raise ValueError('WoA capture checkpoint/source mismatch')
    cross={(c['layer'],c['phase']):(c,a)for c,a in cross_cases if c['projection']=='wo_b'}
    if set(cross)!={(l,p)for l in(2,20,21)for p in('pp99','decode1')}:raise ValueError('WoA crosscapture inventory')
    model_path=capture_spec.get('model')or report.get('model')
    if not model_path:raise ValueError('WoA Source artifact missing')
    artifact=inspect_artifact(model_path)
    for source in(capture_spec,cross_spec,report):
        if source.get('original_fingerprint',artifact['identity']['fingerprint'])!=artifact['identity']['fingerprint']:raise ValueError('WoA original Source artifact mismatch')
        if source.get('model')and Path(source['model']).resolve()!=Path(artifact['root']).resolve():raise ValueError('WoA Source artifact path mismatch')
    out=Path(out);c=artifact['args']
    section=report['woa_pp_components']=dict(schema='mlx2.ds41-woa-pp-components.v1',checkpoint=manifest['checkpoint'],
        source_revision=manifest['source_revision'],component_source_snapshot_sha256=report['source_snapshot_sha256'],
        capture_source_snapshot_sha256=manifest['source_snapshot_sha256'],capture_path=str(path),capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        cross_capture_path=str(cross_path),cross_capture_source_snapshot_sha256=cross_manifest['source_snapshot_sha256'],cross_capture_sha256=hashlib.sha256(cross_path.read_bytes()).hexdigest(),
        numerical_profile=profile,repeats=repeats,maximum_allocator_bytes=LIMIT,peak_mlx_bytes=0,
        candidate_used_by_model=False,qualified=False,performance_qualified=False,model_quality_assessed=False,
        baseline_coefficient_cast='Inside every ordinary_grouped invocation; no retained F32 weight cache',
        native_raw_accumulator_exposed=False,scope='Three PP99 derived actual-attention trajectories with exact separate-capture prepared-WoB controls; raw WoA not independently captured',
        cases=[],complete=False,harness_completed=False,required_reference_controls_exact=False,verdicts={})
    def write():atomic(out/'results.json',report)
    def observe(peak=None):
        peak=int(mx.get_peak_memory())if peak is None else peak
        section['peak_mlx_bytes']=max(section['peak_mlx_bytes'],peak);report['component_peak_mlx_bytes']=max(report.get('component_peak_mlx_bytes',0),peak)
        if peak>LIMIT:write();raise MemoryError('WoA PP components exceeded6GiB')
    def case_run(case,arrays):
        layer=case['layer'];length=case['length'];start=case['frontier']
        row=dict(layer=layer,length=length,phase=case['phase'],frontier=start,reference_controls={},timings={},allocator={},native={})
        section['cases'].append(row);write();loads=[];hosts=[];payload=[]
        for name in('attn_output_a','attn_output_b'):
            alias=f'blk.{layer}.{name}.weight';q,s,load=coefficient_payload(np,artifact,alias,read_tensors)
            if name=='attn_output_b':verify_coefficients(load,cross[(layer,'pp99')][0])
            hosts.append((q,s));payload.append(upload_coefficients(mx,q,s));loads.append(load)
        weight=_dequant_wo_a(mx,*payload[0]);wo_b_q,wo_b_s=payload[1];del payload
        if weight.dtype!=mx.bfloat16 or weight.shape!=(8192,4096):raise ValueError('WoA original resident coefficient geometry')
        row['loaded_coefficients']=loads;row['resident_weight_shape']=list(weight.shape);row['resident_weight_dtype']='bfloat16'
        rope=Rotary(c.qk_rope_head_dim,c.compress_rope_theta if c.compress_ratios[layer]else c.rope_theta,c.rope_scaling if c.compress_ratios[layer]else None)
        raw=mx.array(arrays['reference_raw'],mx.float32)
        def prepare():return grouped_layout(rotate_qat(rounded(raw),rope,'fp8',start,inverse=True))
        def prepare_b(value):return fused_quantize(rounded(value).reshape(1,99,8192),'fp8')
        def project_b(value):return mx.quantized_matmul(value,wo_b_q,scales=wo_b_s,mode='mxfp8')
        x=prepare();plain=grouped_layout(rounded(fused_quantize(rotate(rounded(raw),rope,start,inverse=True),'fp8')))
        product=ordinary_grouped(mx,x,weight);owner=rounded(product);prepared_b=prepare_b(product);out_b=project_b(prepared_b);final=rounded(out_b)
        native=native_grouped(mx,x,weight);native_b=prepare_b(native);native_out_b=project_b(native_b);native_final=rounded(native_out_b)
        mx.eval(x,plain,product,owner,prepared_b,out_b,final,native,native_b,native_out_b,native_final);observe()
        other,actual=cross[(layer,'pp99')]
        checks=dict(attention_owner=raw_details(np,arrays['reference_bf16'],np.array(rounded(raw))),
            rotation_QAT_equations=sequence_details(np,np.array(plain),np.array(x)),
            actual_prepared_WoB=sequence_details(np,actual['prepared_input'],np.array(prepared_b)),
            actual_WoB_raw=sequence_details(np,actual['reference_raw'],np.array(out_b)),actual_WoB_boundary=sequence_details(np,actual['reference_bf16'],np.array(final)))
        row['reference_controls']=checks
        if not all(v['exact']for v in checks.values()):write();raise ValueError('WoA ordinary owning/crosscapture controls failed')
        x_host=np.array(x);roundtrip=sequence_details(np,x_host,round_bf16_f32(np,x_host));row['actual_derived_WoA_input_BF16_roundtrip']=roundtrip
        if not roundtrip['exact']:write();raise ValueError('WoA current input does not losslessly roundtrip BF16')
        native_host=np.array(native);ref_owner_host=np.array(owner);source_raw_host=np.array(product)
        owner_check=sequence_details(np,ref_owner_host,native_host)
        consumer_checks=dict(prepared_WoB=sequence_details(np,np.array(prepared_b),np.array(native_b)),raw_WoB=sequence_details(np,np.array(out_b),np.array(native_out_b)),final_boundary=sequence_details(np,np.array(final),np.array(native_final)))
        row['native'].update(owner_check=owner_check,consumer_checks=consumer_checks,internal_pre_BF16_F32_exposed=False)
        def resident_rows(first,last):
            # At most128rows: audit actual loaded coefficients without a full
            # global F32 expansion or a persistent timing-weight cache.
            slab=weight[first:last].astype(mx.float32);mx.eval(slab)
            result=np.array(slab);del slab;return result
        started=time.perf_counter();truth,weight_audit=fp64_reference(np,x_host,*hosts[0],resident_rows=resident_rows)
        row['common_FP64_oracle']=dict(coefficient_audit=weight_audit,seconds=time.perf_counter()-started,
            source_raw=fp64_error(np,truth,source_raw_host),source_boundary=fp64_error(np,truth,ref_owner_host),native_boundary=fp64_error(np,truth,native_host),
            native_raw_accumulator='Unexposed: native output is already BF16; no fabricated rawF32 parity/error')
        del truth,plain,product,owner,prepared_b,out_b,final,native,native_b,native_out_b,native_final
        # Timing does not retain either the ordinary F32 cast or its result.
        mx.synchronize();mx.clear_cache()
        def reference_op():return rounded(ordinary_grouped(mx,x,weight))
        def native_op():return native_grouped(mx,x,weight)
        def reference_chain():return rounded(project_b(prepare_b(ordinary_grouped(mx,prepare(),weight))))
        def native_chain():return rounded(project_b(prepare_b(native_grouped(mx,prepare(),weight))))
        first=native_op();mx.eval(first);one=np.array(first);del first
        second=native_op();mx.eval(second);two=np.array(second);del second
        operator_gate=native_timing_gate(np,one,two,profile=profile,owner_exact=owner_check['exact'])
        first=native_chain();mx.eval(first);one=np.array(first);del first
        second=native_chain();mx.eval(second);two=np.array(second);del second
        chain_exact=owner_check['exact']and all(v['exact']for v in consumer_checks.values())
        chain_gate=native_timing_gate(np,one,two,profile=profile,owner_exact=chain_exact)
        row['native'].update(operator_gate=operator_gate,full_chain_gate=chain_gate)
        functions={'reference':reference_op};chains={'reference':reference_chain}
        if operator_gate['timing_admitted']:functions['native_bf16']=native_op
        if chain_gate['timing_admitted']:chains['native_bf16']=native_chain
        row['timings']['grouped_with_owner']=profile_timings(mx,np,functions,repeats=repeats)
        row['timings']['full_producer_consumer_chain']=profile_timings(mx,np,chains,repeats=repeats)
        row['timing_scope']='Prepared-input grouped+owner includes eachcall coefficientF32 cast; fullchain additionally includes inverseRoPE/QAT/layout and unchanged WoB inputQAT/product/owner. Native input/output casts inside both timed closures.'
        observe();write()
        for name,fn in functions.items():
            result=allocator_probe(mx,fn);row['allocator'][name]=result;observe(result['peak_active_bytes']);write()
        row['payload_geometry']=dict(input_F32_bytes=99*8*4096*4,native_input_BF16_bytes=99*8*4096*2,resident_weight_BF16_bytes=8192*4096*2,
            reference_eachcall_F32_weight_bytes=8192*4096*4,output_F32_bytes=99*8*1024*4,native_output_BF16_bytes=99*8*1024*2,
            scope='Logical tensor payload only; no DRAM traffic claim')
        row['complete']=True;write()
    write();observe()
    for case,arrays in cases:
        if case['phase']!='pp99':continue
        case_run(case,arrays);mx.synchronize();mx.clear_cache();observe()
    section['required_reference_controls_exact']=all(all(v['exact']for v in r['reference_controls'].values())for r in section['cases'])
    section['verdicts']=dict(native_all3_owner_exact=all(r['native']['owner_check']['exact']for r in section['cases']),
        native_all3_finite_stable=all(r['native']['operator_gate']['finite']and r['native']['operator_gate']['self_repeat_exact']and r['native']['full_chain_gate']['finite']and r['native']['full_chain_gate']['self_repeat_exact']for r in section['cases']),
        actual_derived_input_all3_BF16_roundtrip_exact=all(r['actual_derived_WoA_input_BF16_roundtrip']['exact']for r in section['cases']))
    section.update(complete=True,harness_completed=True);write();return section
