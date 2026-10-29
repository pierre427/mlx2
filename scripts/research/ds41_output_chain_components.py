"""Source-shaped output-chain replay; GPU ownership belongs to the caller.

Actual raw attention operands seed a derived producer/consumer graph. Separate
projection captures are trajectory evidence only if their prepared inputs match.
No Source provider, model load, serving selection or MLX import on import.
"""
import hashlib
import json
from pathlib import Path

LIMIT=6<<30
_DEQUANT_QUARANTINE=[]
VARIANTS={'packed_inverse_QAT':(True,False),'WoB_input_QAT':(False,True),'combined_chain':(True,True)}


def component_options(report):
    value=report.get('output_chain_options',{})
    if not isinstance(value,dict)or set(value)-{'repeats','variants'}:raise ValueError('Unknown output-chain options')
    repeats=value.get('repeats',3);names=value.get('variants',list(VARIANTS))
    if type(repeats)is not int or not 1<=repeats<=15:raise ValueError('Output-chain repetitions must be integer1..15')
    if (not isinstance(names,list)or not names or any(type(n)is not str for n in names)
            or len(set(names))!=len(names)or any(n not in VARIANTS for n in names)):
        raise ValueError('Invalid output-chain variant subset')
    return repeats,[(name,*VARIANTS[name])for name in names]


def grouped_layout(value):
    return value.reshape(1,8,8,value.shape[2],512).transpose(0,3,1,2,4).reshape(1,value.shape[2],8,4096)


def owning_grouped(mx,x,bf16_weight,f32_grouped_weight,policy,maybe_project):
    """Mirror Source.grouped M1/M99 selection, including structural refusal."""
    if policy in ('woa_f32','direct_f32')and x.shape[1]==1:
        direct=maybe_project(x.astype(mx.float32),bf16_weight,'wo_a')
        if direct is not None:return direct
    return mx.einsum('btgi,goi->btgo',x.astype(mx.float32),f32_grouped_weight)


def _dequant_wo_a(mx,q,s):
    values=[]
    try:
        values.append(mx.dequantize(q,scales=s,mode='mxfp8',dtype=mx.bfloat16));mx.eval(*values)
        return values[0]
    except BaseException:
        try:mx.synchronize()
        except BaseException:_DEQUANT_QUARANTINE.append((q,s,*values))
        raise


def case_inventory(cases):
    expected={(layer,phase)for layer in (2,20,21)for phase in ('pp99','decode1')}
    if len(cases)!=6 or {(c['layer'],c['phase'])for c,a in cases}!=expected:
        raise ValueError('Output chain requires all six actual attention operands')
    for c,a in cases:
        if c['length']!=(99 if c['phase']=='pp99'else 1)or c['frontier']!=(0 if c['phase']=='pp99'else 100):
            raise ValueError('Output chain capture phase/frontier mismatch')


def run_components(mx,np,captureDir,out,report,compare,atomic):
    if mx.default_device()!=mx.gpu:raise RuntimeError('Output-chain components require caller-owned GPU')
    if 'output_chain_components'in report:raise ValueError('Refusing to overwrite output-chain evidence')
    repeats,variants=component_options(report)
    from ds41_attention_components import capture_module,raw_details
    from ds41_attention_projection_components import capture_module as projection_module,coefficient_payload,verify_coefficients,upload_coefficients
    from mlx2.adapters.deepseek_v41_source_artifact import inspect_artifact,read_tensors
    from mlx2.runtime.models.deepseek_v4 import Rotary
    from mlx2.runtime.models.deepseek_v41_fused import rounded,rotate,rotate_qat
    from mlx2.runtime.models.deepseek_v41_quant import fused_quantize
    from mlx2.runtime.models import deepseek_v41_output_chain as candidate
    from mlx2.runtime.gguf.source_dense import maybe_project,supports
    parser,path=capture_module(captureDir);manifest,cases=parser.load_capture(path);case_inventory(cases)
    capture_spec=json.loads((path.parent/'spec.json').read_text())
    for key in ('checkpoint','source_revision'):
        if manifest[key]!=report.get(key):raise ValueError('Output-chain capture identity mismatch')
    artifact=inspect_artifact(capture_spec.get('model')or report['model']);c=artifact['args']
    for identity in (capture_spec,report):
        if identity.get('original_fingerprint',artifact['identity']['fingerprint'])!=artifact['identity']['fingerprint']:
            raise ValueError('Output-chain original Source identity mismatch')
    if report.get('model')and Path(report['model']).resolve()!=Path(artifact['root']).resolve():
        raise ValueError('Output-chain Source artifact path mismatch')
    cross={};cross_spec=None
    if report.get('output_chain_projection_capture_path'):
        cross_parser,cross_path,cross_spec=projection_module(report['output_chain_projection_capture_path'])
        cross_manifest,cross_cases=cross_parser.load_capture(cross_path)
        if any(cross_manifest[k]!=manifest[k]for k in ('checkpoint','source_revision')):
            raise ValueError('Output-chain projection capture identity mismatch')
        cross={(c['layer'],c['phase']):(c,a)for c,a in cross_cases if c['projection']=='wo_b'}
        if len(cross)!=6:raise ValueError('Output-chain six WoB projection captures required')
    section=report['output_chain_components']=dict(schema='mlx2.ds41-output-chain-components.v1',
        checkpoint=manifest['checkpoint'],source_revision=manifest['source_revision'],
        capture_source_snapshot_sha256=manifest['source_snapshot_sha256'],component_source_snapshot_sha256=report.get('source_snapshot_sha256'),
        capture_path=str(path),capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        original_fingerprint=artifact['identity']['fingerprint'],scope='Actualattention inputs; reconstructed chain, cross-run trajectory onlyifpreparedWoBexact',
        qualified=False,performance_qualified=False,candidate_used_by_model=False,all_dense_provider_constructed=False,
        maximum_allocator_bytes=LIMIT,arithmetic='OwnerBF16/inverseRoPE/BF16/FP8QAT/BF16→stockBF16WoAF32product→BF16/FP8QAT→stockWoB',
        repetitions=repeats,variant_subset=[name for name,a,b in variants],
        grouped_weight_policy='Official BF16WoA; captured Source.grouped M1directF32 policy or M99stockF32einsum; no FP8WoA tensor arithmetic',
        capture_extra_source_policy=capture_spec.get('extra_source_policy',{}),
        cross_capture_present=bool(cross),cross_capture_source_snapshot_sha256=None if cross_spec is None else cross_spec['source_snapshot_sha256'],
        loaded_coefficients=[],cases=[],complete=False,harness_completed=False,required_reference_controls_exact=False,verdicts={})
    out=Path(out)
    def write():atomic(out/'results.json',report)
    def observe():
        peak=int(mx.get_peak_memory());section['peak_mlx_bytes']=max(section.get('peak_mlx_bytes',0),peak)
        report['component_peak_mlx_bytes']=max(report.get('component_peak_mlx_bytes',0),peak)
        if peak>LIMIT:write();raise MemoryError('Output-chain component exceeded6GiB')
    write();observe();bank={}
    for case,arrays in cases:
        layer,length,start=case['layer'],case['length'],case['frontier'];p=candidate.layout(length)
        if not np.isfinite(arrays['reference_raw']).all():raise ValueError('Output-chain finite input contract')
        if arrays['reference_raw'].shape!=p['input_shape']:raise ValueError('Output-chain raw attention shape')
        if layer not in bank:
            loads=[];payload=[]
            for name in ('attn_output_a','attn_output_b'):
                alias=f'blk.{layer}.{name}.weight';q,s,load=coefficient_payload(np,artifact,alias,read_tensors)
                if name=='attn_output_b'and cross:verify_coefficients(load,cross[(layer,case['phase'])][0])
                q,s=upload_coefficients(mx,q,s);payload.append((q,s));loads.append(load)
            wo_a=_dequant_wo_a(mx,*payload[0]);q,s=payload[1]
            bank[layer]=(wo_a,q,s,loads[1]);section['loaded_coefficients'].extend(loads)
            del payload
        wo_a,q,s,wo_b_load=bank[layer]
        if cross:verify_coefficients(wo_b_load,cross[(layer,case['phase'])][0])
        weight=wo_a.astype(mx.float32).reshape(8,1024,4096)
        rope=Rotary(c.qk_rope_head_dim,c.compress_rope_theta if c.compress_ratios[layer]else c.rope_theta,
            c.rope_scaling if c.compress_ratios[layer]else None)
        raw=mx.array(arrays['reference_raw'],mx.float32);mx.eval(raw,weight);observe()
        def reference_input_a():return grouped_layout(rotate_qat(rounded(raw),rope,'fp8',start,inverse=True))
        grouped_policy=capture_spec.get('extra_source_policy',{}).get('deepseek_source_bf16_loads','reference')
        def grouped(x):return owning_grouped(mx,x,wo_a,weight,grouped_policy,maybe_project)
        def input_b(value):return fused_quantize(rounded(value).reshape(1,length,8192),'fp8')
        def project_b(value):return mx.quantized_matmul(value,q,scales=s,mode='mxfp8')
        ref_a=reference_input_a();plain_a=grouped_layout(rounded(fused_quantize(rotate(rounded(raw),rope,start,inverse=True),'fp8')))
        ref_product=grouped(ref_a);ref_b=input_b(ref_product);ref_output=project_b(ref_b)
        ref_boundary=rounded(ref_output);mx.eval(ref_a,plain_a,ref_product,ref_b,ref_output,ref_boundary)
        direct=(grouped_policy in ('woa_f32','direct_f32')and length==1 and supports(ref_a,wo_a,'wo_a'))
        row=dict(capture=case,layout=p,grouped_reference_path='source_bf16_grouped_direct_f32'if direct else'official_bf16_grouped_f32',
            captured_grouped_policy=grouped_policy,reference_controls=dict(attention_owner=raw_details(np,arrays['reference_bf16'],np.array(rounded(raw))),
            rotation_QAT_equations=raw_details(np,np.array(plain_a),np.array(ref_a))),variants={},
            trajectory_scope='derived chain seeded byactualattention raw input; no actualWoA outputcapture')
        section['cases'].append(row);write()
        if not all(v['exact']for v in row['reference_controls'].values()):
            raise ValueError('Output-chain owning reference equations differ')
        if cross:
            other,other_arrays=cross[(layer,case['phase'])]
            if other['length']!=length or other['frontier']!=start:raise ValueError('Output-chain crosscapture phase/frontier mismatch')
            check=raw_details(np,other_arrays['prepared_input'],np.array(ref_b));row['cross_run_prepared_WoB']=check
            if check['exact']:
                row['trajectory_scope']='PreparedWoB exact across actualseparate-run captures; intermediateWoAraw remains reconstructed'
                row['reference_controls']['actual_WoB_raw']=raw_details(np,other_arrays['reference_raw'],np.array(ref_output))
                row['reference_controls']['actual_WoB_boundary']=raw_details(np,other_arrays['reference_bf16'],np.array(ref_boundary))
                if not all(v['exact']for v in row['reference_controls'].values()):raise ValueError('Crosscapture exactinput WoB output mismatch')
        functions={'reference':lambda:rounded(project_b(input_b(grouped(reference_input_a()))))}
        for name,use_a,use_b in variants:
            a=candidate.prepare_wo_a(raw,rope,start)if use_a else reference_input_a()
            product=grouped(a);b=candidate.prepare_wo_b(product)if use_b else input_b(product)
            output=project_b(b);boundary=rounded(output);mx.eval(a,product,b,output,boundary)
            checks=dict(grouped_input=raw_details(np,np.array(ref_a),np.array(a)),raw_WoA=raw_details(np,np.array(ref_product),np.array(product)),
                rounded_WoA=raw_details(np,np.array(rounded(ref_product)),np.array(rounded(product))),
                prepared_WoB=raw_details(np,np.array(ref_b),np.array(b)),raw_WoB=raw_details(np,np.array(ref_output),np.array(output)),
                final_boundary=raw_details(np,np.array(ref_boundary),np.array(boundary)))
            exact=all(v['exact']for v in checks.values());row['variants'][name]=dict(checks=checks,exact=exact,timing_admitted=exact,
                observed_scope='bounded derived producerconsumer graph; no modeluse')
            if exact:
                def pipeline(use_a=use_a,use_b=use_b):
                    aa=candidate.prepare_wo_a(raw,rope,start)if use_a else reference_input_a()
                    value=grouped(aa);bb=candidate.prepare_wo_b(value)if use_b else input_b(value)
                    return rounded(project_b(bb))
                functions[name]=pipeline
            del a,product,b,output,boundary;observe();write()
        row['preparation_timings']={};row['preparation_timing_scope']='Residentcapturedrawattention and reconstructedrawWoA; excludesmatrixproducts'
        if row['variants'].get('packed_inverse_QAT',{}).get('exact',False):
            row['preparation_timings']['wo_a_input']=compare(mx,np,{'reference':reference_input_a,
                'packed_inverse_QAT':lambda:candidate.prepare_wo_a(raw,rope,start)},repeats=repeats)
        if row['variants'].get('WoB_input_QAT',{}).get('exact',False):
            row['preparation_timings']['wo_b_input']=compare(mx,np,{'reference':lambda:input_b(ref_product),
                'WoB_input_QAT':lambda:candidate.prepare_wo_b(ref_product)},repeats=repeats)
        del ref_a,plain_a,ref_product,ref_b,ref_output,ref_boundary
        row['timings']=compare(mx,np,functions,repeats=repeats);observe();write()
        mx.synchronize();del functions,raw,weight;mx.clear_cache()
    section['required_reference_controls_exact']=all(all(v['exact']for v in r['reference_controls'].values())for r in section['cases'])
    section['verdicts']={name+'_all6_exact':all(r['variants'][name]['exact']for r in section['cases'])for name,a,b in variants}
    section['cross_capture_prepared_inputs_exact']=all(r.get('cross_run_prepared_WoB',{}).get('exact',False)for r in section['cases'])
    section['complete']=section['harness_completed']=True;write();return section
