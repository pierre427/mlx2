"""Actual-operand HC partial-composite replay; caller owns the6GiB GPU unit."""
import hashlib
import importlib.util
import json
from pathlib import Path

LIMIT=6<<30
INPUTS=('branch','residual','post','comb','pre','norm_weight')


def options(report,available):
    """Bounded replication options; omitted options preserve the original run."""
    value=report.get('hc_boundary_options',{})
    if not isinstance(value,dict) or set(value)-{'repeats','policies'}:
        raise ValueError('HC boundary unknown replication options')
    repeats=value.get('repeats',3);policies=value.get('policies',list(available))
    if type(repeats)is not int or not 3<=repeats<=7:
        raise ValueError('HC boundary repeats must be3..7')
    if (not isinstance(policies,(list,tuple))or not policies
            or any(type(p)is not str or p not in available for p in policies)
            or len(set(policies))!=len(policies)):
        raise ValueError('HC boundary policies must be unique known choices')
    return repeats,tuple(policies)


def capture_module(capture_path):
    """Validate the capture-owned parser seal before importing local code."""
    path=Path(capture_path)
    if path.is_symlink():raise ValueError('HC capture path symlink')
    if path.is_dir():path=path/'hc-boundary-capture.json'
    if path.is_symlink():raise ValueError('HC capture manifest symlink')
    unit=path.parent;spec_path=unit/'spec.json';parser=unit/'harness/hc_boundary_probe.py'
    if spec_path.is_symlink()or parser.is_symlink():raise ValueError('HC parser/spec symlink')
    manifest=json.loads(path.read_text());spec=json.loads(spec_path.read_text())
    expected=spec.get('harness_sha256',{}).get('hc_boundary_probe.py')
    if not expected or manifest.get('helper_sha256')!=expected or hashlib.sha256(parser.read_bytes()).hexdigest()!=expected:
        raise ValueError('HC parser seal mismatch')
    for key in ('checkpoint','source_revision','source_snapshot_sha256'):
        if not manifest.get(key)or manifest[key]!=spec.get(key):raise ValueError('HC capture identity mismatch')
    module_spec=importlib.util.spec_from_file_location('_ds41_sealed_hc_boundary_capture',parser)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    return module,path


def restored_inputs(mx,np,arrays,*,packet=False):
    """Upload exact values; optional source-packet view reconstruction is inferred."""
    values={k:mx.array(arrays[k])for k in INPUTS}
    if packet:
        rows=arrays['branch'].shape[1]
        host=np.concatenate((arrays['pre'],arrays['post'],arrays['comb'].reshape(1,rows,16)),-1)
        owner=mx.array(host)
        values.update(pre=owner[...,:4],post=owner[...,4:8],comb=owner[...,8:].reshape(1,rows,4,4))
    mx.eval(*values.values())
    return [values[k]for k in INPUTS],dict(kind='inferred_Source_split24_packet_views'if packet else'ordinary_captured_value_upload',
        actual_device_strides_observed=False,
        explanation='Source split returns24-channel slices, but serialized Npy proves values only; saved-boundary replay remains required')


def run_components(mx,np,capture_path,out,report,compare,atomic):
    """Six actual cases, raw/BF16 frontiers and exact-only isolated timings."""
    if mx.default_device()!=mx.gpu:raise RuntimeError('HC components require caller-owned Metal')
    if 'hc_boundary_components'in report:raise ValueError('Refusing HC evidence overwrite')
    from mlx2.runtime.models import deepseek_v41_source_hc_boundary as candidate
    from ds41_attention_components import raw_details,timed_interleaved
    repeats,policies=options(report,candidate.COLLAPSE_POLICIES)
    parser,path=capture_module(capture_path);manifest,cases=parser.load_capture(path)
    if any(report.get(k)!=manifest[k]for k in('checkpoint','source_revision')):
        raise ValueError('HC component/capture identity mismatch')
    if len(cases)!=6 or {(c['layer'],c['phase'])for c,a in cases}!={(l,p)for l in(2,20,21)for p in('pp99','decode1')}:
        raise ValueError('HC components require six actual captured cases')
    out=Path(out)
    section=report['hc_boundary_components']=dict(schema='mlx2.source-hc-boundary-components.v1',
        checkpoint=manifest['checkpoint'],source_revision=manifest['source_revision'],capture_path=str(path),
        capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        capture_source_snapshot_sha256=manifest['source_snapshot_sha256'],
        component_source_snapshot_sha256=report['source_snapshot_sha256'],
        scope='Actual HC attention→FFN operands; partial residual/input composite only; coefficient projection/Sinkhorn outside',
        maximum_allocator_bytes=LIMIT,candidate_used_by_model=False,qualified=False,performance_qualified=False,
        timing_scope=f'Prepared-input isolated operators;{repeats}rotated repetitions after all saved/raw/BF16 gates; not serving critical-path',
        repeats=repeats,policies=list(policies),
        common_compare_callback_used=False,cases=[],complete=False,harness_completed=False,
        required_reference_controls_exact=False,verdicts={})
    def write():atomic(out/'results.json',report)
    def observe():
        peak=int(mx.get_peak_memory());section['peak_mlx_bytes']=max(section.get('peak_mlx_bytes',0),peak)
        report['component_peak_mlx_bytes']=max(report.get('component_peak_mlx_bytes',0),peak)
        if peak>LIMIT:write();raise MemoryError('HC components exceeded6GiB allocator ceiling')
    def host(values):return [np.array(v)for v in values]
    def saved_checks(arrays,outputs,frontiers):
        return dict(updated_h=raw_details(np,arrays['updated_h'],np.array(outputs[0])),
                    collapsed=raw_details(np,arrays['collapsed'],np.array(frontiers['collapsed_bf16'])),
                    normalized=raw_details(np,arrays['normalized'],np.array(outputs[1])))
    write();observe()
    for case,arrays in cases:
        eps=case.get('norm_eps',1e-6)
        row=dict(layer=case['layer'],phase=case['phase'],frontier=case['frontier'],capture=case,
                 reference_controls={},variants={})
        section['cases'].append(row)
        args,layout=restored_inputs(mx,np,arrays)
        ref,front=candidate.reference(*args,eps=eps,return_frontiers=True);mx.eval(ref,front)
        row['reference_controls']['ordinary_upload_saved']=saved_checks(arrays,ref,front)
        if not all(v['exact']for v in row['reference_controls']['ordinary_upload_saved'].values()):
            # Preserve the failed ordinary layout control. This reconstruction
            # follows the owning split's slices, without asserting GPU aliasing.
            args,layout=restored_inputs(mx,np,arrays,packet=True)
            ref,front=candidate.reference(*args,eps=eps,return_frontiers=True);mx.eval(ref,front)
            row['reference_controls']['inferred_packet_saved']=saved_checks(arrays,ref,front)
            if not all(v['exact']for v in row['reference_controls']['inferred_packet_saved'].values()):
                write();raise ValueError('HC owning reference replay differs from saved boundaries')
        row['prepared_layout']=layout
        row['component_plan']=candidate.plan(*args,eps=eps)
        reference_outputs=host(ref);reference_frontiers={k:np.array(v)for k,v in front.items()}
        del ref,front
        admitted={'reference':lambda:candidate.reference(*args,eps=eps)}
        for policy in policies:
            result,frontiers=candidate.boundary(*args,eps=eps,collapse_policy=policy,return_frontiers=True)
            mx.eval(result,frontiers)
            checks={name:raw_details(np,expected,np.array(actual))for name,expected,actual in
                    zip(('updated_h','normalized_router','prepared_FP8'),reference_outputs,result)}
            stage_checks={k:raw_details(np,reference_frontiers[k],np.array(v))for k,v in frontiers.items()}
            normal=candidate.boundary(*args,eps=eps,collapse_policy=policy);mx.eval(normal)
            normal_checks={name:raw_details(np,expected,np.array(actual))for name,expected,actual in
                           zip(('updated_h','normalized_router','prepared_FP8'),reference_outputs,normal)}
            exact=all(v['exact']for v in checks.values())and all(v['exact']for v in stage_checks.values())and all(v['exact']for v in normal_checks.values())
            row['variants'][policy]=dict(checks=checks,stage_checks=stage_checks,normal_output_checks=normal_checks,
                all_stage_and_output_exact=exact,timing_admitted=exact,
                inline_reduction_unproved_until_this_case=policy!='stock_frontier',
                changed_numerical_profile=not exact)
            if exact:admitted[policy]=lambda policy=policy:candidate.boundary(*args,eps=eps,collapse_policy=policy)
            del result,frontiers,normal
            observe();write()
        row['timings']=timed_interleaved(mx,np,admitted,repeats=repeats)
        observe();write()
    section['required_reference_controls_exact']=True
    section['verdicts']={p+'_all_six_exact':all(c['variants'][p]['all_stage_and_output_exact']for c in section['cases'])for p in policies}
    section.update(complete=True,harness_completed=True)
    observe();write();return section
