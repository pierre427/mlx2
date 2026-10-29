# SPDX-License-Identifier: MIT
"""Opt-in actual native-MX geometry/cast-chain components; no import GPU work.

Provenance: deepseek-v41-native-mx-owned-probe.json. Default first-probe worker
is separate and unchanged. These are distinct native reduction profiles, not
serving defaults or model-quality tests.
"""
import hashlib
import json
from pathlib import Path

ROWS = (34, 98, 1, 5)
WIDE_TILES = ((16, 32), (16, 64), (32, 32), (32, 64))
TINY_TILES = ((16, 32), (8, 32))
LIMIT = 6 << 30


def expected_options():
    return dict(mode='geometry_sweep_v1', repeats=7, projection='down', actual_rows=list(ROWS),
                wide_tiles=[list(t) for t in WIDE_TILES], tiny_tiles=[list(t) for t in TINY_TILES])


def options(report):
    value = report.get('native_mx_options')
    expected = expected_options()
    if (report.get('geometry_sweep') is not True or type(value) is not dict or value != expected
            or any(type(value.get(k)) is not type(v) for k,v in expected.items())
            or any(type(v) is not int for v in value['actual_rows'])
            or any(type(tile) is not list or any(type(v) is not int for v in tile)
                   for key in ('wide_tiles','tiny_tiles') for tile in value[key])):
        raise ValueError('Invalid explicit native MX geometry-sweep options')
    return value


def tiles_for(rows):
    if type(rows) is not int or rows not in ROWS:
        raise ValueError('Geometry sweep requires actual selected row shape')
    return WIDE_TILES if rows in (34,98) else TINY_TILES


def engineering_verdicts(cases):
    variants=[v for c in cases for v in c.get('variants',{}).values()]
    inventory=tuple(c['capture']['rows'] for c in cases)==ROWS
    return dict(actual_reference_replay=inventory and all(c['reference_exact'] for c in cases),
        coefficient_identity=inventory and all(c['coefficient_identity'] and c['coefficient_gate']['eligible'] for c in cases),
        half_roundtrip=inventory and all(c['half_roundtrip']['finite'] and c['half_roundtrip']['bitwise_exact'] for c in cases),
        native_finite=bool(variants) and all(v['native_finite'] for v in variants),
        native_stable=bool(variants) and all(v['native_stable'] and v['canary_passed']
            and v['admitted_raw_exact'] and v['admitted_owner_exact'] for v in variants),
        actual_fp64_bound=bool(variants) and all(v['fp64_oracle']['native']['within_bound'] for v in variants)
            and all(c['fp64_reference']['within_bound'] for c in cases),
        native_dispatched=inventory and all(set(c['variants'])=={f'{a}x{b}' for a,b in tiles_for(c['capture']['rows'])}
            and all(v['native_dispatched'] for v in c['variants'].values()) for c in cases))


def run_components(mx,np,capture_dir,out,report,compare,atomic):
    if mx.default_device()!=mx.gpu:
        raise RuntimeError('Native MX geometry requires caller-owned GPU')
    cfg=options(report)
    if 'native_mx_geometry_components' in report:
        raise ValueError('Refusing geometry evidence overwrite')
    from ds41_down_actual_components import capture_module,check_coefficients,_materialize
    from ds41_native_mx_components import select_cases,half_roundtrip,coefficient_gate,dot_bounds,_error
    from ds41_native_mx_operands import inspect_case,oracle
    from ds41_prefill_wide_components import raw_details
    from ds41_attention_components import timed_interleaved
    from mlx2.adapters.deepseek_v41_source_artifact import inspect_artifact
    from mlx2.adapters.deepseek_v41_source_buffers import read_original_packed
    from mlx2.runtime.gguf import source_native_mx
    parser,path,capture_spec=capture_module(capture_dir)
    manifest,captured=parser.load_capture(path)
    for key in ('checkpoint','source_revision'):
        if report.get(key)!=manifest[key]:raise ValueError('Geometry capture/source identity mismatch')
    cases=select_cases(captured,rows=ROWS)
    artifact=inspect_artifact(capture_spec['model'])
    if artifact['identity']['fingerprint']!=manifest['original_artifact_fingerprint']:
        raise ValueError('Geometry original artifact changed')
    hot=Path(capture_spec['expert_policy']['hotset_path'])
    if hot.is_symlink() or hashlib.sha256(hot.read_bytes()).hexdigest()!=manifest['hot_map_binding']:
        raise ValueError('Geometry frozen precision map changed')
    hot_keys=frozenset(map(tuple,json.loads(hot.read_text())['experts']))
    section=report['native_mx_geometry_components']=dict(schema='mlx2.native-mx-geometry-components.v1',
        checkpoint=manifest['checkpoint'],source_revision=manifest['source_revision'],
        component_source_snapshot_sha256=report['source_snapshot_sha256'],
        capture_source_snapshot_sha256=manifest['source_snapshot_sha256'],
        capture_path=str(path),capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        original_artifact_fingerprint=manifest['original_artifact_fingerprint'],
        q1_bank_binding=manifest['q1_bank_binding'],hot_map_binding=manifest['hot_map_binding'],
        native_used_by_model=False,qualification=False,performance_qualified=False,
        maximum_allocator_bytes=LIMIT,loaded_experts_concurrently=1,extra_expert_acquisitions=0,
        explicit_device_scratch_bytes=0,implicit_tensorops_scratch='not exposed by SDK',
        scope='Actual original DOWN tile/cast-chain operators; no serving, quality or throughput claim',
        repetitions=cfg['repeats'],cases=[],complete=False,harness_completed=False,
        required_reference_controls_exact=False,verdicts={})
    out=Path(out);rounded=lambda x:x.astype(mx.bfloat16).astype(mx.float32)

    def write():
        peak=int(mx.get_peak_memory());section['peak_mlx_bytes']=max(section.get('peak_mlx_bytes',0),peak)
        report['component_peak_mlx_bytes']=max(report.get('component_peak_mlx_bytes',0),peak)
        atomic(out/'results.json',report)
        if peak>LIMIT:raise MemoryError('Native MX geometry allocator ceiling exceeded')

    write()
    for case,arrays in cases:
        if (case['layer'],case['expert']) not in hot_keys:
            raise ValueError('Geometry original outside frozen hot map')
        host=read_original_packed(artifact,case['layer'],case['expert']);views=host.views()
        check_coefficients(np,case,views)
        cpu=inspect_case(np,case,arrays,views,projection='down')
        coeff=coefficient_gate(np,views['down.qweight'],views['down.scales'])
        half=half_roundtrip(np,arrays['activation'])
        row=dict(capture=case,cpu_operands=cpu,coefficient_identity=True,coefficient_gate=coeff,
            half_roundtrip=half,reference_exact=False,variants={},
            staged_host_bytes=int(host.host_allocation_nbytes),
            timed_scope='Fresh resident graph, synchronized interleaved operator timing; no model critical path claim')
        section['cases'].append(row)
        if not coeff['eligible'] or not half['finite'] or not half['bitwise_exact']:
            write();raise ValueError('Geometry native operand eligibility failed')
        expert=_materialize(mx,host);x=mx.array(arrays['activation'],mx.float32);mx.eval(x,expert.backing)
        q,s=expert['down.qweight'],expert['down.scales']
        def stock():return mx.quantized_matmul(x,q,scales=s,mode='mxfp4')
        raw=stock();owner=rounded(raw);mx.eval(raw,owner)
        raw_cpu,owner_cpu=np.array(raw),np.array(owner)
        row['reference_controls']=dict(raw=raw_details(np,arrays['reference_raw'],raw_cpu),
                                      owner=raw_details(np,arrays['reference_boundary'],owner_cpu))
        row['reference_exact']=all(v['exact'] for v in row['reference_controls'].values())
        if not row['reference_exact']:
            write();raise ValueError('Geometry stock replay differs from actual capture')
        admission=source_native_mx.prepare_input(x)
        zero_admission=source_native_mx.prepare_input(mx.zeros_like(x))
        row['outside_timing_admission']=admission.receipt
        input_rows=tuple(sorted({0,case['rows']//2,case['rows']-1}));output_rows=(0,1,31,32,5119)
        truth=oracle(np,arrays['activation'],views['down.qweight'],views['down.scales'],input_rows=input_rows,output_rows=output_rows)
        bounds,metadata=dot_bounds(np,arrays['activation'],views['down.qweight'],views['down.scales'],input_rows=input_rows,output_rows=output_rows)
        ix=np.ix_(input_rows,output_rows)
        row['fp64_reference']=_error(np,truth,raw_cpu[ix],bounds)
        if not row['fp64_reference']['within_bound']:
            write();raise ValueError('Geometry stock F64 bound failed')
        functions={'reference':lambda:rounded(stock())}
        for tile in tiles_for(case['rows']):
            name=f'{tile[0]}x{tile[1]}'
            native=source_native_mx.project(admission,expert,projection='down',tile=tile)
            native_owner=rounded(native)
            repeated=source_native_mx.project(admission,expert,projection='down',tile=tile)
            admitted_raw=source_native_mx.project_from_admitted_raw(admission,expert,projection='down',tile=tile,rounded=False)
            admitted_owner=source_native_mx.project_from_admitted_raw(admission,expert,projection='down',tile=tile,rounded=True)
            mx.eval(native,native_owner,repeated,admitted_raw,admitted_owner)
            n,o=np.array(native),np.array(native_owner)
            zero=source_native_mx.project_from_admitted_raw(zero_admission,expert,projection='down',tile=tile,rounded=False)
            again=source_native_mx.project_from_admitted_raw(admission,expert,projection='down',tile=tile,rounded=False);mx.eval(zero,again)
            item=dict(tile=list(tile),native_dispatched=True,plan=source_native_mx.plan('down',case['rows'],tile=tile),
                native_finite=bool(np.isfinite(n).all() and np.isfinite(o).all()),
                native_stable=raw_details(np,n,np.array(repeated))['exact'],
                canary_passed=bool(np.all(np.array(zero)==0) and np.any(n!=0) and raw_details(np,n,np.array(again))['exact']),
                admitted_raw_exact=raw_details(np,n,np.array(admitted_raw))['exact'],
                admitted_owner_exact=raw_details(np,o,np.array(admitted_owner))['exact'],
                native_raw=raw_details(np,raw_cpu,n),native_owner=raw_details(np,owner_cpu,o),
                fp64_oracle=dict(input_rows=list(input_rows),output_rows=list(output_rows),bounds=bounds.tolist(),
                    bound_metadata=metadata,native=_error(np,truth,n[ix],bounds)),
                logical_activation_cast_bytes=case['rows']*2304*2,explicit_device_scratch_bytes=0)
            row['variants'][name]=item
            if not all(item[k] for k in ('native_finite','native_stable','canary_passed','admitted_raw_exact','admitted_owner_exact')) or not item['fp64_oracle']['native']['within_bound']:
                write();raise ValueError('Geometry native finite/stability/ABA/admitted/F64 gate failed')
            functions['prepared_'+name]=lambda tile=tile:rounded(source_native_mx.project(admission,expert,projection='down',tile=tile))
            functions['admitted_cast_project_owner_'+name]=lambda tile=tile:source_native_mx.project_from_admitted_raw(admission,expert,projection='down',tile=tile,rounded=True)
            del native,native_owner,repeated,admitted_raw,admitted_owner,zero,again,n,o
        row['timing']=timed_interleaved(mx,np,functions,repeats=cfg['repeats'])
        row['timing_scope']=dict(reference='Fresh stock F32 QMM+BF16/F32 owner',
            prepared='Fresh native QMM+BF16/F32 owner, excludes prior admitted half cast',
            admitted_cast_project_owner='Fresh contiguous half cast from immutable admitted raw+native QMM+owner; no diagnostic eval/item/roundtrip readbacks inside graph',
            diagnostic_complete='Excluded from geometry timing; unchanged old mode retains diagnostic complete API evidence')
        write();mx.synchronize()
        del expert,x,q,s,raw,owner,host,views,admission,zero_admission,functions,stock
        mx.clear_cache()
    section['verdicts']=engineering_verdicts(section['cases'])
    section['required_reference_controls_exact']=section['verdicts']['actual_reference_replay']
    variants=[v for c in section['cases'] for v in c['variants'].values()]
    section['all_native_raw_and_owner_exact']=bool(variants) and all(v['native_raw']['exact'] and v['native_owner']['exact'] for v in variants)
    section.update(complete=True,harness_completed=True);write();return section
