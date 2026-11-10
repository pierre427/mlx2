"""Bounded actual-weight Source dense components; caller owns the GPU lease.

No MLX import, model load or GPU work on import. run() receives the existing
atomic evidence writer and parity-gated timing helper from the parent probe.
Allocator counters are not DRAM traffic measurements or whole-model rates.
"""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
import gc
import hashlib
import importlib.util
from pathlib import Path
import sys
import time


LIMIT = 8 << 30
ROWS = (1, 2, 3, 7, 99)


def run(mx, np, artifact, out, report, compare, atomic):
    from mlx2.adapters.deepseek_v41_source_artifact import read_tensors
    from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights
    from mlx2.runtime.gguf import source_dense
    from mlx2.runtime.models.deepseek_v41_quant import quantize, fused_quantize
    if mx.default_device() != mx.gpu:
        raise RuntimeError('Source dense components require caller-owned Metal GPU')
    if 'dense_components' in report:
        raise ValueError('Refusing to overwrite dense component evidence')
    out = Path(out)
    rounded = lambda value:value.astype(mx.bfloat16).astype(mx.float32)
    rng = np.random.default_rng(410252026)
    section = report['dense_components'] = {
        'checkpoint':'DeepSeek-V4.1-Flash-1017-Source@'+artifact['identity']['fingerprint'],
        'scope':'Actual layer0 WoA, actual full head, actual cropped shared gate/up; synthetic finite activations; no model inference',
        'qualified':False,'quality_established':False,'allocation_ceiling_bytes':LIMIT,
        'timing_scope':'Raw F32 parity first; only exact variants timed, interleaved3; setup and allocator probes excluded',
        'memory_scope':'MLX allocator active/cache/peak bytes, not physical residency, process footprint or DRAM traffic',
        'memory_peak_resets':True,'prior_peak_mlx_bytes':int(mx.get_peak_memory()),
        'maximum_observed_peak_mlx_bytes':int(mx.get_peak_memory()),
        'loads':[],'cases':[],'errors':[],
    }
    reference_path = out/'dense-provider-reference.py'
    ReferenceWeights = None
    if reference_path.exists():
        digest = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        module_name = 'mlx2.adapters._ds41_round2_dense_reference_'+digest[:12]
        spec = importlib.util.spec_from_file_location(module_name,reference_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        ReferenceWeights = module.SourceV41Weights
        section['frozen_provider_reference'] = {'path':str(reference_path),'sha256':digest,
            'use':'Frozen Source linear/gate_up methods for cropped FP8 projection oracle'}
    else:
        section['frozen_provider_reference'] = {'available':False,
            'use':'Explicit ordinary Source formula: quantize FP8 then F32 MXFP8 QMM, no output rounding'}

    def write():
        atomic(out/'results.json',report)

    def observe_peak():
        value = int(mx.get_peak_memory())
        section['maximum_observed_peak_mlx_bytes'] = max(section['maximum_observed_peak_mlx_bytes'],value)
        report['component_peak_mlx_bytes'] = max(report.get('component_peak_mlx_bytes',0),
            section['maximum_observed_peak_mlx_bytes'])
        if value > LIMIT:
            write()
            raise MemoryError('Source dense component exceeded 8 GiB MLX allocator ceiling')

    def clear():
        mx.synchronize()
        gc.collect()
        mx.clear_cache()
        observe_peak()

    def allocation(function):
        # Resident input/weight buffers are deliberately retained. No outputs
        # from previous parity/timing calls survive this function's entry.
        clear()
        before = int(mx.get_active_memory())
        cached_before = int(mx.get_cache_memory())
        observe_peak()
        mx.reset_peak_memory()
        result = function()
        mx.eval(result)
        active = int(mx.get_active_memory())
        peak = int(mx.get_peak_memory())
        observe_peak()
        value = {'resident_active_before_bytes':before,'active_after_eval_bytes':active,
            'active_delta_bytes':active-before,'peak_active_bytes':peak,
            'peak_increment_over_resident_bytes':max(0,peak-before),
            'cache_before_bytes':cached_before,'cache_after_bytes':int(mx.get_cache_memory()),
            'scope':'Allocator probe only; no timing, bus-byte or materialization claim'}
        del result
        clear()
        value['active_after_output_release_bytes'] = int(mx.get_active_memory())
        return value

    def raw(function):
        value = function()
        mx.eval(value)
        array = np.ascontiguousarray(np.array(value))
        del value
        return array

    def details(expected, actual):
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            return {'exact':False,'shape':list(actual.shape),'dtype':str(actual.dtype),
                'reason':'shape or dtype differs from reference'}
        a,b = expected.view(np.uint32).ravel(),actual.view(np.uint32).ravel()
        different = np.flatnonzero(a != b)
        result = {'exact':not len(different),'shape':list(actual.shape),'dtype':str(actual.dtype),
            'mismatched_f32_words':int(len(different)),
            'output_sha256':hashlib.sha256(actual.tobytes()).hexdigest(),
            'max_abs_error':float(np.max(np.abs(actual-expected)))}
        if len(different):
            i = int(different[0])
            result['first_raw_mismatch'] = {'flat_index':i,
                'coordinate':[int(v) for v in np.unravel_index(i,actual.shape)],
                'reference_f32_bits':f'{int(a[i]):08x}','candidate_f32_bits':f'{int(b[i]):08x}',
                'reference_value':float(expected.ravel()[i]),'candidate_value':float(actual.ravel()[i])}
        return result

    def record(name, functions, **metadata):
        row = {'name':name,**metadata,'raw':{},'allocator':{},'variants':{},'exact':True}
        section['cases'].append(row)
        write()
        expected = raw(functions['reference'])
        available = {}
        for variant,function in functions.items():
            try:
                actual = expected if variant == 'reference' else raw(function)
                row['raw'][variant] = details(expected,actual)
                if variant != 'reference':del actual
                available[variant] = function
                if not row['raw'][variant]['exact']:row['exact'] = False
            except (ValueError,NotImplementedError) as error:
                row['raw'][variant] = {'exact':False,'error':str(error),'unsupported':True}
                row['exact'] = False
            observe_peak()
        del expected
        clear()
        # Parent helper independently gates timing on bitwise raw equality.
        # Refuse to give it any candidate already known to differ.
        admitted = {key:fn for key,fn in available.items() if row['raw'][key]['exact']}
        row['variants'] = compare(mx,np,admitted,repeats=3)
        for variant in functions:
            if variant not in admitted:
                row['variants'][variant] = {'exact':False,'samples':[],
                    'timing_skipped':'Raw F32 parity failed or shape unsupported'}
        for variant,function in available.items():
            row['allocator'][variant] = allocation(function)
        observe_peak()
        write()
        return row

    def read(names, label):
        before = int(mx.get_active_memory())
        start = time.perf_counter_ns()
        host = read_tensors(artifact,names)
        done = time.perf_counter_ns()
        row = {'name':label,'tensors':list(names),'read_ms':(done-start)/1e6,
            'staged_host_payload_bytes':sum(value.nbytes for value in host.values()),
            'active_before_prepare_bytes':before,
            'tensor_metadata':{name:{'shape':list(artifact['tensors'][name]['shape']),
                'dtype':artifact['tensors'][name]['dtype'],'size_bytes':artifact['tensors'][name]['size']}
                for name in names}}
        section['loads'].append(row)
        write()
        return host,row

    def woa():
        name = artifact['aliases']['blk.0.attn_output_a.weight']
        scale_name = name.removesuffix('weight')+'scale'
        host,load = read((name,scale_name),'official_layer0_wo_a')
        start = time.perf_counter_ns()
        q = mx.array(host[name].view('<u4'))
        scales = mx.repeat(mx.array(host[scale_name]),32,axis=0)[:8192]
        weight = mx.dequantize(q,scales=scales,mode='mxfp8',dtype=mx.bfloat16)
        mx.eval(weight)
        load['prepare_ms'] = (time.perf_counter_ns()-start)/1e6
        load['prepared_dtype'] = 'bfloat16'
        load['prepared_shape'] = list(weight.shape)
        load['logical_prepared_coefficient_bytes'] = weight.size*2
        load['active_after_prepare_bytes'] = int(mx.get_active_memory())
        del host,q,scales
        clear();write()
        for rows in ROWS:
            x = rounded(quantize(rounded(mx.array(rng.normal(0,.25,(1,rows,8,4096)).astype(np.float32))),'fp8'))
            mx.eval(x)
            functions={
                'reference':lambda:mx.einsum('btgi,goi->btgo',x.astype(mx.float32),
                    weight.astype(mx.float32).reshape(8,1024,4096)),
                'direct_bf16_f32':lambda:source_dense.project(x,weight,'wo_a')}
            if rows>1:
                from mlx2.runtime.gguf import source_dense_legacy
                functions['legacy_bf16_f32']=lambda:source_dense_legacy.project(x,weight)
            record('grouped_wo_a_raw',functions,
                rows=rows,weight_shape=[8192,4096],groups=8,
                input_profile='BF16-rounded then group32 FP8 QAT then BF16-rounded F32 storage',
                candidate_plan=source_dense.plan('wo_a',rows),
                ordinary_logical_cast_input_bytes=8192*4096*2,
                ordinary_logical_cast_result_bytes=8192*4096*4,
                cast_materialization='Allocator peak differences are evidence; logical expressions alone do not prove DRAM traffic')
            x=None
            clear()
        weight=None

    def head():
        name = artifact['aliases']['output.weight']
        host,load = read((name,),'official_full_head')
        if artifact['tensors'][name]['dtype'] != 'BF16':
            raise ValueError('Source head is no longer BF16')
        start = time.perf_counter_ns()
        weight = mx.array(host[name]).view(mx.bfloat16)
        mx.eval(weight)
        load['candidate_prepare_ms'] = (time.perf_counter_ns()-start)/1e6
        load['active_after_candidate_prepare_bytes'] = int(mx.get_active_memory())
        del host
        clear()
        before = int(mx.get_active_memory());start = time.perf_counter_ns()
        reference_weight = weight.astype(mx.float32)
        mx.eval(reference_weight)
        load['reference_expansion_ms'] = (time.perf_counter_ns()-start)/1e6
        load['reference_expansion_allocator_delta_bytes'] = int(mx.get_active_memory())-before
        load['candidate_coefficient_bytes'] = weight.size*2
        load['reference_coefficient_bytes'] = reference_weight.size*4
        load['reference_expansion_scope'] = 'Prepared once like ordinary Source loader, not charged to each head matmul'
        load['parity_setup_holds_both_profiles'] = True
        write();observe_peak()
        # Only M1 is admitted by Source's direct-head profile. Wider NAX was
        # already nonexact in r1; repeating its large temporary allocations is
        # unnecessary for the legacy-WoA/HC repair gate.
        load['admitted_head_rows']=[1]
        load['wider_head_evidence']='ds41-source-round2-components-20261025-r1 (nonexact; unselected)'
        for rows in (1,):
            # Head consumes normalized BF16-rounded hidden; it has no FP8 QAT.
            x = rounded(mx.array(rng.normal(0,.25,(1,rows,5120)).astype(np.float32)))
            mx.eval(x)
            record('head_raw',{
                'reference':lambda:x.astype(mx.float32)@reference_weight.T,
                'f32_cast_each_call_diagnostic':lambda:x.astype(mx.float32)@weight.astype(mx.float32).T,
                'direct_bf16_f32':lambda:source_dense.project(x,weight,'head')},
                rows=rows,weight_shape=[129280,5120],input_profile='BF16-rounded finite synthetic hidden; no head activation QAT',
                candidate_plan=source_dense.plan('head',rows),
                reference_scope='Current ordinary Source head retains F32 coefficients prepared at startup',
                cast_diagnostic_scope='Additional diagnostic BF16-to-F32 cast per call; not the current ordinary loader policy',
                allocator_resident_setup='BF16 and prepared F32 coefficient tables held concurrently only for component parity/timing')
            x=None
            clear()
        reference_weight=None;weight=None

    def dense_qat():
        gate_alias,up_alias = 'blk.0.ffn_gate_shexp.weight','blk.0.ffn_up_shexp.weight'
        names = [artifact['aliases'][alias] for alias in (gate_alias,up_alias)]
        scales = [name.removesuffix('weight')+'scale' for name in names]
        host,load = read(tuple(names+scales),'cropped_shared_gate_up_mxfp8')
        start = time.perf_counter_ns()
        packed = {}
        for alias,name,scale in zip((gate_alias,up_alias),names,scales):
            if artifact['tensors'][name]['dtype'] != 'F8_E4M3':
                raise ValueError('Source shared gate/up is no longer FP8')
            n = min(256,host[name].shape[0]);n -= n%32
            q = mx.array(host[name][:n].copy().view('<u4'))
            s = mx.repeat(mx.array(host[scale][:n//32].copy()),32,axis=0)[:n]
            mx.eval(q,s)
            packed[alias] = (q,s)
        del host,q,s
        load['prepare_ms'] = (time.perf_counter_ns()-start)/1e6
        load['crop'] = 'First256 output rows; original FP8 codes and repeated group32 E8M0 scales, no weight requantization'
        load['packed_output_rows'] = 256
        clear();write()
        def provider(cls,policy):
            value = cls.__new__(cls)
            value._quant = packed
            value._ensure_dense = value._ready = lambda:None
            value.scheduled_dense_calls = {}
            value.source_dense_qat_policy = policy
            value.source_bf16_loads_policy = 'reference'
            return value
        reference = provider(ReferenceWeights,'reference') if ReferenceWeights is not None else None
        candidate = provider(SourceV41Weights,'fused_fp32')
        def ordinary_pair(x):
            if reference is not None:
                return mx.stack(reference.gate_up(x,gate_alias,up_alias))
            return mx.stack([mx.quantized_matmul(quantize(x,'fp8').astype(mx.float32),q,scales=s,mode='mxfp8')
                for q,s in packed.values()])
        def ordinary_linear(x):
            if reference is not None:return reference.linear(gate_alias)(x)
            q,s = packed[gate_alias]
            return mx.quantized_matmul(quantize(x,'fp8').astype(mx.float32),q,scales=s,mode='mxfp8')
        for rows in (1,7,99):
            x = rounded(mx.array(rng.normal(0,.25,(1,rows,5120)).astype(np.float32)))
            mx.eval(x)
            record('source_dense_input_qat',{'reference':lambda:quantize(x,'fp8').astype(mx.float32),
                'fused':lambda:fused_quantize(x.astype(mx.float32),'fp8')},rows=rows)
            record('source_dense_linear_qat_raw',{'reference':lambda:ordinary_linear(x),
                'fused':lambda:candidate.linear(gate_alias)(x)},rows=rows,
                original_tensor=names[0],cropped_output_rows=256)
            row = record('source_dense_gate_up_reused_qat_raw',{'reference':lambda:ordinary_pair(x),
                'fused_reused':lambda:mx.stack(candidate.gate_up(x,gate_alias,up_alias))},rows=rows,
                original_tensors=names,cropped_output_rows=256)
            row['scheduled_candidate_counters_cumulative'] = dict(candidate.scheduled_dense_calls)
            if reference is not None:row['scheduled_reference_counters_cumulative'] = dict(reference.scheduled_dense_calls)
            x=None
            clear();write()
        reference=None;candidate=None;packed=None

    write()
    # Each helper's local closures die on return. WoA is freed before loading
    # the full head, and the full head before cropped FP8 projection probes.
    for name,function in (('wo_a',woa),('head',head),('dense_qat',dense_qat)):
        try:
            function()
        except (ValueError,NotImplementedError) as error:
            section['errors'].append({'section':name,'error':str(error),
                'failure_kind':('harness_error' if 'ndarray is not C-contiguous' in str(error)
                    else 'input_or_function_error_requires_classification'),
                'scope':'Section did not finish; numerical mismatch is recorded separately per cell; remaining sections continued'})
            write()
        clear()
    section['all_raw_cases_exact'] = all(row['exact'] for row in section['cases']) and not section['errors']
    section['all_raw_exact'] = section['all_raw_cases_exact']
    qat = [row for row in section['cases'] if row['name'].startswith('source_dense_')]
    width_one = [row for row in section['cases'] if row['name'] in ('grouped_wo_a_raw','head_raw') and row['rows']==1]
    section['source_qat_raw_exact'] = len(qat)==9 and all(row['exact'] for row in qat)
    section['bf16_width_one_raw_exact'] = len(width_one)==2 and all(row['exact'] for row in width_one)
    section['required_raw_exact'] = section['source_qat_raw_exact'] and section['bf16_width_one_raw_exact']
    section['required_raw_exact_scope'] = 'Source fused/reused QAT at1/7/99 and intended width-one direct WoA/head; wider BF16 candidates exploratory only'
    section['direct_bf16_exact_shapes'] = {kind:[row['rows'] for row in section['cases']
        if row['name']==name and row['raw'].get('direct_bf16_f32',{}).get('exact',False)]
        for kind,name in (('wo_a','grouped_wo_a_raw'),('head','head_raw'))}
    section['legacy_bf16_exact_shapes']=[row['rows'] for row in section['cases']
        if row['name']=='grouped_wo_a_raw' and row['raw'].get('legacy_bf16_f32',{}).get('exact',False)]
    section['completed'] = True
    observe_peak();write()
    return section
