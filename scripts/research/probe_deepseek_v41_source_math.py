#!/usr/bin/env python3
"""Resident Source expert math diagnostic; no serving or quality qualification.

Replay real layer-zero route IDs against their real mixed-precision weights,
using synthetic BF16-rounded activations/scores. All objects are loaded before
timing. Candidate functions are exact, recorded textual variants of the owning
provider, keeping its gather, projection, rounding and reduction structure.
"""
import argparse
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import textwrap
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from mlx2.desktop_campaign_worker import atomic, check_ownership
from mlx2.desktop_runner import source_manifest, digest


def candidates(module, out):
    original = (out / "provider-reference.py").read_text()
    functions = {}
    for mode in ("reference", "hoisted", "fused", "hoisted_fused"):
        code = original
        if "hoisted" in mode:
            needle = "    values = np.asarray(ids)"
            assert code.count(needle) == 1
            code = code.replace(needle, "    all_selected = quantize(x, 'fp8').astype(mx.float32)\n" + needle)
            needle = 'selected = quantize(x[rows], "fp8").astype(mx.float32)'
            assert code.count(needle) == 1
            code = code.replace(needle, "selected = all_selected[rows]")
        if "fused" in mode:
            needle = "from ..runtime.models.deepseek_v41_quant import quantize"
            assert code.count(needle) == 1
            code = code.replace(needle, "from ..runtime.models.deepseek_v41_quant import fused_quantize as quantize")
        namespace = dict(vars(module))
        exec(compile(code, str(out / (mode + ".py")), "exec"), namespace)
        functions[mode] = namespace["moe"]
        (out / (mode + ".py")).write_text(code)
    (out / "reference.py").write_text(original)
    return functions


def dispatch_candidates(module, out):
    original = (out / "provider-reference.py").read_text()
    namespace = dict(vars(module))
    exec(compile(original, str(out / "provider-reference.py"), "exec"), namespace)
    frozen = namespace["moe"]
    def wrapped(policy, method):
        def call(weights, *args, **kwargs):
            weights.expert_math_policy = policy
            return method(weights, *args, **kwargs)
        return call
    return {
        "reference": wrapped("hoisted_fused_fp32", frozen),
        "ordered": wrapped("ordered_fused_fp32", module.SourceV41Weights.moe),
        "paired_q1": wrapped("paired_q1_fused_fp32", module.SourceV41Weights.moe),
        "paired": wrapped("paired_fused_fp32", module.SourceV41Weights.moe),
    }


def fusion_candidates(module,out):
    original=(out/'provider-reference.py').read_text();namespace=dict(vars(module))
    exec(compile(original,str(out/'provider-reference.py'),'exec'),namespace)
    def reference(w,*args,**kwargs):
        w.expert_math_policy='paired_fused_fp32';return namespace['moe'](w,*args,**kwargs)
    def fused(w,*args,**kwargs):
        w.expert_math_policy='epilogue_fused_fp32';return module.SourceV41Weights.moe(w,*args,**kwargs)
    def routed(w,*args,**kwargs):
        w.expert_math_policy='routed_fused_fp32';return module.SourceV41Weights.moe(w,*args,**kwargs)
    return {'reference':reference,'epilogue':fused,'routed':routed}


def dispatch_components(mx, np, bank, out, report):
    from mlx2.adapters.deepseek_v41_source_math import ordered_combine
    from mlx2.runtime.gguf.source_q1 import matmul, paired_matmul
    rng = np.random.default_rng(4102518)
    report["combine"] = []
    for rows, topk, duplicates in ((1,1,False),(1,6,False),(1,6,True),(7,6,False),(99,6,True)):
        ids = np.tile(np.arange(topk,dtype=np.uint32)[::-1],(rows,1))
        if duplicates: ids %= 3
        order = np.argsort(ids.ravel(),kind="stable")
        route = rng.normal(size=(rows*topk,65)).astype(np.float32)
        # BF16-representable cancellation cases expose a reordered reduction.
        special = np.array([2**25,1,-2**25,-0.0,2**-120,-2**-120],np.float32)
        route[:,0] = np.resize(special,rows*topk)
        route[:,1] = -0.0
        packed = mx.array(route[order]).astype(mx.bfloat16).astype(mx.float32)
        sorted_ids = ids.ravel()[order]
        cuts = np.flatnonzero(np.diff(sorted_ids))+1
        parts = list(mx.split(packed,cuts.tolist(),axis=0))
        def reference():
            restored=packed[mx.array(np.argsort(order),mx.uint32)].reshape(rows,topk,65)
            ranking=mx.array(np.argsort(ids,axis=1,kind="stable"),mx.uint32)
            total=mx.zeros((rows,65),mx.float32)
            for slot in range(topk):
                total=total+mx.take_along_axis(restored,ranking[:,slot:slot+1,None],axis=1)[:,0]
            return total
        value=compare(mx,np,{"reference":reference,"ordered":lambda:ordered_combine(parts,ids,order)})
        report["combine"].append({"rows":rows,"topk":topk,"duplicates":duplicates,"variants":value})
        atomic(out/'results.json',report)
    report["q1_pair"] = []
    data = next(d for d in bank.values() if 'gate.biases' in d)
    args = tuple(data[kind+field] for kind in ('gate','up') for field in ('.qweight','.scales','.biases'))
    for rows in (1,2,3,7,32,99):
        x=mx.array(rng.normal(0,.25,(rows,args[0].shape[1]*32)).astype(np.float32)).astype(mx.bfloat16).astype(mx.float32)
        mx.eval(x)
        value=compare(mx,np,{
            'reference':lambda:mx.stack([matmul(x,*args[:3]),matmul(x,*args[3:])]),
            'paired':lambda:mx.stack(paired_matmul(x,*args)),
        })
        report['q1_pair'].append({'rows':rows,'variants':value});atomic(out/'results.json',report)


def timed(mx, function):
    start = time.perf_counter_ns()
    value = function()
    built = time.perf_counter_ns()
    mx.eval(value)
    end = time.perf_counter_ns()
    return value, {"wall_ms": (end-start)/1e6, "graph_build_ms": (built-start)/1e6,
                   "completion_ms": (end-built)/1e6}


def original_pair_components(mx, np, bank, out, report):
    from mlx2.runtime.gguf.source_mxfp4_pair import paired_matmul
    from mlx2.runtime.models.deepseek_v41_quant import fused_quantize
    rng = np.random.default_rng(4102519)
    data = next(d for d in bank.values() if 'gate.biases' not in d)
    args = tuple(data[kind+field] for kind in ('gate','up') for field in ('.qweight','.scales'))
    report['original_pair'] = []
    for rows in range(1,13):
        x = mx.array(rng.normal(0,.25,(rows,5120)).astype(np.float32)).astype(mx.bfloat16).astype(mx.float32)
        x = fused_quantize(x,'fp8').astype(mx.float32); mx.eval(x)
        value = compare(mx,np,{
            'reference':lambda:mx.stack([mx.quantized_matmul(x,args[0],scales=args[1],mode='mxfp4'),mx.quantized_matmul(x,args[2],scales=args[3],mode='mxfp4')]),
            'paired':lambda:mx.stack(paired_matmul(x,*args,verified_m5_vector_band=True)),
        })
        report['original_pair'].append({'rows':rows,'variants':value});atomic(out/'results.json',report)


def compare(mx, np, functions, repeats=9):
    expected = functions["reference"]()
    mx.eval(expected)
    expected_bytes = np.array(expected).tobytes()
    result = {}
    for mode, function in functions.items():
        actual = function(); mx.eval(actual)
        values = np.array(actual)
        result[mode] = {"exact": values.tobytes() == expected_bytes,
                        "max_abs_error": float(np.max(np.abs(values-np.array(expected)))),
                        "output_sha256": hashlib.sha256(values.tobytes()).hexdigest(), "samples": []}
        del actual, values
    # Warm compilation and pipelines, then interleave modes in a seeded order.
    admitted = [m for m in functions if result[m]["exact"]]
    for mode in admitted:
        for _ in range(2):
            value, _ = timed(mx, functions[mode]); del value
    rng = np.random.default_rng(4102517)
    for _ in range(repeats):
        for mode in rng.permutation(admitted):
            value, sample = timed(mx, functions[mode]); del value
            result[mode]["samples"].append(sample)
    for row in result.values():
        if row["samples"]:
            row["median"] = {k: statistics.median(x[k] for x in row["samples"])
                             for k in row["samples"][0]}
    return result


def unit(out):
    spec = json.loads((out / "spec.json").read_text())
    if spec.get('experiment')=='round2':
        from ds41_round2_unit import run
        return run(out,ROOT)
    report = {**spec, "passed": False, "qualification": False, "qat": [], "moe": []}
    mx = None
    try:
        report["ownership"] = check_ownership(spec)
        if digest(source_manifest(ROOT)) != spec["source_snapshot_sha256"]:
            raise RuntimeError("source snapshot changed before probe")
        from mlx2.runtime.os_memory import host_memory_snapshot
        host = host_memory_snapshot()
        if host is None or host.available_bytes < (15 << 30):
            raise MemoryError("resident component requires 6 GiB process plus 9 GiB available reserve")
        report["host_available_before"] = host.available_bytes
        from probe_deepseek_v41_gpu import environment
        report["before"] = environment()
        import mlx.core as mx
        import numpy as np
        from mlx2.adapters import deepseek_v41_source_weights as owner
        from mlx2.adapters.deepseek_v41_source_artifact import inspect_artifact, read_tensors
        from mlx2.adapters.deepseek_v41_source_lfu import load_q1_manifest, q1_read
        from mlx2.runtime.models.deepseek_v41_quant import quantize, fused_quantize
        mx.set_default_device(mx.gpu); mx.set_memory_limit(6 << 30); mx.set_cache_limit(128 << 20)
        artifact = inspect_artifact(spec["model"])
        if artifact["identity"]["fingerprint"] != spec["original_fingerprint"]:
            raise ValueError("original artifact changed")
        hot = frozenset(tuple(v) for v in json.loads((out / "hotmap.json").read_text())["experts"])
        qroot, records, binding = load_q1_manifest(spec["q1_root"], spec["original_fingerprint"])
        if binding != spec["q1_fingerprint"]: raise ValueError("Q1 bank changed")
        routes = json.loads((out / "routes.json").read_text())
        expert_ids = sorted({e for row in routes.values() for ids in row for e in ids})
        bank = {}; objects = []
        for expert in expert_ids:
            if spec.get('experiment') in ('fusion','prefill'):
                from mlx2.adapters.deepseek_v41_source_buffers import read_original_packed,borrow_q1_packed,materialize
                host=(read_original_packed(artifact,0,expert) if (0,expert) in hot
                      else borrow_q1_packed(q1_read(qroot,records[(0,expert)])))
                data=materialize(host);mx.eval(data.backing,*data.values());del host
                kind='original' if (0,expert) in hot else 'q1'
            elif (0,expert) in hot:
                raw = read_tensors(artifact, owner.expert_names(0,expert)); data = {}
                for kind, part in owner.PROJECTIONS.items():
                    prefix = f"layers.0.ffn.experts.{expert}.{part}."
                    data[kind+'.qweight'] = mx.array(raw[prefix+'weight'].view('<u4'))
                    data[kind+'.scales'] = mx.array(raw[prefix+'scale'])
                kind = "original"
            else:
                raw = q1_read(qroot, records[(0,expert)])
                data = {n:mx.array(v) for n,v in raw.items()}; kind = "q1"
            mx.eval(*data.values()); bank[expert] = data
            objects.append({"expert":expert,"kind":kind,"bytes":sum(v.nbytes for v in data.values())})
            if 'raw' in locals():del raw
            del data
        report["resident_objects"] = objects
        report["resident_payload_bytes"] = sum(v['bytes'] for v in objects)
        atomic(out/'results.json', report)
        dispatch = spec.get('experiment') == 'dispatch'
        fusion = spec.get('experiment') == 'fusion'
        prefill = spec.get('experiment') == 'prefill'
        functions = {} if prefill else fusion_candidates(owner,out) if fusion else dispatch_candidates(owner,out) if dispatch else candidates(owner, out)
        if prefill:
            if spec.get('prefill_components')=='down-missing':
                from ds41_prefill_down_components import run_down_missing
                run_down_missing(mx,np,bank,artifact,out,report,compare,atomic)
            else:
                from ds41_prefill_components import run_gu
                run_gu(mx,np,bank,artifact,out,report,compare,atomic)
            if spec.get('prefill_components','all')=='all':
                from ds41_prefill_down_components import run_down
                run_down(mx,np,bank,artifact,out,report,compare,atomic)
                from ds41_prefill_wide_components import run_wide
                run_wide(mx,np,bank,artifact,out,report,compare,atomic)
        if fusion:
            from ds41_fusion_components import run
            run(mx,np,bank,artifact,out,report,compare,atomic)
        if dispatch: dispatch_components(mx,np,bank,out,report)
        if spec.get('probe_original_pair'): original_pair_components(mx,np,bank,out,report)
        rng = np.random.default_rng(4102517)
        hidden, intermediate = artifact['args'].hidden_size, artifact['args'].moe_intermediate_size
        for rows in (() if dispatch or fusion or prefill else (1,7,32,99)):
            for width in (hidden,intermediate):
                x = mx.array(rng.normal(0,.25,(rows,width)).astype(np.float32)).astype(mx.bfloat16).astype(mx.float32)
                mx.eval(x)
                value = compare(mx,np,{'reference':lambda:quantize(x,'fp8'), 'fused':lambda:fused_quantize(x,'fp8')})
                report['qat'].append({'rows':rows,'width':width,'variants':value})
                atomic(out/'results.json',report)
        weights = object.__new__(owner.SourceV41Weights); weights._ready = lambda:None
        weights.artifact = {'args':SimpleNamespace(hidden_size=hidden,num_experts_per_tok=6)}
        weights.expert_stream = SimpleNamespace(_active=True,acquire=lambda prefix,e:bank[e],stats={'lookups':0},packed_buffers=fusion)
        weights.scheduled_expert_calls = {}
        for name, route in (() if prefill else routes.items()):
            ids = np.array(route,np.uint32); rows = ids.shape[0]
            x = mx.array(rng.normal(0,.25,(rows,hidden)).astype(np.float32)).astype(mx.bfloat16).astype(mx.float32)
            scores = mx.array(rng.uniform(.05,.95,ids.shape).astype(np.float32)); mx.eval(x,scores)
            call = {k:(lambda fn=fn:fn(weights,x,ids,scores,'blk.0')) for k,fn in functions.items()}
            value = compare(mx,np,call)
            report['moe'].append({'name':name,'rows':rows,'experts':len(np.unique(ids)),
                'route_sha256':hashlib.sha256(ids).hexdigest(),'variants':value})
            atomic(out/'results.json',report)
        report['peak_mlx_bytes'] = mx.get_peak_memory()
        report['after'] = environment()
        report['passed'] = all(v['exact'] for row in report['qat']+report['moe']+report.get('combine',[])+report.get('q1_pair',[])+report.get('original_pair',[])+report.get('fusion',[])+report.get('prefill_gu',[])+report.get('prefill_down',[])+report.get('prefill_combine',[]) for v in row['variants'].values())
        if prefill:
            required=(('prefill_down',) if spec.get('prefill_components')=='down-missing' else
                      ('prefill_gu',) if spec.get('prefill_components')=='gu' else ('prefill_gu','prefill_down','prefill_combine'))
            if not all(report.get(k) for k in required):
                raise ValueError('prefill probe required component inventory missing')
            if report['peak_mlx_bytes']>spec['memory_limit_bytes']:
                raise MemoryError('prefill probe allocator high-water mark exceeded admitted component limit')
        report['passed'] &= all(r['exact'] and r['one_backing_bound'] for r in report.get('packed_objects',[]))
        if digest(source_manifest(ROOT)) != spec['source_snapshot_sha256']:
            raise RuntimeError('source snapshot changed during probe')
    except BaseException as error:
        report.update(passed=False,error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    finally:
        if mx is not None:
            mx.synchronize(); gc.collect(); mx.clear_cache()
        atomic(out/'results.json',report)
    print(json.dumps({'passed':report['passed'],'moe':report['moe'],'error':report.get('error')}),flush=True)
    return 0 if report['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('run','unit'));parser.add_argument('--round2',action='store_true');parser.add_argument('--prefill',action='store_true');parser.add_argument('--prefill-components',choices=('gu','all','down-missing'),default='all')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--dispatch',action='store_true');parser.add_argument('--original-pair',action='store_true');parser.add_argument('--fusion',action='store_true');args=parser.parse_args();out=args.output.resolve()
    if args.prefill and (args.round2 or args.dispatch or args.fusion or args.original_pair):
        parser.error('--prefill is a separate bounded component experiment')
    if args.mode=='unit': return unit(out)
    out.mkdir(parents=True,exist_ok=False)
    from mlx2.runtime.os_memory import host_memory_snapshot
    host=host_memory_snapshot()
    if host is None or host.available_bytes < ((17 if args.round2 else 15)<<30):raise MemoryError('component memory admission refused')
    baseline=ROOT/'qualification/runs'/('ds41-source-lfu75-hot60-q1-dispatch-paired-20261025-r1' if args.fusion else 'ds41-source-lfu75-hot60-q1-math-20261025-r1' if args.dispatch else 'ds41-source-lfu75-hot60-q1-optimized-20261025-r1')
    prior=json.loads((baseline/'spec.json').read_text());routes={}
    provider=(baseline/'source/src/mlx2/adapters/deepseek_v41_source_weights.py').read_text()
    if args.round2:(out/'dense-provider-reference.py').write_text(provider)
    provider_tree=ast.parse(provider)
    provider_class=next(n for n in provider_tree.body if isinstance(n,ast.ClassDef) and n.name=='SourceV41Weights')
    method=next(n for n in provider_class.body if isinstance(n,ast.FunctionDef) and n.name=='moe')
    method_text=textwrap.dedent('\n'.join(provider.splitlines()[method.lineno-1:method.end_lineno]))+'\n'
    (out/'provider-reference.py').write_text(method_text)
    for name,filename in [('prefill99','boundary-0066.json'),('decode1','boundary-0068.json')]:
        routes[name]=json.loads((baseline/filename).read_text())['routing'][0]['expert_ids']
    routes['prefill7']=routes['prefill99'][:7];atomic(out/'routes.json',routes)
    shutil.copyfile(ROOT/'.local-source-lfu/hot60-map.json',out/'hotmap.json')
    manifest=source_manifest(ROOT);sha=digest(manifest);source=out/'source';source.mkdir()
    for relative in manifest:
        target=source/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,target)
    assert digest(source_manifest(source))==sha
    spec={'out':str(out),'job_id':out.name,'source_revision':prior['source_revision'],'source_snapshot_sha256':sha,
        'checkpoint':prior['checkpoint'],'original_fingerprint':'c30fd25e3cc79ba7044686fa3f9a7a1c0f514f77ee6293d2f5f62bc409e02ddb',
        'model':prior['model'],'q1_root':prior['expert_policy']['q1_root'],
        'q1_fingerprint':'09508a4061700fbfc06f37f024e4dbee027e7bd90bd1fe6ecc38ec8b3fc27289',
        'hotmap_sha256':hashlib.sha256((out/'hotmap.json').read_bytes()).hexdigest(),
        'routes_sha256':hashlib.sha256((out/'routes.json').read_bytes()).hexdigest(),
        'provider_reference_sha256':hashlib.sha256((out/'provider-reference.py').read_bytes()).hexdigest(),
        'scope':'Resident layer0 math only; actual weights and route IDs; synthetic BF16 activations/scores; no model inference',
        'experiment':'prefill' if args.prefill else 'round2' if args.round2 else 'fusion' if args.fusion else 'dispatch' if args.dispatch else 'activation_qat',
        'prefill_components':args.prefill_components if args.prefill else None,
        'numerical_environment':{'MLX_ENABLE_TF32':'0'},
        'probe_original_pair':args.original_pair,
        'mcp_url':'http://127.0.0.1:8766/mcp','thermal_controlled':False,'thermal_policy':'user waived gating; recorded',
        'memory_limit_bytes':(8 if args.round2 else 6)<<30,'host_available_before':host.available_bytes,'kv_apc_server':'not applicable to operator probe'}
    if args.round2:
        spec['scope']='Actual Source dense coefficients with synthetic BF16/F32 activations; synthetic real-geometry KV/index and epilogue operators; no model inference or expert-cache performance claim'
    atomic(out/'spec.json',spec);atomic(out/'source.json',manifest)
    from mlx2.desktop_campaign_lease import DesktopCPGLease
    from mlx2.desktop_campaign_process import spawn
    result=1;child=None
    try:
        with DesktopCPGLease(out,out.name,checkpoint=spec['checkpoint'],source_revision=spec['source_revision']) as lease:
            command=[sys.executable,str(source/'scripts/run_with_gpu_locks.py'),'--session',lease.owner['cpg_session'],
                '--label',out.name,'--desktop-cpg-lease',str(lease.path),'--receipt',str(out/'paired-locks.json'),'--',
                sys.executable,str(source/'scripts/research/probe_deepseek_v41_source_math.py'),'unit','--output',str(out)]
            try:
                with (out/'process.log').open('x') as log:
                    child=spawn(command,env=dict(os.environ,PYTHONPATH=str(source/'src'),MLX_ENABLE_TF32='0'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    atomic(out/'process.json',{'pid':child.pid,'command':command});deadline=time.monotonic()+360
                    while child.poll() is None:
                        lease.check()
                        if time.monotonic()>deadline:raise TimeoutError('bounded resident math probe deadline')
                        time.sleep(1)
                    result=child.returncode;lease.success=result==0
            finally:
                if child is not None:child.stop(timeout=15)
    finally:
        atomic(out/'release-verified.json',{'finished_at':time.time(),
            'owners_absent':all(not Path(p).exists() for p in ['/Users/Shared/mlxuag/gpu.lock/owner.json','/tmp/gpu.lock/owner.json']),
            'source_unchanged':digest(source_manifest(source))==sha})
    return result


if __name__=='__main__': raise SystemExit(main())
