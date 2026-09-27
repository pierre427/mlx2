"""Isolated rsqrt A/B for the compact-tape GDN verification route.

Derives only the rsqrt expressions/name from the local MIT-origin GDN source.
The local source/provenance, unchanged launch function and helpers are hashed.
No production mutation. Run only via cpg_job.py and owned_exec.py.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import types
from common import require_ownership, compare_arrays, time_variants, sha256
from gdn_bench import ROOT, clone_function, variant_source, make_inputs, chain_call, compiled_chain_call, admitted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260925)
    parser.add_argument('--rounds', type=int, default=31)
    parser.add_argument('--chain', type=int, default=16)
    args = parser.parse_args()
    ownership = require_ownership()
    import mlx.core as mx
    import numpy as np
    from mlx2.runtime.models import qwen4_fused_gdn_verify as module
    from mlx2.runtime.models import qwen4_fused_gdn as decode_module
    factory = clone_function(module._replay_verify_kernel.__wrapped__, {
        'mx': types.SimpleNamespace(fast=types.SimpleNamespace(metal_kernel=lambda **kw: kw))})()
    variants, sources = {}, {}
    for arm in ('precise', 'fast_qk', 'fast_output', 'fast_all'):
        source, sites = variant_source(factory['source'], arm, 3)
        kernel = mx.fast.metal_kernel(**{**factory, 'source': source,
                                       'name': 'research_rsqrt_compact_' + arm})
        variants[arm] = clone_function(module.qwen4_fused_gdn_replay_verify, {
            '_replay_verify_kernel': lambda kernel=kernel: kernel})
        sources[arm] = {'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                        'header_sha256':hashlib.sha256(factory['header'].encode()).hexdigest(), 'sites': sites}
    source_path = Path(module.__file__)
    helper_path = Path(__file__).with_name('gdn_bench.py')
    result = {'completed': False, 'scope': 'compact replay verify, synthetic inputs; no model throughput/quality',
              'source_path': str(source_path), 'source_sha256': sha256(source_path),
              'header_module_path':decode_module.__file__, 'header_module_sha256':sha256(decode_module.__file__),
              'source_provenance': json.loads((ROOT/'provenance/flashnext.json').read_text())['models/qwen4_fused_gdn_verify.py'],
              'harness_sha256': sha256(__file__), 'gdn_helper_sha256': sha256(helper_path),
              'common_sha256': sha256(Path(__file__).with_name('common.py')),
              'repo_head': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'device': mx.metal.device_info(), 'mlx_version': importlib.metadata.version('mlx'),
              'ownership': ownership, 'seed': args.seed, 'variants': sources, 'cases': []}
    names = ['output', 'conv_state', 'recurrent_state', 'replay_keys', 'replay_corrections', 'replay_decay']

    def metrics(ref, candidate):
        values = {name: compare_arrays(r,c) for name,r,c in zip(names,ref,candidate,strict=True)}
        assert all(m['nonfinite_reference'] == m['nonfinite_candidate'] == 0 for m in values.values())
        return values

    def save():
        args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

    for steps in (2,4,8):
        ty = module.probe_qwen4_fused_gdn_replay_verify(mx.bfloat16, steps)
        assert ty is not None
        kwargs = {'threadgroup_y': ty}
        case = {'steps': steps, 'threadgroup_y':ty, 'accuracy':[], 'chain_calls_per_eval':args.chain}
        for seed in (args.seed,args.seed+1):
            for scale in (0.0,1e-4,1.0,8.0):
                inputs = make_inputs(mx,np,seed=seed,steps=steps,scale=scale)
                mx.eval(*inputs.values())
                admitted(module,inputs,'verify','qwen4')
                outputs = {arm:fn(**inputs,norm_eps=1e-6,**kwargs) for arm,fn in variants.items()}
                exported = module.qwen4_fused_gdn_replay_verify(**inputs,norm_eps=1e-6,**kwargs)
                parity = metrics(exported,outputs['precise'])
                assert all(m['bit_mismatches']==0 for m in parity.values())
                checks = {arm:metrics(outputs['precise'],out) for arm,out in outputs.items() if arm!='precise'}
                rollback = {}
                for accepted in sorted({0,1,steps//2,steps-1}):
                    states = {arm:module.qwen4_fused_gdn_reconstruct(inputs['recurrent_state'],*out[3:],
                              accepted=mx.array(accepted,dtype=mx.int32),threadgroup_y=ty) for arm,out in outputs.items()}
                    rollback[str(accepted)] = {arm:compare_arrays(states['precise'],state)
                                              for arm,state in states.items() if arm!='precise'}
                    assert all(m['nonfinite_reference']==m['nonfinite_candidate']==0
                               for m in rollback[str(accepted)].values())
                case['accuracy'].append({'seed':seed,'scale':scale,'exported_original_parity':parity,
                                        'vs_precise':checks,'rollback':rollback})
                if seed==args.seed and scale==1:
                    timing_inputs=inputs
        single = {arm:lambda fn=fn:fn(**timing_inputs,norm_eps=1e-6,**kwargs) for arm,fn in variants.items()}
        compiled = {arm:compiled_chain_call(mx,fn,timing_inputs,kwargs,args.chain) for arm,fn in variants.items()}
        case['compiled_chain_parity'] = {}
        for arm,fn in variants.items():
            parity = metrics(chain_call(fn,timing_inputs,kwargs,args.chain),compiled[arm]())
            assert all(m['bit_mismatches']==0 for m in parity.values())
            case['compiled_chain_parity'][arm]=parity
        case['single_timing']=time_variants(single,rounds=args.rounds,inner=3,seed=args.seed)
        case['compiled_timing']=time_variants(compiled,rounds=args.rounds,inner=3,seed=args.seed+1)
        result['cases'].append(case)
        save()
        print(json.dumps({'completed_steps':steps}),flush=True)
    assert result['source_sha256']==sha256(source_path)
    assert result['header_module_sha256']==sha256(decode_module.__file__)
    assert result['harness_sha256']==sha256(__file__)
    assert result['gdn_helper_sha256']==sha256(helper_path)
    assert result['common_sha256']==sha256(Path(__file__).with_name('common.py'))
    result['peak_memory_bytes']=mx.get_peak_memory()
    result['completed']=True
    save()


if __name__=='__main__':
    main()
