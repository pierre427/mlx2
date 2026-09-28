"""Compare rsqrt implementations in the unchanged local fused PP GroupRMSNorm.

Provenance: src/mlx2/runtime/models/qwen4_fused_group_norm.py, original mlx2
Apache-2.0 code. Kernel text is loaded at run time; only the rsqrt expression
and private kernel name change. No production file or routing flag is edited.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

from common import compare_arrays, require_ownership, sha256, time_variants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=20260925)
    parser.add_argument('--rounds', type=int, default=25)
    parser.add_argument('--chain', type=int, default=8)
    parser.add_argument('--rows', type=int, nargs='+', default=[1024, 4096, 16384])
    args = parser.parse_args()
    owner = require_ownership()
    import mlx.core as mx
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / 'src'))
    from mlx2.runtime.models import qwen4_fused_group_norm as original

    expression = 'metal::precise::rsqrt(mean + 1e-06f)'
    assert original._SOURCE.count(expression) == 1
    expressions = {
        'precise': expression,
        'fast': 'metal::fast::rsqrt(mean + 1e-06f)',
        'fast_nr': 'rsqrt_refine(mean + 1e-06f)',
    }
    refine = '''
inline float rsqrt_refine(float x) {
    float r = metal::fast::rsqrt(x);
    return r * metal::fma(-0.5f * x, r * r, 1.5f);
}
'''
    kernels = {}
    variants = {}
    for arm, replacement in expressions.items():
        source = original._SOURCE.replace(expression, replacement)
        header = original._HEADER + (refine if arm == 'fast_nr' else '')
        kernels[arm] = mx.fast.metal_kernel(
            name='research_pp_rsqrt_' + arm, input_names=['x', 'w'], output_names=['out'],
            header=header, source=source, ensure_row_contiguous=True)
        variants[arm] = {'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                         'header_sha256': hashlib.sha256(header.encode()).hexdigest(),
                         'rsqrt_expression': replacement}

    def launch(arm, x, weight):
        return kernels[arm](inputs=[x, weight], template=[('T', x.dtype)],
                            grid=(256, 1, x.shape[0] * 4), threadgroup=(256, 1, 1),
                            output_shapes=[x.shape], output_dtypes=[x.dtype])[0]

    def chain(arm, x, weight):
        for _ in range(args.chain):
            x = launch(arm, x, weight)
        return x

    def make_compiled(arm):
        return mx.compile(lambda x, weight: chain(arm, x, weight))

    compiled = {arm: make_compiled(arm) for arm in kernels}

    result = {'scope': 'fused PP GroupRMSNorm only; no model or end-to-end throughput',
              'device': mx.metal.device_info(), 'mlx_version': importlib.metadata.version('mlx'),
              'seed': args.seed, 'ownership': owner, 'variants': variants,
              'repo_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
              'source_path': original.__file__, 'source_sha256': sha256(original.__file__),
              'harness_sha256': sha256(__file__), 'common_sha256': sha256(Path(__file__).with_name('common.py')),
              'timing_cases': [], 'stress_cases': []}

    def save():
        Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')

    def accuracy(x, weight):
        outputs = {arm: launch(arm, x, weight) for arm in kernels}
        mx.eval(outputs)
        reference = original._launch(x, weight)
        fidelity = compare_arrays(reference, outputs['precise'])
        assert fidelity['bit_mismatches'] == 0, fidelity
        return {'original_vs_precise_clone': fidelity,
                **{arm: compare_arrays(outputs['precise'], outputs[arm])
                   for arm in ('fast', 'fast_nr')}}

    mx.random.seed(args.seed)
    weight = (1.0 + 0.05 * mx.random.normal((10240,))).astype(mx.bfloat16)
    mx.eval(weight)
    for rows in args.rows:
        assert rows in original.CANDIDATE_ROW_COUNTS
        x = (mx.random.normal((rows, 10240)) * 1.7).astype(mx.bfloat16)
        mx.eval(x)
        case = {'rows': rows, 'distribution': 'normal sd=1.7, weight=1+normal sd=.05',
                'accuracy': accuracy(x, weight)}
        case['timing'] = time_variants(
            {arm: lambda arm=arm, x=x: launch(arm, x, weight) for arm in kernels},
            rounds=args.rounds, inner=3, seed=args.seed + rows)
        case['compiled_chain_calls_per_eval'] = args.chain
        case['compiled_chain_parity'] = {}
        for arm in kernels:
            parity = compare_arrays(chain(arm, x, weight), compiled[arm](x, weight))
            assert parity['bit_mismatches'] == 0, (arm, parity)
            assert parity['nonfinite_candidate'] == parity['nonfinite_reference'] == 0
            case['compiled_chain_parity'][arm] = parity
        case['compiled_chain_timing'] = time_variants(
            {arm: lambda arm=arm, x=x: compiled[arm](x, weight) for arm in kernels},
            rounds=args.rounds, inner=3, seed=args.seed + rows + 1)
        result['timing_cases'].append(case)
        save()
        print(json.dumps({'rows': rows, 'accuracy': case['accuracy'],
                         'medians_us': {k: v['median_us'] for k, v in case['timing']['arms'].items()}}), flush=True)
        del x
        mx.clear_cache()

    # Stress the epsilon-dominated and large-magnitude domains without enormous allocations.
    for label, scale in [('zero', 0.0), ('tiny', 1e-4), ('small', .01), ('large', 32.0)]:
        x = (mx.random.normal((1024, 10240)) * scale).astype(mx.bfloat16)
        mx.eval(x)
        entry = {'case': label, 'rows': 1024, 'scale': scale, 'accuracy': accuracy(x, weight)}
        result['stress_cases'].append(entry)
        save()
        print(json.dumps({'stress': label, 'accuracy': entry['accuracy']}), flush=True)
        del x
        mx.clear_cache()
    assert result['source_sha256'] == sha256(original.__file__), 'source changed during benchmark'
    result['peak_memory_bytes'] = mx.get_peak_memory()
    result['completed'] = True
    save()


if __name__ == '__main__':
    main()
