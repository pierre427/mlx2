#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Real-weight per-layer 27B decode gate; verify checks the counted reference fallback.

This never selects or qualifies a serving route. No MLX import, model load,
or Metal work occurs before --i-own-the-gpu and the TF32 guard are checked.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--layers', nargs='+', type=int,
                    help='default: every GDN layer; subsets are partial coverage')
    ap.add_argument('--steps', type=int, default=3)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--i-own-the-gpu', action='store_true')
    args = ap.parse_args()
    if not args.i_own_the_gpu:
        ap.error('refusing Metal execution without --i-own-the-gpu')
    if os.environ.get('MLX_ENABLE_TF32') != '0':
        ap.error('set MLX_ENABLE_TF32=0 (the adapter pins it)')
    if args.steps < 1:
        ap.error('--steps must be positive')
    sys.path.insert(0, str(ROOT / 'src'))
    import mlx.core as mx
    import mlx.nn as nn
    from mlx2.runtime.models.qwen38_fused_gdn import GatedDeltaNet, VERIFY_REFUSAL
    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.cache import ArraysCache
    from mlx2.runtime.models.qwen4_fused_gdn import fused_gdn_runtime_supported

    mx.set_default_device(mx.gpu)
    if not fused_gdn_runtime_supported():
        raise RuntimeError('Metal runtime unavailable')
    model_path = Path(args.model).expanduser().resolve()
    config_bytes = (model_path / 'config.json').read_bytes()
    index_bytes = (model_path / 'model.safetensors.index.json').read_bytes()
    config = json.loads(config_bytes)
    text = TextModelArgs.from_dict(config.get('text_config', config))
    if (text.linear_num_key_heads, text.linear_num_value_heads,
        text.linear_key_head_dim, text.linear_value_head_dim,
        text.linear_conv_kernel_dim) != (16, 48, 128, 128, 4):
        raise ValueError('requires the production 27B GDN geometry')
    weight_map = json.loads(index_bytes)['weight_map']
    expected_layers = [i for i in range(text.num_hidden_layers)
                       if (i + 1) % text.full_attention_interval != 0]
    indices = expected_layers if args.layers is None else args.layers
    if not indices or len(set(indices)) != len(indices) or any(i not in expected_layers for i in indices):
        raise ValueError('--layers must name unique GDN layers')
    report = {'schema': 'mlx2.qwen38-fused-gdn-check.v1', 'model': str(model_path),
              'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
              'index_sha256': hashlib.sha256(index_bytes).hexdigest(),
              'source_sha256': {}, 'seed': args.seed, 'steps': args.steps,
              'all_layers_covered': indices == expected_layers,
              'qualified': False, 'route_selected': False,
              'verify_fused_implemented': False, 'verify_refusal': VERIFY_REFUSAL,
              'passed': False, 'layers': {}}
    for name in ('qwen38_fused_gdn.py', 'qwen4_fused_gdn.py', 'qwen3_5.py', 'gated_delta.py'):
        report['source_sha256'][name] = hashlib.sha256(
            (ROOT / 'src/mlx2/runtime/models' / name).read_bytes()).hexdigest()
    mx.random.seed(args.seed)

    def load_layer(i):
        prefix = f'language_model.model.layers.{i}.linear_attn.'
        if not any(k.startswith(prefix) for k in weight_map):
            prefix = f'model.language_model.layers.{i}.linear_attn.'
        shards = sorted({v for k, v in weight_map.items() if k.startswith(prefix)})
        if not shards:
            raise ValueError(f'no safetensors for layer {i}')
        weights = {}
        for shard in shards:
            loaded = mx.load(str(model_path / shard))
            weights.update({k[len(prefix):]: v for k, v in loaded.items() if k.startswith(prefix)})
            del loaded
        conv = weights['conv1d.weight']
        if conv.shape[-1] != 1:
            weights['conv1d.weight'] = conv.moveaxis(2, 1)
        obj = GatedDeltaNet(text)
        quant = config.get('quantization', config.get('quantization_config'))
        if quant:
            def predicate(path, module):
                if prefix + path in quant:
                    return quant[prefix + path]
                return hasattr(module, 'to_quantized') and path + '.scales' in weights
            nn.quantize(obj, group_size=quant['group_size'], bits=quant['bits'],
                        mode=quant.get('mode', 'affine'), class_predicate=predicate)
        obj.load_weights(list(weights.items()), strict=True)
        obj.eval()
        mx.eval(obj.parameters())
        return obj, shards

    def fresh(conv=None, state=None):
        c = ArraysCache(size=2)
        c[0], c[1] = conv, state
        return c

    def same(a, b):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        unsigned = mx.uint16 if a.dtype.size == 2 else mx.uint32
        return bool(mx.all(mx.isfinite(a)).item() and mx.all(mx.isfinite(b)).item()
                    and mx.array_equal(a.view(unsigned), b.view(unsigned)).item())

    def compare(left, right):
        return dict(zip(('output', 'conv_state', 'recurrent_state'),
                        (same(a, b) for a, b in zip(left, right))))

    def decode(obj, x, start, enabled):
        obj.set_fused_gdn_enabled(enabled)
        c = fresh(*start)
        outs = []
        for t in range(args.steps):
            y = obj(x[t], cache=c)
            mx.eval(y, c[0], c[1])
            outs.append(y)
        return mx.concatenate(outs, axis=1), c[0], c[1]

    def verify(obj, x, start, enabled, accepted):
        obj.set_fused_gdn_enabled(enabled)
        c = fresh(*start)
        c.start_speculation(rollback_window=32)
        y = obj(x, cache=c)
        mx.eval(y, c[0], c[1])
        c.trim(x.shape[1] - accepted)
        mx.eval(c[0], c[1])
        c.stop_speculation()
        return y, c[0], c[1]

    try:
        for i in indices:
            obj, shards = load_layer(i)
            entry = {'safetensors': shards, 'decode': {}, 'verify_reference_fallback': {}}
            report['layers'][str(i)] = entry
            for rows in (1, 2, 4, 8, 16):
                obj.set_fused_gdn_enabled(False)
                c = fresh()
                warm = mx.random.normal((rows, 5, text.hidden_size)).astype(mx.bfloat16)
                y = obj(warm, cache=c)
                mx.eval(y, c[0], c[1])
                start = (c[0], c[1])
                x = mx.random.normal((args.steps, rows, 1, text.hidden_size)).astype(mx.bfloat16)
                ref = decode(obj, x, start, False)
                counter = 'decode_calls' if rows == 1 else 'batch_decode_calls'
                before = obj.fused_gdn_counters[counter]
                candidate = decode(obj, x, start, True)
                check = compare(ref, candidate)
                check['fused_calls'] = obj.fused_gdn_counters[counter] - before
                check['passed'] = all(check[k] for k in ('output', 'conv_state', 'recurrent_state')) and check['fused_calls'] == args.steps
                entry['decode'][str(rows)] = check
                for width in (3, 9, 17):
                    x = mx.random.normal((rows, width, text.hidden_size)).astype(mx.bfloat16)
                    checks = []
                    for accepted in sorted({0, 1, width // 2, width - 1, width}):
                        ref = verify(obj, x, start, False, accepted)
                        before = obj.fused_gdn_counters['reasons'].get(VERIFY_REFUSAL, 0)
                        candidate = verify(obj, x, start, True, accepted)
                        check = compare(ref, candidate)
                        check.update(accepted=accepted, counted_reference_fallback=
                                     obj.fused_gdn_counters['reasons'].get(VERIFY_REFUSAL, 0) == before + 1)
                        check['passed'] = all(check[k] for k in ('output', 'conv_state', 'recurrent_state', 'counted_reference_fallback'))
                        checks.append(check)
                    entry['verify_reference_fallback'][f'{rows}x{width}'] = checks
            entry['counters'] = dict(obj.fused_gdn_counters)
            entry['passed'] = all(c['passed'] for c in entry['decode'].values()) and all(
                c['passed'] for checks in entry['verify_reference_fallback'].values() for c in checks)
            print(json.dumps({'layer': i, 'passed': entry['passed']}), flush=True)
            del obj
            mx.clear_cache()
        report['passed'] = all(e['passed'] for e in report['layers'].values())
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        output_path = Path(args.out).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
