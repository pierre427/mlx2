#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""GPU bit gate for the Qwen3.8 27B fused GDN prefill (omlx #3903 kernels).

Two parts, both reference-vs-fused in one process on the real checkpoint:

* ``layers``: every GDN layer's real weights (or ``--layers``), one layer at
  a time.  Per row count, a cold chunk (``cache[0] is None``) and a warm chunk
  (state left by an eager 64-row chunk) run through the layer with the fused
  prefill off and on; the output, conv state and recurrent state must be
  bit-identical and the fused route must have run (``prefill_chunk_calls``).
  Layer 0 also compares the prework q/k/v/conv and the swish norm-gate
  against the eager ops directly.
* ``model``: the full model through the adapter.  Each prompt is prefilled in
  fixed chunks with the prefill switch off and on; the last chunk's logits,
  every cache entry (GDN conv/recurrent state, attention K/V) and a short
  greedy continuation must be bit-identical.

This never selects or qualifies a serving route.  No MLX import, model load
or Metal work happens before --i-own-the-gpu and the TF32 guard are checked.
Exit status 0 only when everything is bit-identical and engaged.

    MLX_ENABLE_TF32=0 PYTHONPATH=src .venv/bin/python \\
      scripts/check_qwen38_27b_fused_gdn_prefill.py --i-own-the-gpu \\
      --model ~/mlx-models/Qwen3.8-27B-oQ4e-mtp --out receipt.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROWS = (64, 512, 2048)
DEFAULT_CHUNKS = (512, 2048)
SOURCES = ('qwen38_fused_gdn.py', 'qwen4_fused_gdn_prefill.py', 'qwen4_fused_gdn.py',
           'qwen3_5.py', 'qwen3_next.py', 'gated_delta.py')


def _prompt_texts(root: Path):
    """Deterministic real-text prompts from the repository's own docs."""
    corpus = []
    for name in ('docs/SERVING.md', 'docs/QUALIFICATION.md', 'AGENTS.md', 'docs/PROVENANCE.md'):
        path = root / name
        if path.is_file():
            corpus.append(path.read_text(errors='replace'))
    text = '\n\n'.join(corpus)
    return {
        'short': 'Explain how a hash map handles collisions, in one paragraph.',
        'medium': text[:1500],
        'long': text[:24000],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--layers', nargs='+', type=int,
                    help='default: every GDN layer; subsets are partial coverage')
    ap.add_argument('--rows', nargs='+', type=int, default=list(DEFAULT_ROWS))
    ap.add_argument('--chunks', nargs='+', type=int, default=list(DEFAULT_CHUNKS))
    ap.add_argument('--continuation', type=int, default=8)
    ap.add_argument('--skip-layers', action='store_true')
    ap.add_argument('--skip-model', action='store_true')
    ap.add_argument('--seed', type=int, default=38)
    ap.add_argument('--i-own-the-gpu', action='store_true')
    args = ap.parse_args()
    if not args.i_own_the_gpu:
        ap.error('refusing Metal execution without --i-own-the-gpu')
    if os.environ.get('MLX_ENABLE_TF32') != '0':
        ap.error('set MLX_ENABLE_TF32=0 (the adapter pins it)')
    if any(r < 64 for r in args.rows):
        ap.error('--rows must be at least the 64-row prefill floor')
    sys.path.insert(0, str(ROOT / 'src'))
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    model_path = Path(args.model).expanduser().resolve()
    report = {
        'schema': 'mlx2.qwen38-fused-gdn-prefill-check.v1',
        'model': str(model_path), 'seed': args.seed,
        'mlx_version': getattr(mx, '__version__', None),
        'source_sha256': {
            name: hashlib.sha256(
                (ROOT / 'src/mlx2/runtime/models' / name).read_bytes()).hexdigest()
            for name in SOURCES
        },
        'qualified': False, 'route_selected': False,
        'passed': False,
    }
    output_path = Path(args.out).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # The adapter pins its import-time profile, so the full model is
        # built before the per-layer part imports any model module.
        if not args.skip_model:
            report['model_check'] = check_model(args, mx, model_path)
            mx.clear_cache()
        if not args.skip_layers:
            report['layers'] = check_layers(args, mx, model_path)
        parts = [report.get('layers'), report.get('model_check')]
        report['passed'] = all(p is None or p['passed'] for p in parts) and any(parts)
    except Exception as exc:  # noqa: BLE001 -- recorded in the receipt
        import traceback

        report['error'] = f'{type(exc).__name__}: {exc}'
        report['traceback'] = traceback.format_exc()
    finally:
        output_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': report['passed'], 'error': report.get('error')}))
    return 0 if report['passed'] else 1


def _same(mx, a, b):
    if a is None or b is None:
        return a is None and b is None
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    unsigned = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return bool(mx.array_equal(a.view(unsigned), b.view(unsigned)).item())


def _diff(mx, a, b):
    if a.shape != b.shape:
        return {'shape': [list(a.shape), list(b.shape)]}
    d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
    return {'max_abs': float(d.max().item()),
            'frac_diff': float((d > 0).astype(mx.float32).mean().item())}


def check_layers(args, mx, model_path):
    import mlx.nn as nn
    from mlx2.runtime.models.cache import ArraysCache
    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_fused_gdn import GatedDeltaNet
    from mlx2.runtime.models import qwen4_fused_gdn_prefill as prefill
    from mlx2.runtime.models.qwen4_fused_gdn import fused_gdn_runtime_supported

    if not fused_gdn_runtime_supported():
        raise RuntimeError('Metal runtime unavailable')
    config = json.loads((model_path / 'config.json').read_bytes())
    text = TextModelArgs.from_dict(config.get('text_config', config))
    if (text.linear_num_key_heads, text.linear_num_value_heads,
            text.linear_key_head_dim, text.linear_value_head_dim,
            text.linear_conv_kernel_dim) != (16, 48, 128, 128, 4):
        raise ValueError('requires the production 27B GDN geometry')
    weight_map = json.loads((model_path / 'model.safetensors.index.json').read_bytes())['weight_map']
    expected = [i for i in range(text.num_hidden_layers)
                if (i + 1) % text.full_attention_interval != 0]
    indices = expected if args.layers is None else args.layers
    if not indices or any(i not in expected for i in indices):
        raise ValueError('--layers must name GDN layers')
    mx.random.seed(args.seed)
    result = {'all_layers_covered': indices == expected, 'rows': args.rows,
              'layers': {}, 'components': None, 'passed': False}

    def load_layer(i):
        prefix = f'language_model.model.layers.{i}.linear_attn.'
        if not any(k.startswith(prefix) for k in weight_map):
            prefix = f'model.language_model.layers.{i}.linear_attn.'
        shards = sorted({v for k, v in weight_map.items() if k.startswith(prefix)})
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
        obj.set_fused_gdn_enabled(True)  # the served default on every route
        mx.eval(obj.parameters())
        return obj

    def run(obj, x, start, fused):
        obj.set_fused_gdn_prefill_enabled(fused)
        c = ArraysCache(size=2)
        c[0], c[1] = start
        before = obj.fused_gdn_counters['prefill_chunk_calls']
        y = obj(x, None, c)
        mx.eval(y, c[0], c[1])
        return (y, c[0], c[1]), obj.fused_gdn_counters['prefill_chunk_calls'] - before

    for n, i in enumerate(indices):
        obj = load_layer(i)
        entry = {}
        warm_x = (0.5 * mx.random.normal((1, 64, text.hidden_size))).astype(mx.bfloat16)
        warm, _ = run(obj, warm_x, (None, None), False)
        for rows in args.rows:
            x = (0.5 * mx.random.normal((1, rows, text.hidden_size))).astype(mx.bfloat16)
            for label, start in (('cold', (None, None)), ('warm', warm[1:])):
                ref, _ = run(obj, x, start, False)
                got, engaged = run(obj, x, start, True)
                check = dict(zip(('output', 'conv_state', 'recurrent_state'),
                                 (_same(mx, a, b) for a, b in zip(ref, got))))
                check['fused_calls'] = engaged
                check['passed'] = all(check[k] for k in ('output', 'conv_state',
                                                         'recurrent_state')) and engaged == 1
                if not check['passed']:
                    check['output_diff'] = _diff(mx, ref[0], got[0])
                entry[f'{label}/{rows}'] = check
        if n == 0:
            result['components'] = _components(mx, nn, obj, prefill, args.rows[-1])
        entry['reasons'] = dict(obj.fused_gdn_counters['reasons'])
        entry['passed'] = all(v['passed'] for k, v in entry.items()
                              if isinstance(v, dict) and 'passed' in v)
        result['layers'][str(i)] = entry
        print(json.dumps({'layer': i, 'passed': entry['passed']}), flush=True)
        del obj
        mx.clear_cache()
    result['passed'] = (all(e['passed'] for e in result['layers'].values())
                        and bool(result['components'] and result['components']['passed']))
    return result


def _components(mx, nn, obj, prefill, rows):
    """Prework q/k/v/conv and the norm-gate against the eager ops."""
    C = obj.conv_dim
    out = {}
    qkv = mx.random.normal((1, rows, C)).astype(mx.bfloat16)
    warm = mx.random.normal((1, 3, C)).astype(mx.bfloat16)
    for label, state in (('cold', None), ('warm', warm)):
        fused = prefill.qwen4_gdn_prefill_prework(qkv, state, obj.conv1d.weight,
                                                  numerics='qwen35')
        conv_state = mx.zeros((1, 3, C), qkv.dtype) if state is None else state
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        conv_out = nn.silu(obj.conv1d(conv_input))
        kd = obj.key_dim
        q = conv_out[..., :kd].reshape(1, rows, obj.num_k_heads, obj.head_k_dim)
        k = conv_out[..., kd:2 * kd].reshape(1, rows, obj.num_k_heads, obj.head_k_dim)
        v = conv_out[..., 2 * kd:].reshape(1, rows, obj.num_v_heads, obj.head_v_dim)
        q, k = obj._normalize_qk(q, k)
        eager = (q, k, v, mx.contiguous(conv_input[:, -3:, :]))
        mx.eval(*fused, *eager)
        for part, got, want in zip(('q', 'k', 'v', 'conv_out'), fused, eager):
            same = _same(mx, got, want)
            out[f'prework/{label}/{part}'] = same if same else _diff(mx, got, want)
    y = mx.random.normal((1, rows, obj.num_v_heads, obj.head_v_dim)).astype(mx.bfloat16)
    z = mx.random.normal((1, rows, obj.value_dim)).astype(mx.bfloat16)
    fused = prefill.qwen4_gdn_prefill_norm_gate(y, z, obj.norm.weight, obj.norm.eps,
                                                gate='swish')
    eager = obj.norm(y, z.reshape(y.shape)).reshape(1, rows, -1)
    mx.eval(fused, eager)
    same = _same(mx, fused, eager)
    out['norm_gate'] = same if same else _diff(mx, fused, eager)
    out['passed'] = all(v is True for v in out.values())
    return out


def check_model(args, mx, model_path):
    from mlx2.adapters.registry import resolve_adapter

    started = time.perf_counter()
    adapter = resolve_adapter(model_path)(
        str(model_path), execution_policy={'fused_gdn_prefill': True})
    # Only after the adapter applied its import-time profile.
    from mlx2.runtime.models import qwen38_fused_gdn as route
    load_s = time.perf_counter() - started
    model = adapter.model
    prompts = {name: list(adapter.tokenizer.encode(text))
               for name, text in _prompt_texts(ROOT).items()}

    def prefill(ids, chunk, fused):
        route.configure(model, adapter.fused_gdn, prefill=fused)
        cache = model.make_cache()
        before = route.stats(model)
        logits = None
        for start in range(0, len(ids), chunk):
            logits = model(mx.array(ids[start:start + chunk])[None], cache=cache)
            mx.eval(logits, [c.state for c in cache])
        tokens = []
        nxt = mx.argmax(logits[:, -1, :], axis=-1)
        for _ in range(args.continuation):
            tokens.append(int(nxt.item()))
            step = model(nxt[None], cache=cache)
            nxt = mx.argmax(step[:, -1, :], axis=-1)
        after = route.stats(model)
        delta = {k: after[k] - before[k] for k in ('prefill_chunk_calls', 'prefill_chunk_tokens')}
        return logits, cache, tokens, delta

    def flat_state(cache):
        out = []
        for c in cache:
            state = c.state
            for item in (state if isinstance(state, (list, tuple)) else [state]):
                if isinstance(item, (list, tuple)):
                    out.extend(item)
                else:
                    out.append(item)
        return out

    cases = {}
    for name, ids in prompts.items():
        for chunk in args.chunks:
            if name == 'short' and chunk != args.chunks[0]:
                continue
            ref_logits, ref_cache, ref_tokens, ref_delta = prefill(ids, chunk, False)
            got_logits, got_cache, got_tokens, got_delta = prefill(ids, chunk, True)
            ref_state, got_state = flat_state(ref_cache), flat_state(got_cache)
            states_equal = len(ref_state) == len(got_state) and all(
                _same(mx, a, b) for a, b in zip(ref_state, got_state))
            expected_chunks = sum(
                1 for s in range(0, len(ids), chunk) if min(chunk, len(ids) - s) >= 64)
            gdn_layers = route.stats(model)['layers']
            case = {
                'tokens': len(ids), 'chunk': chunk,
                'logits_bit_identical': _same(mx, ref_logits, got_logits),
                'cache_bit_identical': states_equal,
                'continuation_identical': ref_tokens == got_tokens,
                'continuation': got_tokens,
                'fused_prefill_chunk_calls': got_delta['prefill_chunk_calls'],
                'reference_prefill_chunk_calls': ref_delta['prefill_chunk_calls'],
                'expected_chunk_calls': expected_chunks * gdn_layers,
            }
            if not case['logits_bit_identical']:
                case['logits_diff'] = _diff(mx, ref_logits, got_logits)
            case['passed'] = (case['logits_bit_identical'] and states_equal
                              and case['continuation_identical']
                              and ref_delta['prefill_chunk_calls'] == 0
                              and got_delta['prefill_chunk_calls'] == case['expected_chunk_calls'])
            cases[f'{name}/{chunk}'] = case
            print(json.dumps({'case': f'{name}/{chunk}', 'passed': case['passed']}), flush=True)
            del ref_cache, got_cache
            mx.clear_cache()
    stats = route.stats(model)
    return {'load_s': round(load_s, 1), 'cases': cases,
            'reasons': stats['reasons'],
            'passed': bool(cases) and all(c['passed'] for c in cases.values())}


if __name__ == '__main__':
    raise SystemExit(main())
