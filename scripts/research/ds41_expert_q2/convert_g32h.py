#!/usr/bin/env python3
"""Q2 bank v3 for DS4.1 Source routed experts: affine 2-bit g32, block-Hadamard rotation, MSE clip.

Per projection W (n, k): W' = W H (orthonormal Hadamard over 256-wide blocks of the input dim k);
each 32-wide group of W' gets the 2-bit affine (scale, bias) whose range shrink c in CLIP minimises
the group's squared error; codes are packed in MLX's 2-bit layout so the runtime computes
  y = quantized_matmul(H x, q, scales, biases, group_size=32, bits=2)
which equals W' (H x) = W x up to quantization. 3.0 bits/weight with f16 parameters
(13,271,040 B/expert). Prototype vs current g64 bank (6 experts, captured inputs): expert output
error 0.40-0.53 vs 0.50-0.70. CPU only (MLX pinned to CPU in every worker).
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse, hashlib, json, os, time
import numpy as np
from safetensors.numpy import save_file
from mlx2.adapters.deepseek_v41_source_artifact import inspect_artifact, read_tensors
from mlx2.adapters.deepseek_v41_source_weights import expert_names, PROJECTIONS
from convert import decode_mxfp4

REVISION = 'source-q2-mlx-affine-g32-had256-mseclip-f16-v3'
GROUP, BITS, BLOCK = 32, 2, 256
CLIP = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65)
ARTIFACT = DEST = None


def initialize(model, destination):
    global ARTIFACT, DEST
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    ARTIFACT = inspect_artifact(model); DEST = Path(destination)


def rotate(w):
    import mlx.core as mx
    n, k = w.shape
    with mx.stream(mx.cpu):
        r = mx.hadamard_transform(mx.array(w).reshape(n, k // BLOCK, BLOCK)).reshape(n, k)
        mx.eval(r)
    return np.array(r, np.float32)


def quantize(w):
    """MSE-clipped affine 2-bit per 32-group -> (packed u32 (n,k/16), scales f16, biases f16, dequant f32)."""
    n, k = w.shape
    g = w.reshape(n, k // GROUP, GROUP)
    lo, hi = g.min(-1, keepdims=True), g.max(-1, keepdims=True)
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    best = None
    for c in CLIP:
        # floor at the smallest normal f16 so constant groups never divide by a zero scale
        s = np.maximum((2 * half * c) / 3, 6.103515625e-05).astype(np.float16).astype(np.float32)
        b = (mid - half * c).astype(np.float16).astype(np.float32)
        q = np.clip(np.rint((g - b) / s), 0, 3)
        e = (((q * s + b) - g) ** 2).sum(-1, keepdims=True)
        if best is None:
            best = [e, q, s, b]
        else:
            m = e < best[0]
            best = [np.where(m, e, best[0]), np.where(m, q, best[1]), np.where(m, s, best[2]), np.where(m, b, best[3])]
    _, q, s, b = best
    codes = q.reshape(n, k).astype(np.uint32).reshape(n, k // 16, 16)
    packed = np.zeros((n, k // 16), np.uint32)
    for j in range(16):
        packed |= codes[:, :, j] << np.uint32(2 * j)
    deq = (q * s + b).reshape(n, k)
    return packed, s[..., 0].astype(np.float16), b[..., 0].astype(np.float16), deq


def check_packing(packed, s, b, deq):
    import mlx.core as mx
    with mx.stream(mx.cpu):
        d = mx.dequantize(mx.array(packed), mx.array(s).astype(mx.float32), mx.array(b).astype(mx.float32), group_size=GROUP, bits=BITS)
        mx.eval(d)
    if not np.allclose(np.array(d, np.float32), deq, rtol=0, atol=1e-6 * max(1.0, float(np.abs(deq).max()))):
        raise ValueError('2-bit packing does not match MLX dequantize')


def convert(item):
    layer, expert = item; start = time.perf_counter()
    name = f'l{layer:02d}-e{expert:03d}.safetensors'; target = DEST / name; receipt = target.with_suffix('.json')
    if target.exists() and receipt.exists():
        record = json.loads(receipt.read_text())
        if record['source'] != ARTIFACT['identity']['fingerprint'] or record['quantization'] != REVISION:
            raise ValueError('existing incompatible expert')
        return record
    if target.exists() or receipt.exists():
        raise ValueError('partial expert publication: ' + name)
    raw = read_tensors(ARTIFACT, expert_names(layer, expert)); arrays = {}; errors = {}; original = []
    for kind, part in PROJECTIONS.items():
        prefix = f'layers.{layer}.ffn.experts.{expert}.{part}.'
        source = decode_mxfp4(raw[prefix + 'weight'], raw[prefix + 'scale'])
        rotated = rotate(source)
        q, s, b, deq = quantize(rotated)
        if layer == 0 and expert < 2:
            check_packing(q, s, b, deq)
        err = float(np.sum((deq.astype(np.float64) - rotated) ** 2)); norm = float(np.sum(rotated.astype(np.float64) ** 2))
        arrays.update({kind + '.qweight': q.astype('<u4'), kind + '.scales': s.astype('<f2'), kind + '.biases': b.astype('<f2')})
        errors[kind] = {'squared_error': err, 'source_squared_norm': norm, 'relative_rmse': float(np.sqrt(err / norm)) if norm else 0}
        original.extend([{**ARTIFACT['tensors'][prefix + f], 'tensor': prefix + f} for f in ('weight', 'scale')])
    temp = target.with_suffix('.tmp')
    save_file(arrays, str(temp), metadata={'source': ARTIFACT['identity']['fingerprint'], 'quantization': REVISION,
                                           'layer': str(layer), 'expert': str(expert), 'rotation': f'block_hadamard_{BLOCK}'})
    h = hashlib.sha256(temp.read_bytes()).hexdigest(); os.replace(temp, target)
    record = {'layer': layer, 'expert': expert, 'file': name, 'source': ARTIFACT['identity']['fingerprint'],
              'quantization': REVISION, 'sha256': h, 'size': target.stat().st_size,
              'payload_bytes': sum(v.nbytes for v in arrays.values()), 'original_extents': original,
              'errors': errors, 'seconds': time.perf_counter() - start}
    tmp = receipt.with_suffix('.tmp'); tmp.write_text(json.dumps(record) + '\n'); os.replace(tmp, receipt)
    return record


def write_manifest(destination, source, source_revision, records, seconds):
    manifest = {'schema': 'mlx2.source-expert-q2.v1', 'source': source, 'source_revision': source_revision,
                'quantization': REVISION, 'group_size': GROUP, 'weight_bits': BITS, 'rotation': f'block_hadamard_{BLOCK}',
                'effective_bits_with_parameters': 3.0, 'dense_modified': False,
                'original_objects': 'immutable six-tensor extents in original Source shards',
                'experts': records, 'payload_bytes': sum(r['payload_bytes'] for r in records), 'seconds': seconds}
    manifest['fingerprint'] = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    temp = Path(destination) / 'manifest.tmp'; temp.write_text(json.dumps(manifest) + '\n')
    os.replace(temp, Path(destination) / 'manifest.json')
    return manifest['fingerprint']


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--model', required=True); p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=6); p.add_argument('--limit', type=int, default=15360)
    p.add_argument('--source-revision', required=True)
    a = p.parse_args()
    dest = Path(a.output); dest.mkdir(parents=True, exist_ok=True); art = inspect_artifact(a.model)
    items = [(l, e) for l in range(40) for e in range(384)][:a.limit]; records = []; start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=initialize, initargs=(a.model, a.output)) as pool:
        for record in pool.map(convert, items, chunksize=1):
            records.append(record)
            if len(records) % 256 == 0 or len(records) == len(items):
                print(json.dumps({'completed': len(records), 'total': len(items), 'seconds': round(time.perf_counter() - start, 1),
                                  'mean_rel_rmse': float(np.mean([np.mean([v['relative_rmse'] for v in r['errors'].values()]) for r in records]))}), flush=True)
    if len(records) == 15360:
        print('fingerprint', write_manifest(dest, art['identity']['fingerprint'], a.source_revision, records, time.perf_counter() - start))
