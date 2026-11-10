"""Resumable CPU-only Q2 conversion of all routed Source experts; preserves originals.

MLX native affine 2-bit, group 64, FP16 scales/biases (the layout
``mx.quantized_matmul(..., group_size=64, bits=2)`` consumes). The original
E2M1/E8M0 group-32 codes are decoded to FP32 on the host and quantized in
FP32 with ``mx.quantize`` pinned to the CPU device; only the resulting
scales/biases are cast to FP16 (v1 cast the weights to FP16 first, which
flushed groups of tiny magnitude). MLX never touches the GPU here.
Modelled on scripts/research/ds41_expert_q1/convert.py (same receipt/manifest
schema shape, per-expert relative reconstruction error, resumable sealing).
"""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[3] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse, hashlib, json, os, time
import numpy as np
from safetensors.numpy import save_file
from mlx2.adapters.deepseek_v41_source_artifact import inspect_artifact, read_tensors
from mlx2.adapters.deepseek_v41_source_weights import expert_names, PROJECTIONS

REVISION = 'source-q2-mlx-affine-g64-f32q-f16-v2'
GROUP, BITS = 64, 2
LUT = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
ARTIFACT = DEST = None


def initialize(model, destination):
    global ARTIFACT, DEST
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    ARTIFACT = inspect_artifact(model); DEST = Path(destination)


def decode_mxfp4(packed, exponents):
    """E2M1 nibbles (low nibble first) x 2^(e-127) per 32 values -> float32 (n, k)."""
    if np.any(exponents == 255):
        raise ValueError('nonfinite source exponent')
    n = packed.shape[0]
    codes = np.empty((n, packed.shape[1] * 2), np.uint8)
    codes[:, 0::2] = packed & 15
    codes[:, 1::2] = packed >> 4
    values = LUT[codes].reshape(n, -1, 32)
    scale = np.ldexp(np.float32(1), exponents.astype(np.int32) - 127).astype(np.float32)
    return (values * scale[:, :, None]).reshape(n, -1)


def quantize(source):
    import mlx.core as mx
    with mx.stream(mx.cpu):
        w = mx.array(source.astype(np.float32))
        q, s, b = mx.quantize(w, group_size=GROUP, bits=BITS)
        s, b = s.astype(mx.float16), b.astype(mx.float16)
        # Error of the stored (FP16-parameter) object, dequantized in FP32.
        d = mx.dequantize(q, s.astype(mx.float32), b.astype(mx.float32), group_size=GROUP, bits=BITS)
        mx.eval(q, s, b, d)
    q, s, b, d = np.array(q), np.array(s), np.array(b), np.array(d)
    if not (np.isfinite(s).all() and np.isfinite(b).all()):
        raise ValueError('nonfinite Q2 parameter')
    err = float(np.sum((d.astype(np.float64) - source) ** 2)); norm = float(np.sum(source.astype(np.float64) ** 2))
    return q.astype('<u4'), s.astype('<f2'), b.astype('<f2'), err, norm


def convert(item):
    layer, expert = item; start = time.perf_counter()
    name = f'l{layer:02d}-e{expert:03d}.safetensors'; target = DEST / name; receipt = target.with_suffix('.json')
    if target.exists() and receipt.exists():
        record = json.loads(receipt.read_text())
        if record['source'] != ARTIFACT['identity']['fingerprint'] or record['quantization'] != REVISION:
            raise ValueError('existing incompatible expert')
        if hashlib.sha256(target.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('existing expert hash changed')
        return record
    if target.exists() or receipt.exists():
        raise ValueError('partial expert publication: ' + name)
    raw = read_tensors(ARTIFACT, expert_names(layer, expert)); arrays = {}; errors = {}; original = []
    for kind, part in PROJECTIONS.items():
        prefix = f'layers.{layer}.ffn.experts.{expert}.{part}.'
        source = decode_mxfp4(raw[prefix + 'weight'], raw[prefix + 'scale'])
        q, s, b, err, norm = quantize(source)
        arrays.update({kind + '.qweight': q, kind + '.scales': s, kind + '.biases': b})
        errors[kind] = {'squared_error': err, 'source_squared_norm': norm,
                        'relative_rmse': float(np.sqrt(err / norm)) if norm else 0}
        original.extend([{**ARTIFACT['tensors'][prefix + field], 'tensor': prefix + field} for field in ('weight', 'scale')])
    temp = target.with_suffix('.tmp')
    save_file(arrays, str(temp), metadata={'source': ARTIFACT['identity']['fingerprint'], 'quantization': REVISION,
                                           'layer': str(layer), 'expert': str(expert)})
    h = hashlib.sha256(temp.read_bytes()).hexdigest(); os.replace(temp, target)
    record = {'layer': layer, 'expert': expert, 'file': name, 'source': ARTIFACT['identity']['fingerprint'],
              'quantization': REVISION, 'sha256': h, 'size': target.stat().st_size,
              'payload_bytes': sum(v.nbytes for v in arrays.values()), 'original_extents': original,
              'errors': errors, 'seconds': time.perf_counter() - start}
    temporary = receipt.with_suffix('.tmp'); temporary.write_text(json.dumps(record) + '\n'); os.replace(temporary, receipt)
    return record


def write_manifest(destination, source, source_revision, records, seconds):
    """Seal the bank: fingerprint = sha256 of the canonical manifest without it."""
    manifest = {'schema': 'mlx2.source-expert-q2.v1', 'source': source,
                'source_revision': source_revision, 'quantization': REVISION, 'group_size': GROUP,
                'weight_bits': BITS, 'effective_bits_with_parameters': 2.5, 'dense_modified': False,
                'original_objects': 'immutable six-tensor extents in original Source shards',
                'experts': records, 'payload_bytes': sum(r['payload_bytes'] for r in records),
                'seconds': seconds}
    manifest['fingerprint'] = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    temp = Path(destination) / 'manifest.tmp'; temp.write_text(json.dumps(manifest) + '\n')
    os.replace(temp, Path(destination) / 'manifest.json')
    return manifest['fingerprint']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--model', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=6); parser.add_argument('--limit', type=int, default=15360)
    parser.add_argument('--source-revision', required=True)
    args = parser.parse_args()
    destination = Path(args.output); destination.mkdir(parents=True, exist_ok=True); artifact = inspect_artifact(args.model)
    items = [(l, e) for l in range(40) for e in range(384)][:args.limit]; records = []; start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(args.model, args.output)) as pool:
        for record in pool.map(convert, items, chunksize=1):
            records.append(record)
            if len(records) % 128 == 0 or len(records) == len(items):
                print(json.dumps({'completed': len(records), 'total': len(items), 'seconds': time.perf_counter() - start,
                                  'payload_bytes': sum(r['payload_bytes'] for r in records)}), flush=True)
    if len(records) == 15360:
        write_manifest(destination, artifact['identity']['fingerprint'], args.source_revision, records,
                       time.perf_counter() - start)
