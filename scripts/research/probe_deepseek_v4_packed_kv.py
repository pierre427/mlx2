#!/usr/bin/env python3
"""Bounded physical KV fixture probe; shares packed-weight ownership/source guards."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from probe_deepseek_v4_packed import atomic, compare, measure, source_receipt, verify_ownership
from mlx2.runtime.models.deepseek_v4_packed_kv import pack_fp8_kv, unpack_fp8_kv, LAYOUT_REVISION


def fixture(rows):
    rng = np.random.default_rng(20261024 + rows)
    x = rng.normal(size=(1, rows, 512)).astype(np.float32)
    x *= np.geomspace(1e-6, 1e4, rows).astype(np.float32)[None, :, None]
    x[:, 0, :448] = 0
    if rows > 1:
        codes = np.arange(127)
        e, m = codes // 8, codes % 8
        lut = np.where(e == 0, m / 512, (1 + m / 8) * 2. ** (e - 7)).astype(np.float32)
        x[:, 1, :448] = 448
        x[:, 1, :126] = (lut[:-1] + lut[1:]) / 2
        x[:, 1, 126:252] = -(lut[:-1] + lut[1:]) / 2
        x[:, 1, 63:448:64] = 448
    return x


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run-gpu', action='store_true')
    p.add_argument('--ownership-window', type=Path)
    p.add_argument('--owner-path', type=Path, action='append')
    p.add_argument('--repetitions', type=int, default=3)
    args = p.parse_args(argv)
    if args.output.exists(): p.error('refusing existing evidence directory')
    if not 1 <= args.repetitions <= 20: p.error('repetitions must be1..20')
    owners = args.owner_path or [Path('/Users/Shared/mlxuag/gpu.lock/owner.json'), Path('/tmp/gpu.lock/owner.json')]
    if args.run_gpu and (args.ownership_window is None or len(owners) != 2):
        p.error('GPU requires ownership-window and both paired owner receipts')
    def guard():
        return verify_ownership(args.ownership_window, owners) if args.run_gpu else None
    ownership = guard()
    source = source_receipt()
    from mlx2.runtime.models import deepseek_v4_packed_kv as candidate
    source['packed_kv_module_sha256'] = hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest()
    source['packed_kv_probe_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    plans = [1, 128, 513, 4096]
    identity = hashlib.sha256(json.dumps(plans).encode()).hexdigest()
    result = dict(schema='mlx2.deepseek-v4-packed-kv-probe.v1', status='running',
                  checkpoint='synthetic-packed-kv-fixtures@sha256:' + identity,
                  source=source, ownership=ownership, run_gpu=args.run_gpu,
                  qualification=False, performance_claim=False, artifact_loaded=False,
                  layout=LAYOUT_REVISION, cases=[])
    args.output.mkdir(parents=True)
    atomic(args.output / 'results.json', result)
    try:
        if args.run_gpu:
            import mlx.core as mx
            from mlx2.runtime.models.deepseek_v4 import simulate_fp8_kv
            mx.set_default_device(mx.gpu)
            mx.set_memory_limit(512 << 20)
            mx.set_cache_limit(32 << 20)
            result['device_info'] = mx.metal.device_info()
        for rows in plans:
            guard()
            x = fixture(rows)
            cpu = pack_fp8_kv(x, xp=np)
            expected = unpack_fp8_kv(cpu, xp=np)
            case = dict(rows=rows, original_bytes=x.nbytes, packed_bytes=cpu.nbytes,
                        finite=bool(np.isfinite(expected).all()), passed=True)
            if args.run_gpu:
                a = mx.array(x)
                mx.eval(a)
                packed = pack_fp8_kv(a, xp=mx)
                mx.eval(packed.codes, packed.scales, packed.rope)
                case['codes_exact'] = bool(np.array_equal(np.array(packed.codes), cpu.codes))
                case['scales_exact'] = bool(np.array_equal(np.array(packed.scales), cpu.scales))
                case['physical_packed_bytes'] = packed.nbytes
                actual, timing = measure(lambda: unpack_fp8_kv(pack_fp8_kv(a, xp=mx), xp=mx), args.repetitions)
                qat = simulate_fp8_kv(a); mx.eval(qat)
                case['cpu_oracle'] = compare(actual, expected, atol=0, rtol=0)
                case['forward_qat'] = compare(actual, np.array(qat), atol=0, rtol=0)
                case['rope_bitwise'] = bool(np.array_equal(actual[...,448:].view(np.uint32), x[...,448:].view(np.uint32)))
                case['pack_unpack_timing'] = timing
                case['passed'] = all((case['codes_exact'], case['scales_exact'], case['cpu_oracle']['passed'],
                                      case['forward_qat']['passed'], case['rope_bitwise']))
            guard()
            result['cases'].append(case)
            atomic(args.output / 'results.json', result)
        result['status'] = 'completed'
        result['all_checks_passed'] = all(c['passed'] and c['finite'] for c in result['cases'])
    except BaseException as error:
        result.update(status='failed', error={'type':type(error).__name__, 'message':str(error)})
        raise
    finally:
        atomic(args.output / 'results.json', result)
    return 0 if result['all_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
