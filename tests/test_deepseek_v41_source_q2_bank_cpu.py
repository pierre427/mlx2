"""Q2 bank: converter numerics, quantized_matmul oracle and manifest round trip.

CPU only (MLX pinned to the CPU device); synthetic source tensors, no model.
"""
from collections import defaultdict
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mlx2.adapters import deepseek_v41_source_q1_on_miss as q1m
from mlx2.adapters import deepseek_v41_source_tiers as tiers

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def conv():
    import mlx.core as mx
    spec = importlib.util.spec_from_file_location('ds41_q2_convert', ROOT / 'scripts/research/ds41_expert_q2/convert.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    previous = mx.default_device(); mx.set_default_device(mx.cpu)
    yield module
    mx.set_default_device(previous)


def test_decode_mxfp4_matches_the_e2m1_table(conv):
    packed = np.array([[0x21, 0xF9] + [0] * 14], np.uint8)          # 32 values, one exponent
    exponents = np.array([[128]], np.uint8)                           # x2
    out = conv.decode_mxfp4(packed, exponents)
    assert out.shape == (1, 32)
    assert out[0, :4].tolist() == [1.0, 2.0, -1.0, -12.0]             # nibbles 1,2,9,15 -> .5,1,-.5,-6 (x2)
    with pytest.raises(ValueError):
        conv.decode_mxfp4(packed, np.array([[255]], np.uint8))


def test_quantized_matmul_matches_dequantized_oracle(conv):
    import mlx.core as mx
    rng = np.random.default_rng(3)
    source = (rng.standard_normal((64, 256)) * 0.02).astype(np.float32)
    source[3, :64] = 0.0                                              # an all-zero group stays zero
    q, s, b, err, norm = conv.quantize(source)
    assert q.shape == (64, 16) and s.shape == b.shape == (64, 4) and s.dtype == np.float16
    deq = np.array(mx.dequantize(mx.array(q), mx.array(s).astype(mx.float32), mx.array(b).astype(mx.float32),
                                 group_size=64, bits=2))
    assert np.allclose(deq[3, :64], 0.0)
    assert err == pytest.approx(float(np.sum((deq.astype(np.float64) - source) ** 2)), rel=1e-6)
    x = rng.standard_normal((3, 256)).astype(np.float32)
    got = np.array(mx.quantized_matmul(mx.array(x), mx.array(q), mx.array(s), mx.array(b),
                                       transpose=True, group_size=64, bits=2))
    assert np.allclose(got, x @ deq.T, rtol=2e-3, atol=2e-5)
    assert 0.2 < np.sqrt(err / norm) < 0.6                            # 2-bit Gaussian ballpark


def synthetic_raw(layer, expert, seed):
    rng = np.random.default_rng(seed)
    raw = {}
    for part, (n, k) in (('w1', (2304, 5120)), ('w3', (2304, 5120)), ('w2', (5120, 2304))):
        prefix = f'layers.{layer}.ffn.experts.{expert}.{part}.'
        raw[prefix + 'weight'] = rng.integers(0, 256, (n, k // 2), dtype=np.uint8)
        raw[prefix + 'scale'] = rng.integers(118, 124, (n, k // 32), dtype=np.uint8)
    return raw


def test_converter_output_round_trips_through_the_runtime_loader(conv, tmp_path, monkeypatch):
    source = 'c' * 64
    conv.ARTIFACT = {'identity': {'fingerprint': source},
                     'tensors': defaultdict(lambda: {'file': 'shard.safetensors', 'offset': 0})}
    conv.DEST = tmp_path
    monkeypatch.setattr(conv, 'read_tensors', lambda artifact, names: synthetic_raw(
        int(names[0].split('.')[1]), int(names[0].split('.')[4]), 7))
    records = [conv.convert((0, e)) for e in range(2)]
    assert all(r['payload_bytes'] == tiers.Q2_BYTES for r in records)
    assert conv.convert((0, 0)) == records[0]                          # resumable, hash-checked
    fingerprint = conv.write_manifest(tmp_path, source, 'test', records, 1.0)
    root, loaded, binding = q1m.load_q2_manifest(tmp_path, source, expected_experts=2)
    assert binding == fingerprint and set(loaded) == {(0, 0), (0, 1)}
    arrays = q1m.q2_read(root, loaded[(0, 1)])
    assert arrays['gate.q2weight'].shape == (2304, 320) and arrays['down.q2scales'].shape == (5120, 36)
    with pytest.raises(ValueError, match='not the admitted'):
        q1m.load_q2_manifest(tmp_path, 'd' * 64, expected_experts=2)
    manifest = (tmp_path / 'manifest.json').read_text()
    (tmp_path / 'manifest.json').write_text(manifest.replace('"test"', '"tampered"'))
    with pytest.raises(ValueError, match='fingerprint'):
        q1m.load_q2_manifest(tmp_path, source, expected_experts=2)
