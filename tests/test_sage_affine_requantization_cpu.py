"""Requantization format and fidelity checks, without GPU execution."""
import importlib.util
import json
import struct
from pathlib import Path

import numpy as np
import pytest

P = Path(__file__).resolve().parents[1] / "scripts/quantize_sage_a15.py"
spec = importlib.util.spec_from_file_location("sage_quant", P)
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


@pytest.mark.parametrize("bits,shape", [(8, [5, 128]), (4, [2, 3, 128])])
@pytest.mark.parametrize("dtype", ["F32", "BF16"])
def test_streamed_affine_layout_decodes_to_expected_shape_and_error(tmp_path, bits, shape, dtype):
    values = np.random.default_rng(5).normal(size=shape).astype("<f4")
    key = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"
    src, dst = tmp_path / "source.safetensors", tmp_path / "output.safetensors"
    if dtype == "BF16":
        stored_values = (values.view(np.uint32) >> 16).astype("<u2")
        values = (stored_values.astype(np.uint32) << 16).view(np.float32)
    else:
        stored_values = values
    header = {key: {"dtype": dtype, "shape": shape, "data_offsets": [0, stored_values.nbytes]}}
    raw = json.dumps(header).encode()
    src.write_bytes(struct.pack("<Q", len(raw)) + raw + stored_values.tobytes())
    name, entry, offset = q.tensor_header(src)
    out, error = q.quantize_tensor(src, dst, name, entry, offset, bits, 2)
    with dst.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        stored = json.loads(f.read(n))
    assert stored == out
    def read(name, dtype):
        e = stored[name]
        return np.memmap(dst, mode="r", dtype=dtype, offset=8 + n + e["data_offsets"][0], shape=tuple(e["shape"]))
    packed = read(key, "<u4")
    codes = np.stack([(packed >> shift) & ((1 << bits) - 1) for shift in range(0, 32, bits)], axis=-1).reshape(shape)
    base = key.removesuffix(".weight")
    scale = (read(base + ".scales", "<u2").astype(np.uint32) << 16).view(np.float32)
    bias = (read(base + ".biases", "<u2").astype(np.uint32) << 16).view(np.float32)
    recovered = codes.reshape(*shape[:-1], -1, 64) * scale[..., None] + bias[..., None]
    recovered = recovered.reshape(shape)
    assert np.mean((recovered - values)**2) < (.0001 if bits == 8 else .02)
    assert error["elements"] == values.size
    assert error["error_after"] <= error["error_before"]


@pytest.mark.parametrize("key,shape", [
    ("language_model.model.layers.0.mlp.gate.weight", [256, 2048]),
    ("language_model.model.layers.0.mlp.shared_expert_gate.weight", [1, 2048]),
    ("language_model.model.layers.0.linear_attn.conv1d.weight", [4096, 4, 1]),
    ("language_model.model.layers.0.linear_attn.norm.weight", [128]),
])
def test_sensitive_parameters_are_preserved(key, shape):
    assert q.policy(key, shape, "q8") is None
    assert q.policy(key, shape, "mixed-q4") is None


def test_mixed_policy_preserves_dense_and_mtp_precision():
    assert q.policy("language_model.mtp.fc.weight", [2048, 4096], "mixed-q4") == 8
    assert q.policy("language_model.mtp.layers.0.mlp.switch_mlp.down_proj.weight", [256, 2048, 512], "mixed-q4") == 4
    assert q.policy("language_model.model.layers.0.linear_attn.in_proj_qkv.weight", [8192, 2048], "mixed-q4") == 8


def test_parallel_shards_are_byte_exact_to_serial_and_source_hashes_are_checked(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    source, serial, parallel = [tmp_path / n for n in ("source", "serial", "parallel")]
    for directory in (source, serial, parallel):
        directory.mkdir()
    jobs = []
    for index in range(8):
        key = f"language_model.model.layers.{index}.mlp.switch_mlp.gate_proj.weight"
        values = np.random.default_rng(index).normal(size=(2, 3, 64)).astype("<f4")
        name = f"model-{index}.safetensors"
        raw = json.dumps({key: {"dtype": "F32", "shape": list(values.shape), "data_offsets": [0, values.nbytes]}}).encode()
        (source / name).write_bytes(struct.pack("<Q", len(raw)) + raw + values.tobytes())
        jobs.append((name, {"sha256": q.sha256(source / name)}))
    def run(destination, job):
        name, record = job
        return q.convert_shard(source, destination, name, record, "mixed-q4", 2)
    expected = [run(serial, job) for job in jobs]
    with ThreadPoolExecutor(max_workers=5) as pool:
        actual = list(pool.map(lambda job: run(parallel, job), jobs))
    assert actual == expected
    assert all((serial / name).read_bytes() == (parallel / name).read_bytes() for name, _ in jobs)
    with pytest.raises(ValueError, match="hash mismatch"):
        q.convert_shard(source, parallel, "model-0.safetensors", {"sha256": "wrong"}, "mixed-q4", 2)


def test_spawned_workers_write_the_same_shards_as_serial(tmp_path):
    import subprocess
    import sys

    source, serial, parallel = [tmp_path / n for n in ("source", "serial", "parallel")]
    for directory in (source, serial, parallel):
        directory.mkdir()
    key = "language_model.mtp.layers.0.mlp.switch_mlp.down_proj.weight"
    values = np.random.default_rng(4).normal(size=(2, 3, 64)).astype("<f4")
    raw = json.dumps({key: {"dtype": "F32", "shape": list(values.shape), "data_offsets": [0, values.nbytes]}}).encode()
    for index in range(4):
        name = f"model-{index}.safetensors"
        (source / name).write_bytes(struct.pack("<Q", len(raw)) + raw + values.tobytes())
        q.convert_shard(source, serial, name, {"sha256": q.sha256(source / name)}, "mixed-q4", 2)
    code = '''
import sys,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
sys.path.insert(0,sys.argv[1])
import quantize_sage_a15 as q
source,output=map(Path,sys.argv[2:])
names=sorted(x.name for x in source.iterdir())
records=[{'sha256':q.sha256(source/name)} for name in names]
with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
    list(pool.map(q.convert_shard,repeat(source),repeat(output),names,records,repeat('mixed-q4'),repeat(2)))
'''
    subprocess.run([sys.executable, "-c", code, str(P.parent), str(source), str(parallel)], check=True, timeout=30)
    assert all((serial / name.name).read_bytes() == (parallel / name.name).read_bytes() for name in source.iterdir())
