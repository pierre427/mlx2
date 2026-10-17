import importlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from mlx2.conversion import kernel_probe as preparation
from mlx2.conversion.nvfp4 import sha256


def write_source(path, *, scale_shape=None):
    scale_shape = [1] if scale_shape is None else scale_shape
    arrays = {"fc.weight": ("U8", np.full((2, 32), 0x32, np.uint8)),
              "fc.weight_scale": ("F8_E4M3", np.full((2, 4), 32, np.uint8)),
              "fc.weight_scale_2": ("F32", np.array(.0625, np.float32).reshape(scale_shape))}
    header, payload = {}, b""
    for name, (dtype, array) in arrays.items():
        block = array.tobytes()
        header[name] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [len(payload), len(payload) + len(block)]}
        payload += block
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


@pytest.mark.parametrize("scalar_shape", [[1], []])
def test_actual_source_triple_decodes_independently_and_refuses_wrong_geometry(tmp_path, scalar_shape):
    path = tmp_path / "model.safetensors"
    write_source(path, scale_shape=scalar_shape)
    arrays = preparation.read_matrix(path, "fc", expected_shape=(2, 64))
    np.testing.assert_array_equal(arrays["dense_fp32"], np.tile(np.array([1., 1.5], np.float32) * .125 * .0625, (2, 32)))
    assert arrays["native_weight"].dtype == np.uint32
    with pytest.raises(ValueError, match="geometry"):
        preparation.read_matrix(path, "fc", expected_shape=(4, 64))
    with pytest.raises(ValueError, match="bound"):
        preparation.read_matrix(path, "fc", expected_shape=(2, 0))


def test_preparation_binds_real_payload_and_preserves_exclusive_output(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    write_source(source / "model.safetensors")
    (source / "config.json").write_text("{}")
    monkeypatch.setattr(preparation, "SOURCE_WEIGHT_SHA256", sha256(source / "model.safetensors"))
    monkeypatch.setattr(preparation, "SOURCE_CONFIG_SHA256", sha256(source / "config.json"))
    monkeypatch.setattr(preparation, "MATRICES", {"fc": (2, 64)})
    destination = tmp_path / "prepared"
    manifest = preparation.prepare(source, destination, reserve_bytes=0)
    assert preparation.verify(destination) == manifest
    with np.load(destination / "matrix-0.npz", allow_pickle=False) as archive:
        for bits in (4, 8):
            packed = archive[f"affine{bits}_weight"]
            # Independent bit extraction and stored-BF16 decode oracle.
            codes = np.stack([(packed >> shift) & ((1 << bits) - 1) for shift in range(0, 32, bits)], axis=-1).reshape(2, 64)
            scales = (archive[f"affine{bits}_scales"].astype(np.uint32) << 16).view(np.float32)
            biases = (archive[f"affine{bits}_biases"].astype(np.uint32) << 16).view(np.float32)
            decoded = codes * scales + biases
            error = np.mean((decoded - archive["dense_fp32"])**2)
            assert error == pytest.approx(manifest["matrices"][0]["fitting"][str(bits)]["weight_mse"])
    with pytest.raises(FileExistsError):
        preparation.prepare(source, destination, reserve_bytes=0)
    payload = destination / "matrix-0.npz"
    payload.write_bytes(payload.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="payload"):
        preparation.verify(destination)


def test_correctness_metrics_reject_nonfinite_and_different_shapes():
    assert preparation.error_metrics([1., 2.], [1., 2.])["relative_rmse"] == 0.
    assert preparation.error_metrics([2., 4.], [1., 2.])["relative_rmse"] == 1.
    with pytest.raises(ValueError):
        preparation.error_metrics([float("inf")], [1.])
    with pytest.raises(ValueError):
        preparation.error_metrics([1.], [1., 2.])


def test_kernel_child_requires_same_live_owner_and_stopped_server(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    probe = importlib.import_module("probe_nemotron35_nvfp4_kernels")
    owner = {"cpg_owner": {"task": "task"}, "paired_gpu_owners": {"shared": {"pair": "pair"}}}
    receipt = {"schema": "mlx2.nemotron35-kernel-server-unit.v1", "source_sha256": "source", "kernel_inputs": "inputs",
               "ordinary_reference_passed": True, "ordinary_server_passed": True, "server_group_stopped": True,
               "max_lanes": 1, "max_inflight": 1, "server_pgid": 4321, "dependencies": {"mlx": "fixture"},
               "cpg_owner": owner["cpg_owner"], "paired_owner": owner["paired_gpu_owners"]["shared"]}
    monkeypatch.setattr(probe.common, "runtime_dependencies", lambda: {"mlx": "fixture"})
    monkeypatch.setattr(probe.common, "group_running", lambda _: [])
    probe.validate_unit(receipt, owner=owner, source_sha="source", inputs="inputs")
    for key, value in (("kernel_inputs", "different"), ("max_lanes", 4), ("source_sha256", "different"),
                       ("dependencies", {}), ("ordinary_reference_passed", False), ("server_group_stopped", False)):
        with pytest.raises(ValueError):
            probe.validate_unit({**receipt, key: value}, owner=owner, source_sha="source", inputs="inputs")
    monkeypatch.setattr(probe.common, "group_running", lambda _: [4321])
    with pytest.raises(ValueError, match="live"):
        probe.validate_unit(receipt, owner=owner, source_sha="source", inputs="inputs")
