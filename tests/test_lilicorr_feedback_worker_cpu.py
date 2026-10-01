"""CPU admission checks run before allocating any shadow-head parameters."""

import json
import struct
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from test_lilicorr_feedback_cpu import capture, manager

from mlx2.runtime.lilicorr_feedback_worker import (
    _inspect_initial_head,
    file_sha256,
    run,
)

mx.set_default_device(mx.cpu)


@pytest.fixture
def pending_job(tmp_path):
    value = manager(tmp_path)
    value.submit_verified(capture(value), [1])
    child = MagicMock()
    child.poll.return_value = None
    with patch("mlx2.runtime.lilicorr_feedback.subprocess.Popen", return_value=child):
        value.settle_round()
    assert value.child_directory is not None, value.receipt()
    yield value, value.child_directory / "job.json"
    value.close()


def rewrite_header(path, mutate, tail=b""):
    original = path.read_bytes()
    length = struct.unpack("<Q", original[:8])[0]
    header = json.loads(original[8 : 8 + length])
    mutate(header)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + original[8 + length :] + tail
    )


def invoke_without_allocation(job_path, match):
    with (
        patch(
            "mlx2.runtime.drafters.lilicorr.LiLiCorrHead",
            side_effect=AssertionError("allocated head"),
        ) as head,
        patch.object(
            mx, "load", side_effect=AssertionError("loaded tensor bytes")
        ) as load,
    ):
        with pytest.raises(ValueError, match=match):
            run(job_path)
        head.assert_not_called()
        load.assert_not_called()


def test_geometry_closure_parameter_count_matches_resident_head(pending_job):
    value, path = pending_job
    count = _inspect_initial_head(
        path.parent / "initial.safetensors", value.buffer.config
    )
    assert count == sum(
        tensor.size for _, tensor in tree_flatten(value.drafter.lilicorr.parameters())
    )


@pytest.mark.parametrize(
    "field", ["lilicorr_factor_dim", "lilicorr_hidden_size", "lilicorr_num_layers"]
)
def test_changed_config_is_rejected_before_head_allocation(pending_job, field):
    _value, path = pending_job
    job = json.loads(path.read_text())
    job["config"][field] = 100_000_000
    path.write_text(json.dumps(job))
    invoke_without_allocation(path, "config mismatch")


def test_memory_budget_is_rechecked_before_head_or_tensor_allocation(pending_job):
    _value, path = pending_job
    job = json.loads(path.read_text())
    job["policy"]["max_training_bytes"] = 1
    path.write_text(json.dumps(job))
    invoke_without_allocation(path, "memory exceeds")


def test_available_memory_budget_is_rechecked_before_allocation(pending_job):
    _value, path = pending_job
    with patch(
        "mlx2.runtime.lilicorr_feedback.training_budget_available", return_value=False
    ) as budget:
        invoke_without_allocation(path, "memory exceeds")
        assert budget.call_args.args[0] > 0


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "dtype", "shape", "offset", "overlap", "tail"]
)
def test_header_and_payload_geometry_rejects_rehashed_invalid_input(
    pending_job, mutation
):
    _value, path = pending_job
    weights = path.parent / "initial.safetensors"

    def mutate(header):
        names = [name for name in header if name != "__metadata__"]
        first = names[0]
        if mutation == "missing":
            del header[first]
        elif mutation == "extra":
            header["unrelated.weight"] = header[first].copy()
        elif mutation == "dtype":
            header[first]["dtype"] = "I32"
        elif mutation == "shape":
            header[first]["shape"] = [1]
        elif mutation == "offset":
            header[first]["data_offsets"][0] = -1
        elif mutation == "overlap":
            # Keep this tensor's exact byte size while making its range collide.
            largest = max(names, key=lambda name: header[name]["data_offsets"][1])
            item = header[largest]
            length = item["data_offsets"][1] - item["data_offsets"][0]
            item["data_offsets"] = [0, length]

    rewrite_header(weights, mutate, b"unaccounted" if mutation == "tail" else b"")
    job = json.loads(path.read_text())
    job["files_sha256"]["initial.safetensors"] = file_sha256(weights)
    path.write_text(json.dumps(job))
    invoke_without_allocation(path, "schema|dtype|offset|payload")


def test_unhashed_weight_change_rejected_before_header_or_head_allocation(pending_job):
    _value, path = pending_job
    weights = path.parent / "initial.safetensors"
    with weights.open("ab") as stream:
        stream.write(b"modified")
    invoke_without_allocation(path, "hash mismatch")
