"""Private CPU-only shadow trainer; input closure is hash checked before use."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import zipfile
from itertools import pairwise
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_initial_head(path, config):
    """Close config geometry and every payload byte before head allocation."""
    from ..adapters.dflash2 import _DTYPE_BYTES, _decode_unique_json
    from ..adapters.lilicorr import expected_weight_shapes

    size = path.stat().st_size
    with path.open("rb") as weights:
        length = weights.read(8)
        if len(length) != 8:
            raise ValueError("shadow worker truncated weight header")
        header_size = struct.unpack("<Q", length)[0]
        if not 0 < header_size <= min(4 << 20, size - 8):
            raise ValueError("shadow worker invalid weight header length")
        header = _decode_unique_json(weights.read(header_size), "shadow initial head")
    if not isinstance(header, dict):
        raise ValueError("shadow worker invalid weight header")  # noqa: TRY004 - malformed file schema
    # Geometry expansion must itself stay bounded by the supplied header. Each
    # layer contributes tensor records; impossible layer counts fail before
    # expected_weight_shapes constructs its per-layer dictionaries.
    if max(config.num_hidden_layers, config.lilicorr_num_layers) > len(header):
        raise ValueError("shadow worker head schema/config mismatch")
    expected = {
        name.removeprefix("lilicorr."): shape
        for name, shape in expected_weight_shapes(config).items()
        if name.startswith("lilicorr.")
    }
    payload_size = size - 8 - header_size
    observed, ranges = {}, []
    for name, item in header.items():
        if name == "__metadata__":
            if item is not None and (
                not isinstance(item, dict)
                or any(
                    not isinstance(key, str) or not isinstance(value, str)
                    for key, value in item.items()
                )
            ):
                raise ValueError("shadow worker invalid weight metadata")
            continue
        if (
            name not in expected
            or not isinstance(item, dict)
            or set(item) != {"dtype", "shape", "data_offsets"}
        ):
            raise ValueError("shadow worker head schema/config mismatch")
        shape, offsets, dtype = item["shape"], item["data_offsets"], item["dtype"]
        if (
            not isinstance(dtype, str)
            or dtype not in _DTYPE_BYTES
            or not isinstance(shape, list)
            or any(type(number) is not int for number in shape)
            or shape != expected[name]
        ):
            raise ValueError("shadow worker head dtype/shape/config mismatch")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(number) is not int for number in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= payload_size
            or offsets[1] - offsets[0] != math.prod(shape) * _DTYPE_BYTES[dtype]
        ):
            raise ValueError("shadow worker invalid head payload offsets")
        observed[name] = shape
        ranges.append(tuple(offsets))
    if observed != expected:
        raise ValueError("shadow worker head schema/config mismatch")
    ranges.sort()
    if (
        not ranges
        or ranges[0][0] != 0
        or ranges[-1][1] != payload_size
        or any(previous[1] != following[0] for previous, following in pairwise(ranges))
    ):
        raise ValueError("shadow worker head payload has overlap, holes or tail bytes")
    return sum(math.prod(shape) for shape in expected.values())


def run(path):
    import mlx.core as mx
    import numpy as np

    from ..adapters.lilicorr import LiLiCorrConfig
    from .drafters.lilicorr import LiLiCorrHead
    from .lilicorr_feedback import (
        LiLiCorrFeedbackPolicy,
        estimate_training_bytes,
        training_budget_available,
    )
    from .lilicorr_training import TeacherBuffer, export_shadow, train_shadow

    mx.set_default_device(mx.cpu)
    job = json.loads(path.read_text())
    directory = path.parent
    for name in ("initial.safetensors", "teachers.npz"):
        if file_sha256(directory / name) != job["files_sha256"][name]:
            raise ValueError("shadow worker input hash mismatch")
    policy = LiLiCorrFeedbackPolicy.from_value(job["policy"])
    config = LiLiCorrConfig(**job["config"])
    parameters = _inspect_initial_head(directory / "initial.safetensors", config)
    with zipfile.ZipFile(directory / "teachers.npz") as teachers:
        teacher_bytes = sum(item.file_size for item in teachers.infolist())
    if len(
        job["examples"]
    ) > policy.max_examples or teacher_bytes > policy.max_bytes + (128 << 10):
        raise ValueError("shadow worker teacher input exceeds budget")
    estimated = estimate_training_bytes(
        parameters, teacher_bytes, len(job["examples"]), config
    )
    if not training_budget_available(estimated, policy.max_training_bytes):
        raise ValueError(
            "shadow worker estimated memory exceeds available training budget"
        )
    head = LiLiCorrHead(config)
    head.load_weights(
        list(mx.load(str(directory / "initial.safetensors")).items()), strict=True
    )
    buffer = TeacherBuffer(
        config,
        target_revision=job["target_revision"],
        draft_revision=job["draft_revision"],
        max_examples=policy.max_examples,
        max_bytes=policy.max_bytes,
    )
    with np.load(directory / "teachers.npz", allow_pickle=False) as tensors:
        for index, example in enumerate(job["examples"]):
            fields = {
                field: tensors[f"{index}.{field}"]
                for field in (
                    "candidate_ids",
                    "token_embeddings",
                    "candidate_log_probs",
                    "pass_hidden",
                    "anchor_hidden",
                )
            }
            buffer.add(**fields, **example)
    result = train_shadow(
        head, buffer, steps=policy.steps, learning_rate=policy.learning_rate
    )
    manifest = export_shadow(result, directory / "head")
    return {
        "passed": True,
        "binding": job["binding"],
        "device": "CPU",
        "estimated_training_bytes": estimated,
        "target_revision": job["target_revision"],
        "draft_revision": job["draft_revision"],
        "head_directory": str(directory / "head"),
        "manifest": manifest,
        "initial_loss": result.initial_loss,
        "final_loss": result.final_loss,
        "qualified": False,
        "selected": False,
        "live_head_changed": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = run(args.job.resolve())
    except Exception as error:  # noqa: BLE001 - persist training failure without changing inference
        report = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    (args.job.parent / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
