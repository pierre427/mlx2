#!/usr/bin/env python3
"""Collect one bounded offline calibration unit after its owned server smoke.

The outer runner supplies an ordinary-reference/server receipt from the SAME
live CPG and paired-lock window. Capture occurs in this separate process after
that server is stopped. Never run this in a serving or performance process.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import nemotron35_probe_common as common


def release_sample_memory(mx):
    """A dedicated collector may release temporary graphs and free buffers.

    This is outside timed serving paths. Resident model arrays remain live;
    only unreachable objects and the allocator's free cache are released.
    """
    mx.synchronize()
    gc.collect()
    mx.clear_cache()
    return {"active_bytes": mx.get_active_memory(), "cache_bytes": mx.get_cache_memory(),
            "peak_active_bytes": mx.get_peak_memory()}


def validate_unit_receipt(receipt, *, owner, artifact, source_sha):
    if (receipt.get("schema") != "mlx2.nemotron35-calibration-server-unit.v1"
            or receipt.get("artifact") != artifact or receipt.get("source_sha256") != source_sha
            or receipt.get("ordinary_reference_passed") is not True
            or receipt.get("ordinary_server_passed") is not True
            or receipt.get("server_group_stopped") is not True
            or receipt.get("max_lanes") != 1 or receipt.get("max_inflight") != 1
            or receipt.get("cpg_owner") != owner["cpg_owner"]
            or receipt.get("paired_owner") != owner["paired_gpu_owners"]["shared"]):
        raise ValueError("Calibration unit lacks matching ordinary/server/ownership/source evidence")
    if not receipt.get("dependencies") or receipt["dependencies"] != common.runtime_dependencies():
        raise ValueError("Calibration dependency/native-build identity changed since the references")
    if type(receipt.get("server_pgid")) is not int or receipt["server_pgid"] < 2:
        raise ValueError("Calibration unit has no owned server process-group identity")
    if common.group_running(receipt["server_pgid"]):
        raise ValueError("Calibration cannot load while the unit's server process group is live")


def other_waiters(snapshot, owner):
    own_pid = owner["paired_gpu_owners"]["shared"]["pid"]
    waiting, now = [], time.time()
    for name in snapshot["waiter_files"]:
        try:
            fields = name.rsplit(".", 3)
            pid, created = int(fields[-2]), int(fields[-1])
            if pid == own_pid or now - created > 5400:
                continue
            path = Path("/Users/Shared/mlxuag/gpu.lock.waiters") / name
            if now - path.stat().st_mtime > 300:
                continue
            os.kill(pid, 0)
        except (ValueError, IndexError, FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            pass
        waiting.append(name)
    return waiting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--unit-receipt", type=Path, required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--max-seconds", type=int, default=300)
    parser.add_argument("--mtp-packets", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.count <= 64 or not 30 <= args.max_seconds <= 480 or not 0 <= args.mtp_packets <= 8:
        parser.error("Calibration bounds are count 1..64, seconds 30..480, MTP packets 0..8")
    from mlx2.adapters.nemotron35_media import PROCESSOR_REVISION
    from mlx2.adapters.nemotron35_super import inspect_artifact
    from mlx2.adapters.nemotron35_super_config import (
        SOURCE_CONFIG_SHA256,
        SOURCE_INDEX_SHA256,
        SOURCE_REVISION,
    )
    from mlx2.conversion.calibration_corpus import verify_corpus
    from mlx2.conversion.calibration_inputs import (
        image_text_input,
        interleaved_records,
        unit_text_packets,
    )
    from mlx2.conversion.importance import ActivationMoments, statistics_specs
    from mlx2.conversion.nemotron35_plan import make_plan

    owner = common.require_owner(args.cpg_owner_lock)
    source = common.source_identity()
    artifact = inspect_artifact(args.model)
    unit_receipt = json.loads(args.unit_receipt.read_text())
    validate_unit_receipt(unit_receipt, owner=owner,
                          artifact=artifact["identity"]["fingerprint"], source_sha=source["sha256"])
    manifest, records = verify_corpus(args.corpus)
    records = interleaved_records(records)
    if not 0 <= args.start < len(records):
        parser.error("Calibration start is outside the verified corpus")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Refusing to overwrite calibration unit evidence")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.out_dir).free < 52_000_000_000:
        raise OSError("Calibration evidence would consume the 50 GB free-space reserve")
    binding = {"source_revision": SOURCE_REVISION, "source_config_sha256": SOURCE_CONFIG_SHA256,
               "source_index_sha256": SOURCE_INDEX_SHA256, "proxy_artifact": artifact["identity"]["fingerprint"],
               "corpus_sha256": manifest["corpus_sha256"], "collector_sha256": source["sha256"],
               "processor_revision": PROCESSOR_REVISION}
    binding["dependencies"] = unit_receipt["dependencies"]
    moments = ActivationMoments(statistics_specs(make_plan(artifact["config"], expert_bits=artifact["conversion"]["expert_bits"])), binding)
    receipt = {"schema": "mlx2.nemotron35-calibration-unit.v1", "state": "running", "owner": owner,
               "binding": binding, "source": source, "start": args.start, "image_samples": [], "text_samples": [],
               "qualified": False, "performance_evidence": False, "active_sessions": 1}
    adapter, previous = None, {}
    started = time.monotonic()
    common.atomic_json(args.out_dir / "unit.json", receipt)
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, lambda signum, frame: sys.exit(128 + signum))
        before = common.host_snapshot()
        errors = common.admission_errors(before, minimum_available_bytes=artifact["conversion"]["tensor_bytes"] + 8 * common.GIB)
        receipt["host_before"] = before
        if errors:
            raise RuntimeError("Calibration admission failed: " + "; ".join(errors))
        import mlx.core as mx

        # Full model plus transient image/expert activations must fit 128 GiB.
        # The default free-cache allowance can retain many GiB across shapes.
        mx.set_cache_limit(common.GIB)
        receipt["allocator_cache_limit_bytes"] = common.GIB

        from mlx2.adapters.nemotron35_super import Nemotron35SuperAdapter
        from mlx2.conversion.calibration import capture_sample

        adapter = Nemotron35SuperAdapter(str(args.model))
        for index in range(args.start, min(args.start + args.count, len(records))):
            common.require_owner(args.cpg_owner_lock)
            pre = common.host_snapshot()
            errors = common.admission_errors(pre, minimum_available_bytes=4 * common.GIB)
            if errors:
                raise RuntimeError("Calibration sample admission failed: " + "; ".join(errors))
            if time.monotonic() - started > args.max_seconds or other_waiters(pre, owner):
                receipt["yield_reason"] = "ownership time bound or another GPU waiter"
                break
            record = records[index]
            tokens, pixels, digest = image_text_input(adapter, args.corpus, record)
            sample = capture_sample(adapter.model, moments, mx.array(tokens, dtype=mx.int32)[None],
                                    sample_id=record["sample_id"], input_sha256=digest, image_pixels=pixels)
            del tokens, pixels
            sample["memory_after_release"] = release_sample_memory(mx)
            post = common.host_snapshot()
            sample.update(corpus_index=index, input_sha256=digest, host_before=pre, host_after=post,
                          contamination=common.contamination(pre, post))
            receipt["image_samples"].append(sample)
            common.atomic_json(args.out_dir / "unit.json", receipt)
            if sample["contamination"]:
                raise RuntimeError("Calibration host guard changed during a sample")
        packets = unit_text_packets(
            adapter,
            records,
            start=args.start,
            image_count=len(receipt["image_samples"]),
            packet_count=args.mtp_packets,
        )
        for packet in packets:
            common.require_owner(args.cpg_owner_lock)
            pre = common.host_snapshot()
            errors = common.admission_errors(pre, minimum_available_bytes=4 * common.GIB)
            if errors:
                raise RuntimeError("MTP calibration admission failed: " + "; ".join(errors))
            if time.monotonic() - started > args.max_seconds or other_waiters(pre, owner):
                break
            tokens, digest = packet["tokens"], packet["input_sha256"]
            sample = capture_sample(adapter.model, moments, mx.array(tokens, dtype=mx.int32)[None],
                                    sample_id="mtp-text:" + digest, input_sha256=digest, capture_mtp=True)
            del tokens
            sample["memory_after_release"] = release_sample_memory(mx)
            post = common.host_snapshot()
            sample.update(input_sha256=digest,
                          source_record_ids=list(packet["source_record_ids"]),
                          host_before=pre, host_after=post,
                          contamination=common.contamination(pre, post))
            receipt["text_samples"].append(sample)
            common.atomic_json(args.out_dir / "unit.json", receipt)
            if sample["contamination"]:
                raise RuntimeError("Calibration host guard changed during an MTP packet")
        receipt.update(state="completed" if receipt["image_samples"] else "yielded", peak_active_bytes=mx.get_peak_memory())
    except BaseException as error:
        receipt.update(state="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        finalization_errors = []
        if adapter is not None:
            try:
                adapter.close()
            except (OSError, RuntimeError, ValueError) as error:
                finalization_errors.append("adapter cleanup: " + str(error))
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["coverage"] = moments.coverage()
        receipt["next_image_index"] = args.start + len(receipt["image_samples"])
        if moments.samples:
            try:
                moments.save(args.out_dir / "moments")
                receipt["moments_saved"] = True
            except (OSError, RuntimeError, ValueError) as error:
                finalization_errors.append("moments publication: " + str(error))
        try:
            receipt["source_unchanged"] = source["sha256"] == common.source_identity()["sha256"]
        except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as error:
            receipt["source_unchanged"] = False
            finalization_errors.append("source identity: " + str(error))
        try:
            receipt["owner_after"] = common.require_owner(args.cpg_owner_lock)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
            finalization_errors.append("ownership: " + str(error))
        try:
            if common.runtime_dependencies() != binding["dependencies"]:
                finalization_errors.append("dependency/native-build identity changed during calibration")
        except (OSError, RuntimeError, ValueError) as error:
            finalization_errors.append("dependency identity: " + str(error))
        receipt["finalization_errors"] = finalization_errors
        receipt["eligible_for_merge"] = (receipt["state"] in {"completed", "yielded"}
                                         and not finalization_errors and receipt["source_unchanged"]
                                         and bool(receipt.get("moments_saved")))
        common.atomic_json(args.out_dir / "unit.json", receipt)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
