#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU scheduling around the unchanged, source-bound IQ1 repacker.

No tensor algebra is implemented here. Each worker imports the original frozen
write_expert/recipe functions and publishes the same atomic per-file receipt.
The coordinator holds the artifact flock until every running worker joins,
including on failure/cancellation. The original serial repacker performs final
COW cloning and integrity publication after that lock is released.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import time

import repack_deepseek_v41_iq1 as base

_WORKER_RECIPE_HASH = None


def _initialize_worker():
    # Parent cancellation joins active writes at a file boundary. CLI SIGTERM
    # also reaches the coordinator's handler; process-group termination still
    # leaves only unreceipted partial files for a later resume.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def write_one(task):
    global _WORKER_RECIPE_HASH
    if _WORKER_RECIPE_HASH is None:
        _WORKER_RECIPE_HASH = hashlib.sha256(base.canonical(base.recipe())).hexdigest()
    if _WORKER_RECIPE_HASH != task["recipe_sha256"]:
        raise ValueError("parallel worker inverse recipe differs from admitted artifact")
    source, output, relative = Path(task["source"]), Path(task["output"]), task["path"]
    def source_check():
        if base.stamp(source / "mlx2-v41-native-manifest.json") != task["source_manifest_stat"]:
            raise ValueError("affine source manifest changed during parallel repacking")
    source_check()
    base.check_space(output, task["output_bytes"], task["reserve_bytes"])
    target = base.safe_path(output, relative)
    partial = base.safe_path(output, relative + ".partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    record = base.write_expert(source, relative, task["source_record"], task["shapes"], partial)
    source_check()
    record["sha256"] = base.file_hash(partial)
    record["execution"] = dict(orchestrator_sha256=task["orchestrator_sha256"],
                               worker_pid=os.getpid(), started_unix=started,
                               elapsed_seconds=time.time() - started,
                               inverse_recipe_sha256=_WORKER_RECIPE_HASH)
    os.replace(partial, target)
    record["stat"] = base.stamp(target)
    base.verify_file(output, relative, record)
    receipt = base.safe_path(output, "conversion-state/" + hashlib.sha256(relative.encode()).hexdigest() + ".json")
    base.atomic_json(receipt, dict(path=relative, recipe_sha256=task["recipe_sha256"], record=record))
    return dict(path=relative, size=record["size"], elapsed_seconds=record["execution"]["elapsed_seconds"])


def drain_bounded(tasks, workers, *, completed=lambda result: None, executor_factory=None):
    """Keep at most workers tasks submitted; always join before returning."""
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("parallel repacking supports1..4 bounded CPU workers")
    factory = executor_factory or (lambda count: ProcessPoolExecutor(
        max_workers=count, mp_context=multiprocessing.get_context("spawn"), initializer=_initialize_worker))
    executor = factory(workers)
    pending = set()
    total = 0
    iterator = iter(tasks)
    exhausted = False
    try:
        while pending or not exhausted:
            while not exhausted and len(pending) < workers:
                try:
                    task = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                pending.add(executor.submit(write_one, task))
            if not pending:
                continue
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.remove(future)
                result = future.result()
                completed(result)
                total += 1
    finally:
        for future in pending:
            future.cancel()
        # This join occurs inside the caller's conversion_lock context.
        executor.shutdown(wait=True, cancel_futures=True)
    return total


def parallel_repack(source, output, *, workers=4, reserve_bytes=100 << 30,
                    max_files=None, finalize=True, phase=print):
    from mlx2.adapters.deepseek_v41_mlx_artifact import inspect_artifact
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("parallel repacking supports1..4 bounded CPU workers")
    if max_files is not None and (type(max_files) is not int or max_files < 0):
        raise ValueError("parallel max_files must be a nonnegative integer")
    source, output = Path(source).resolve(), Path(output)
    if output.is_symlink():
        raise ValueError("parallel artifact root is a symlink")
    output = output.resolve()
    if not (output / base.MANIFEST).exists():
        base.repack(source, output, reserve_bytes=reserve_bytes, max_files=0, phase=phase)
    admitted = inspect_artifact(source)
    original = admitted["manifest"]
    tensors, experts, sizes, _ = base.build_plan(original)
    del tensors
    source_stamp = admitted["weight_stat"]
    fingerprint = admitted["identity"]["fingerprint"]
    recipe_hash = hashlib.sha256(base.canonical(base.recipe())).hexdigest()
    wrapper_hash = base.file_hash(__file__)
    started = time.monotonic()
    with base.conversion_lock(output):
        manifest = json.loads((output / base.MANIFEST).read_text())
        if (manifest.get("finished") or manifest.get("recipe_sha256") != recipe_hash
                or manifest.get("source_affine", {}).get("fingerprint") != fingerprint
                or manifest["source_affine"].get("derived_identity_sha256") != original["derived_identity_sha256"]
                or manifest["source_affine"].get("path") != str(source)):
            raise ValueError("parallel resume source/recipe/completion binding differs")
        missing, resumed = [], 0
        for relative in sorted(experts):
            receipt_path = base.safe_path(output, "conversion-state/" + hashlib.sha256(relative.encode()).hexdigest() + ".json")
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if receipt.get("path") != relative or receipt.get("recipe_sha256") != recipe_hash:
                    raise ValueError("parallel resume receipt binding differs")
                record = receipt["record"]
                if base.verify_file(output, relative, record) != record["stat"]:
                    raise ValueError("parallel resume file stat seal changed")
                resumed += 1
            else:
                missing.append(relative)
        selected = missing if max_files is None else missing[:max_files]
        session = dict(orchestrator_sha256=wrapper_hash, workers=workers,
                       inverse_recipe_sha256=recipe_hash, started_unix=time.time(),
                       resumed_experts=resumed, scheduled_experts=len(selected), finished=False)
        manifest.setdefault("parallel_repack_sessions", []).append(session)
        base.atomic_json(output / base.MANIFEST, manifest)
        newly_written, byte_count = 0, 0
        def tasks():
            for relative in selected:
                if base.stamp(source / "mlx2-v41-native-manifest.json") != source_stamp:
                    raise ValueError("affine source manifest changed before scheduling")
                # Include up to workers independently growing output files.
                base.check_space(output, workers * sizes[relative], reserve_bytes)
                yield dict(source=str(source), output=str(output), path=relative,
                           source_record=original["files"][relative], shapes=experts[relative],
                           output_bytes=sizes[relative], reserve_bytes=reserve_bytes,
                           source_manifest_stat=source_stamp, recipe_sha256=recipe_hash,
                           orchestrator_sha256=wrapper_hash)
        def complete(result):
            nonlocal newly_written, byte_count
            newly_written += 1
            byte_count += result["size"]
            if newly_written % 32 == 0 or newly_written == len(selected):
                progress = dict(completed_experts=resumed + newly_written, total_experts=len(experts),
                                workers=workers, newly_written=newly_written,
                                new_bytes=byte_count, elapsed_seconds=time.monotonic() - started,
                                last_file=result["path"], updated_unix=time.time())
                base.atomic_json(output / "conversion-progress.json", progress)
                phase(json.dumps(progress))
        try:
            drain_bounded(tasks(), workers, completed=complete)
        except BaseException as error:
            session.update(error_type=type(error).__name__, error=str(error), finished_unix=time.time(),
                           newly_written=newly_written)
            base.atomic_json(output / base.MANIFEST, manifest)
            raise
        if base.stamp(source / "mlx2-v41-native-manifest.json") != source_stamp:
            raise ValueError("affine source changed before parallel completion")
        session.update(finished=True, finished_unix=time.time(), newly_written=newly_written,
                       elapsed_seconds=time.monotonic() - started)
        base.atomic_json(output / base.MANIFEST, manifest)
        summary = dict(parallel_passed=True, workers=workers, resumed_experts=resumed,
                       newly_written=newly_written, scheduled_experts=len(selected),
                       elapsed_seconds=time.monotonic() - started, orchestrator_sha256=wrapper_hash,
                       inverse_recipe_sha256=recipe_hash, finished=False)
    if finalize and len(selected) == len(missing):
        summary["final_verification"] = base.repack(source, output, reserve_bytes=reserve_bytes, phase=phase)
        summary["finished"] = bool(summary["final_verification"]["passed"])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reserve-gib", type=int, default=100)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--no-finalize", action="store_true")
    args = parser.parse_args()
    if args.reserve_gib < 0:
        parser.error("reserve must be nonnegative")
    def cancel(signum, frame):
        raise KeyboardInterrupt(f"parallel repack signal {signum}; joining workers")
    signal.signal(signal.SIGTERM, cancel)
    print(json.dumps(parallel_repack(args.source, args.output, workers=args.workers,
                                    reserve_bytes=args.reserve_gib << 30, max_files=args.max_files,
                                    finalize=not args.no_finalize), sort_keys=True))


if __name__ == "__main__":
    main()
