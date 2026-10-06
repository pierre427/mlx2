"""Bounded, source-bound native numeric/state parity matrix; dry-run by default.

GPU execution is only for an owned shared-gpuq lease. The matrix checks the
opaque native arena, then the pinned Qwen3-0.6B model and atomic publication.
It is an offline candidate screen, not route qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import signal
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = Path("/tmp/mlx2-varlen-price-eaed9ea1/artifact.json")
MODEL = Path("/tmp/mlx2-varlen-price-eaed9ea1/model")
MLX_LM_SOURCE = Path.home() / "Desktop/mlx-uag/mlx-lm-unified"
SOURCE = (
    "native/paged_kv/arena.cpp", "native/paged_kv/arena.h",
    "native/paged_kv/binding.cpp", "src/mlx2/runtime/paged_attention_native.py",
    "src/mlx2/runtime/paged_attention_pack.py", "src/mlx2/runtime/paged_kv_token.py",
    "src/mlx2/runtime/paged_native_atomic_owner.py",
    "src/mlx2/runtime/qwen3_paged_native_backend.py",
    "src/mlx2/adapters/qwen3_paged_candidate.py",
    "scripts/research/varlen_numeric_state_matrix.py",
    "scripts/research/varlen_qwen3_06b_artifact_gate.py",
    "scripts/research/varlen_atomic_qwen3_m5_spot.py",
)
CASES = (
    ("B1-d128-single", 2, 2, 128, (1,), (1,), None),
    ("B1-d128-page63", 4, 2, 128, (63,), (1,), None),
    ("B2-d128-ragged63-65", 4, 2, 128, (63, 65), (2, 3), None),
    ("B2-d128-window", 8, 2, 128, (130, 64), (1, 2), 32),
    ("B1-d256-gqa", 4, 1, 256, (66,), (2,), None),
    ("B2-d256-page64-129", 4, 2, 256, (64, 129), (1, 1), None),
)


def validate_cases() -> None:
    for name, qh, kh, dim, lengths, rows, window in CASES:
        if (not name or qh < 1 or kh < 1 or qh % kh or dim not in (128, 256)
                or not lengths or len(lengths) != len(rows)
                or any(length < 1 or row < 1 or row > length
                       for length, row in zip(lengths, rows))
                or window is not None and window < 1):
            raise ValueError(f"invalid native matrix case: {name}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity() -> dict:
    binary = importlib.util.find_spec("_paged_kv_native")
    if binary is None or binary.origin is None:
        raise RuntimeError("native extension missing")
    manifest = json.loads(ARTIFACT.read_text())
    for name, expected in manifest["files"].items():
        if sha256(MODEL / name) != expected:
            raise RuntimeError(f"model artifact changed: {name}")
    return {
        "source_revision": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCE},
        "native_binary": {"path": str(Path(binary.origin).resolve()),
                          "sha256": sha256(Path(binary.origin))},
        "model_manifest_sha256": sha256(ARTIFACT),
        "model_config_sha256": manifest["files"]["config.json"],
        "model_weights_sha256": manifest["files"]["model.safetensors"],
    }


def preflight(frozen: Path, session: str, lease: str) -> dict:
    current = identity()
    expected = json.loads(frozen.read_text())
    if expected["identity"] != current:
        raise RuntimeError("frozen source, model, or native binary drifted")
    for name in ("/Users/Shared/mlxuag/gpu.lock", "/tmp/gpu.lock"):
        owner = json.loads((Path(name) / "owner.json").read_text())
        if owner.get("session") != session or owner.get("lease_id") != lease:
            raise RuntimeError(f"GPU lock owner mismatch: {name}")
    with urllib.request.urlopen("http://127.0.0.1:8600/health", timeout=2) as response:
        music = json.load(response)
    if music.get("busy") is not False or music.get("loaded") is not False:
        raise RuntimeError("Music3 is busy or loaded")
    return current


def wait_writes(owner, tickets, offset: int) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        owner.poll_completions()
        if owner.writer.poisoned:
            raise RuntimeError("native writer poisoned")
        if owner.offset == offset and not owner.writer.pending_epochs:
            return
        time.sleep(0.01)
    raise TimeoutError("native write terminal callbacks missing")


def matrix_cell(case) -> dict:
    import mlx.core as mx
    import numpy as np
    from mlx2.runtime.paged_attention_metal import paged_attention_cpu_reference
    from mlx2.runtime.paged_attention_native import (
        native_paged_attention_read_fp16, poll_native_paged_read_events,
        complete_packed_read_after_event,
    )
    from mlx2.runtime.paged_attention_pack import prepare_packed_token_read
    from mlx2.runtime.paged_kv_pool import PagedKVPool
    from mlx2.runtime.paged_kv_token import PagedKVTokenOwner, TokenKVProfile
    from mlx2.runtime.paged_kv_write import NativeWriteBackend, PagedKVWriteOwner

    name, qh, kh, dim, lengths, rows, window = case
    rng = np.random.default_rng(314159 + sum(lengths) + dim + qh)
    profile = TokenKVProfile(kh, dim, "float16")
    pages = sum((n + 63) // 64 for n in lengths) + 2
    pool = PagedKVPool(pages)
    stream = mx.default_stream(mx.gpu)
    native = NativeWriteBackend(pages * profile.page_bytes, stream,
                                permit_candidate=True)
    writer = PagedKVWriteOwner(pool, native, page_bytes=profile.page_bytes,
                               permit_candidate=True)
    owners, logical = [], []
    dependency = None
    for length in lengths:
        owner = PagedKVTokenOwner(writer, profile, permit_candidate=True)
        key = rng.normal(0, 0.25, (length, kh, dim)).astype(np.float16)
        value = rng.normal(0, 0.25, key.shape).astype(np.float16)
        spans = owner.planned_spans(length)
        def chunks(array):
            return tuple(mx.array(np.frombuffer(
                array[span.source_token_offset:span.source_token_offset + span.token_count,
                      span.kv_head].tobytes(), dtype=np.uint8).copy())
                for span in spans)
        tickets = owner.append(chunks(key), chunks(value), token_count=length)
        dependency = tickets[-1].dependency
        wait_writes(owner, tickets, length)
        owners.append(owner)
        logical.append((key, value))
    mx.eval(dependency)
    if np.array(dependency).tolist() != [1]:
        raise AssertionError("write dependency did not resolve")
    packed = prepare_packed_token_read(
        tuple(owners), rows, query_heads=qh,
        masks=tuple(("sliding", window) if window else ("causal", None)
                    for _ in owners), permit_candidate=True)
    key_arena = np.zeros((pages, kh, 64, dim), dtype=np.float16)
    value_arena = np.zeros_like(key_arena)
    for lane, (key, value) in enumerate(logical):
        for token in range(len(key)):
            handle, position = packed.plan.page_for_token(lane, token)
            key_arena[handle.page_id, :, position] = key[token]
            value_arena[handle.page_id, :, position] = value[token]
    query = rng.normal(0, 0.25, (sum(rows), qh, dim)).astype(np.float16)
    expected = paged_attention_cpu_reference(
        packed.plan, query, key_arena, value_arena)
    output = native_paged_attention_read_fp16(
        packed, native, mx.array(query), dependency, permit_candidate=True)
    mx.eval(output)
    mx.synchronize(stream)
    events = poll_native_paged_read_events(native)
    if events != ((packed.lease.epoch, True),):
        raise AssertionError(f"unexpected read callbacks: {events}")
    complete_packed_read_after_event(packed, events[0])
    actual = np.asarray(output, dtype=np.float32)
    reference = expected.astype(np.float32)
    error = actual - reference
    nrms = float(np.sqrt(np.mean(error.astype(np.float64) ** 2)) /
                 max(np.sqrt(np.mean(reference.astype(np.float64) ** 2)), 1e-12))
    max_abs = float(np.max(np.abs(error)))
    for owner in owners:
        owner.close()
    pool.retire(writer.ledger.completed_epoch)
    if pool.free_count != pages or writer.ledger.pending_count:
        raise AssertionError("native matrix cell leaked page/lease")
    if nrms > 0.003 or max_abs > 0.02:
        raise AssertionError(f"{name}: numeric mismatch nrms={nrms} max_abs={max_abs}")
    return {"name": name, "query_heads": qh, "kv_heads": kh,
            "head_dim": dim, "kv_lengths": lengths, "query_rows": rows,
            "window": window, "normalized_rms": nrms, "max_abs": max_abs,
            "native_terminal_success": True, "pages_retired": True}


def gpu_run() -> dict:
    import mlx.core as mx
    if not mx.metal.is_available():
        raise RuntimeError("Metal unavailable")
    mx.set_default_device(mx.gpu)
    cells = [matrix_cell(case) for case in CASES]
    # Reuse the already source-pinned offline model and two-layer atomic gates.
    import sys
    sys.path.insert(0, str(ROOT / "scripts/research"))
    from varlen_qwen3_06b_artifact_gate import gpu_gate, preflight as model_preflight
    from varlen_atomic_qwen3_m5_spot import cell as atomic_cell
    model = gpu_gate(
        MODEL,
        MLX_LM_SOURCE,
        model_preflight(MODEL, MLX_LM_SOURCE),
    )
    atomic = atomic_cell()
    return {"native_matrix": cells,
            "model_comparisons": model["comparisons"],
            "atomic_rounds": atomic["rounds"],
            "atomic_pages_retired": atomic["all_pages_retired"],
            "bf16_native_status": "unsupported; explicit fp16-only admission",
            "qualification": False, "serving_route_selected": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-gpu", action="store_true")
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--session")
    parser.add_argument("--lease-id")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    validate_cases()
    result = {"schema": "mlx2.varlen-numeric-state-matrix.v1",
              "identity": identity(), "cases": [case[0] for case in CASES],
              "gpu_executed": False}
    if args.execute_gpu:
        signal.alarm(300)
        if not args.preflight or not args.session or not args.lease_id:
            parser.error("GPU mode requires frozen preflight and owned session/lease")
        preflight(args.preflight, args.session, args.lease_id)
        try:
            result.update(gpu_run(), gpu_executed=True, passed=True)
        except BaseException as error:
            result.update(gpu_executed=True, passed=False,
                          error=f"{type(error).__name__}: {error}")
            raise
        finally:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(json.dumps(result, indent=2, default=str) + "\n")
    else:
        result.update(mode="dry-run", qualification=False,
                      bf16_native_status="unsupported; explicit fp16-only admission")
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
