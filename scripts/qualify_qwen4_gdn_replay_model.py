#!/usr/bin/env python3
"""Model-bound Qwen4 compact GDN replay qualification.

Runs snapshot and compact rollback on separate caches from the same prefill,
then compares verify logits, restored recurrent state, and continuation logits.
The script requires external GPU lock receipts and never selects a serving arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
HOST_LOCK = Path("/Users/Shared/mlxuag/gpu.lock/owner.json")


def _require_lock(task_id: str, *, now: float | None = None) -> dict:
    """Require this process's cpg_job parent, host lock, and live lease receipt."""
    now = time.time() if now is None else now
    if not task_id or not HOST_LOCK.is_file():
        raise RuntimeError("GPU qualification requires a CPG task and host lock")
    owner = json.loads(HOST_LOCK.read_text())
    if not isinstance(owner, dict) or owner.get("agent") != "cpg_job":
        raise RuntimeError("host GPU lock is not owned by cpg_job")
    if type(owner.get("pid")) is not int or owner["pid"] != os.getppid():
        raise RuntimeError("host GPU lock belongs to another process")
    if owner.get("cpg_task") != task_id:
        raise RuntimeError("host GPU lock belongs to another CPG task")
    if any(not isinstance(owner.get(key), str) or not owner[key]
           for key in ("agent_id", "cpg_session", "cpg_worker", "log")):
        raise RuntimeError("host GPU lock lacks CPG identity")
    generation = owner.get("cpg_generation")
    if type(generation) is not int or generation < 1:
        raise RuntimeError("host GPU lock lacks a positive CPG generation")
    radio_path = Path(owner["log"] + ".radio.json")
    radio = json.loads(radio_path.read_text())
    if not isinstance(radio, dict) or radio.get("agent_id") != owner["agent_id"]:
        raise RuntimeError("CPG radio receipt has another owner")
    if "release_task" in radio or "complete_worker" in radio:
        raise RuntimeError("CPG radio receipt is already closed")
    claim = radio.get("claim_task")
    if not isinstance(claim, dict) or claim.get("claimed") is not True:
        raise RuntimeError("CPG radio receipt has no successful claim")
    if (claim.get("task_id") != task_id
            or claim.get("lease_generation") != generation
            or claim.get("owner_agent_id") != owner["agent_id"]):
        raise RuntimeError("CPG claim does not match the host lock")
    lease = radio.get("renew_lease", claim)
    if (not isinstance(lease, dict)
            or lease.get("task_id") != task_id
            or lease.get("lease_generation") != generation
            or (lease is not claim and lease.get("renewed") is not True)
            or type(lease.get("lease_expires_at")) not in (int, float)
            or lease["lease_expires_at"] <= now):
        raise RuntimeError("CPG GPU lease receipt is mismatched or expired")
    return {"host_lock": owner, "radio_claim": claim,
            "radio_renewal": radio.get("renew_lease")}


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def _source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src" / "mlx2").rglob("*.py")):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _compare(mx, left, right) -> dict:
    delta = mx.abs(left.astype(mx.float32) - right.astype(mx.float32))
    maximum = mx.max(delta)
    exact = mx.all(left == right)
    finite = mx.all(mx.isfinite(left)) & mx.all(mx.isfinite(right))
    mx.eval(maximum, exact, finite)
    return {
        "exact": bool(exact.item()),
        "finite": bool(finite.item()),
        "max_abs": float(maximum.item()),
    }


# --dynamic-accept: the compact arm reconstructs from a device-count kernel.
_DYNAMIC_ACCEPT = False


def _set_mode(model, replay_mode: str) -> int:
    from mlx2.runtime.models.qwen4_exp import GatedDeltaNet

    count = 0
    for _, module in model.named_modules():
        if isinstance(module, GatedDeltaNet):
            module.set_fused_gdn_verify_mode("fused")
            module.set_fused_gdn_replay_rollback_mode(replay_mode)
            if _DYNAMIC_ACCEPT:
                module.set_fused_gdn_dynamic_accept(replay_mode == "compact")
            count += 1
    return count


def _eval_linear_cache(mx, cache, linear_indices):
    values = [cache[index][slot] for index in linear_indices for slot in (0, 1)]
    mx.eval(*values)


def _start(cache):
    for entry in cache:
        entry.start_speculation()


def _stop(cache):
    first = None
    for entry in cache:
        try:
            entry.stop_speculation()
        except BaseException as exc:
            if first is None:
                first = exc
    if first is not None:
        raise first


def _trim(cache, count: int):
    applied = [int(entry.trim(count)) for entry in cache]
    if any(value != count for value in applied):
        raise RuntimeError(f"cache trim diverged: expected {count}, got {applied}")


def _prefill(mx, model, tokens):
    cache = model.make_cache()
    logits = model(mx.array(tokens)[None], cache=cache)
    mx.eval(logits)
    return cache, logits


def _run_arm(mx, model, cache, block, accepted, mode, linear_indices):
    from mlx2.runtime.models.qwen4_exp import qwen4_fused_gdn_stats

    _set_mode(model, mode)
    qwen4_fused_gdn_stats(model, reset=True)
    _start(cache)
    mx.clear_cache()
    mx.reset_peak_memory()
    baseline = int(mx.get_active_memory())
    started = time.perf_counter_ns()
    verify_logits = model(mx.array(block)[None], cache=cache)
    mx.eval(verify_logits)
    verify_ms = (time.perf_counter_ns() - started) / 1e6
    peak_delta = max(0, int(mx.get_peak_memory()) - baseline)
    verify_stats = qwen4_fused_gdn_stats(model)

    started = time.perf_counter_ns()
    _trim(cache, len(block) - accepted)
    _eval_linear_cache(mx, cache, linear_indices)
    rollback_ms = (time.perf_counter_ns() - started) / 1e6
    rollback_stats = qwen4_fused_gdn_stats(model)
    _stop(cache)
    return {
        "verify_logits": verify_logits,
        "verify_ms": verify_ms,
        "rollback_ms": rollback_ms,
        "peak_delta_bytes": peak_delta,
        "verify_stats": verify_stats,
        "rollback_stats": rollback_stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--widths", default="3,4,8")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpg-lease", required=True, help="exact CPG GPU task ID held by cpg_job")
    parser.add_argument(
        "--prompt",
        default="Explain why exact transactional state matters in speculative decoding.",
    )
    parser.add_argument(
        "--dynamic-accept",
        action="store_true",
        help="compact arm uses the device-count reconstruct kernel",
    )
    args = parser.parse_args()
    widths = list(map(int, args.widths.split(",")))
    if not widths or any(width < 2 for width in widths) or args.repeats < 1:
        parser.error("widths must be >=2 and repeats must be >=1")
    global _DYNAMIC_ACCEPT
    _DYNAMIC_ACCEPT = args.dynamic_accept
    locks = _require_lock(args.cpg_lease)

    import mlx.core as mx
    from mlx2.adapters.flash_next import FlashNextAdapter

    mx.set_default_device(mx.gpu)
    started_at = time.time()
    adapter = FlashNextAdapter(str(args.model))
    try:
        # The adapter pins its execution environment before importing tensor
        # modules. Keep the qualification oracle on that same production load
        # order so module-level defaults reflect the selected profile.
        from mlx2.runtime.models.qwen4_exp import (
            GatedDeltaNet,
            qwen4_fused_gdn_stats,
        )
        from mlx2.runtime.models.qwen4_fused_gdn_verify import (
            probe_qwen4_fused_gdn_replay_verify,
            probe_qwen4_fused_gdn_verify,
        )

        prompt = adapter.tokenizer.encode(args.prompt, add_special_tokens=False)
        if len(prompt) < 4:
            raise ValueError("qualification prompt encoded to fewer than four tokens")
        linear_indices = [
            index for index, layer in enumerate(adapter.model.layers) if layer.is_linear
        ]
        initial_replay_modes = sorted(
            {
                module.fused_gdn_replay_rollback_mode
                for _, module in adapter.model.named_modules()
                if isinstance(module, GatedDeltaNet)
            }
        )
        profile_default_selected = bool(
            adapter.environment.get("MLX_QWEN4_FUSED_GDN_REPLAY_ROLLBACK") == "1"
            and initial_replay_modes == ["compact"]
        )
        probes = {
            str(width): {
                "snapshot": probe_qwen4_fused_gdn_verify(mx.bfloat16, width),
                "compact": probe_qwen4_fused_gdn_replay_verify(mx.bfloat16, width),
            }
            for width in widths
        }
        results = []
        for width in widths:
            block = (prompt * ((width + len(prompt) - 1) // len(prompt)))[:width]
            accepted = max(1, width // 2)
            repetitions = []
            for repetition in range(args.repeats):
                snapshot_cache, snapshot_prefill = _prefill(
                    mx, adapter.model, prompt
                )
                compact_cache, compact_prefill = _prefill(mx, adapter.model, prompt)
                prefill_parity = _compare(mx, snapshot_prefill, compact_prefill)
                order = ("snapshots", "compact") if repetition % 2 == 0 else (
                    "compact",
                    "snapshots",
                )
                arms = {}
                for mode in order:
                    cache = snapshot_cache if mode == "snapshots" else compact_cache
                    arms[mode] = _run_arm(
                        mx,
                        adapter.model,
                        cache,
                        block,
                        accepted,
                        mode,
                        linear_indices,
                    )

                verify_parity = _compare(
                    mx,
                    arms["snapshots"]["verify_logits"],
                    arms["compact"]["verify_logits"],
                )
                cache_parity = []
                for index in linear_indices:
                    cache_parity.append(
                        {
                            "layer": index,
                            "conv": _compare(
                                mx, snapshot_cache[index][0], compact_cache[index][0]
                            ),
                            "recurrent": _compare(
                                mx, snapshot_cache[index][1], compact_cache[index][1]
                            ),
                        }
                    )

                continuation = block[accepted - 1]
                snapshot_next = adapter.model(
                    mx.array([[continuation]]), cache=snapshot_cache
                )
                compact_next = adapter.model(
                    mx.array([[continuation]]), cache=compact_cache
                )
                mx.eval(snapshot_next, compact_next)
                next_parity = _compare(mx, snapshot_next, compact_next)
                snapshot_argmax = int(mx.argmax(snapshot_next[:, -1], axis=-1).item())
                compact_argmax = int(mx.argmax(compact_next[:, -1], axis=-1).item())
                repetitions.append(
                    {
                        "repetition": repetition,
                        "order": order,
                        "prefill_parity": prefill_parity,
                        "verify_parity": verify_parity,
                        "cache_parity": cache_parity,
                        "continuation_parity": next_parity,
                        "same_continuation_argmax": snapshot_argmax == compact_argmax,
                        "arms": {
                            mode: {
                                key: value
                                for key, value in arm.items()
                                if key != "verify_logits"
                            }
                            for mode, arm in arms.items()
                        },
                    }
                )
                del snapshot_cache, compact_cache, snapshot_prefill, compact_prefill
                mx.clear_cache()

            snapshot_times = [
                row["arms"]["snapshots"]["verify_ms"] for row in repetitions
            ]
            compact_times = [
                row["arms"]["compact"]["verify_ms"] for row in repetitions
            ]
            snapshot_rounds = [
                row["arms"]["snapshots"]["verify_ms"]
                + row["arms"]["snapshots"]["rollback_ms"]
                for row in repetitions
            ]
            compact_rounds = [
                row["arms"]["compact"]["verify_ms"]
                + row["arms"]["compact"]["rollback_ms"]
                for row in repetitions
            ]
            snapshot_peaks = [
                row["arms"]["snapshots"]["peak_delta_bytes"]
                for row in repetitions
            ]
            compact_peaks = [
                row["arms"]["compact"]["peak_delta_bytes"]
                for row in repetitions
            ]
            results.append(
                {
                    "width": width,
                    "accepted": accepted,
                    "repetitions": repetitions,
                    "timing": {
                        "snapshot_median_ms": statistics.median(snapshot_times),
                        "compact_median_ms": statistics.median(compact_times),
                        "speedup": statistics.median(snapshot_times)
                        / statistics.median(compact_times),
                        "snapshot_verify_plus_rollback_median_ms": statistics.median(
                            snapshot_rounds
                        ),
                        "compact_verify_plus_rollback_median_ms": statistics.median(
                            compact_rounds
                        ),
                        "verify_plus_rollback_speedup": statistics.median(
                            snapshot_rounds
                        )
                        / statistics.median(compact_rounds),
                    },
                    "memory": {
                        "snapshot_peak_delta_median_bytes": statistics.median(
                            snapshot_peaks
                        ),
                        "compact_peak_delta_median_bytes": statistics.median(
                            compact_peaks
                        ),
                        "peak_delta_reduction_ratio": statistics.median(
                            snapshot_peaks
                        )
                        / statistics.median(compact_peaks),
                    },
                }
            )

        exact = all(
            row[name]["exact"] and row[name]["finite"]
            for result in results
            for row in result["repetitions"]
            for name in ("prefill_parity", "verify_parity", "continuation_parity")
        ) and all(
            layer[name]["exact"] and layer[name]["finite"]
            for result in results
            for row in result["repetitions"]
            for layer in row["cache_parity"]
            for name in ("conv", "recurrent")
        )
        engaged = all(
            row["arms"]["compact"]["rollback_stats"]["replay_verify_calls"]
            == len(linear_indices)
            and row["arms"]["compact"]["rollback_stats"]["replay_rollback_calls"]
            == len(linear_indices)
            and row["arms"]["compact"]["rollback_stats"]["replay_fallbacks"] == 0
            and row["arms"]["snapshots"]["rollback_stats"]["verify_calls"]
            == len(linear_indices)
            for result in results
            for row in result["repetitions"]
        )
        same_tokens = all(
            row["same_continuation_argmax"]
            for result in results
            for row in result["repetitions"]
        )
        report = {
            "schema": "mlx2.qwen4-gdn-replay-model-qualification.v1",
            "status": (
                "qualified_selected"
                if exact and engaged and same_tokens and profile_default_selected
                else "failed"
            ),
            "qualified": exact and engaged and same_tokens and profile_default_selected,
            "selected": profile_default_selected,
            "observed_used": engaged,
            "profile_default": {
                "environment": adapter.environment.get(
                    "MLX_QWEN4_FUSED_GDN_REPLAY_ROLLBACK"
                ),
                "initial_module_modes": initial_replay_modes,
                "passed": profile_default_selected,
            },
            "source_hash": _source_hash(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "git_head": _git_head(),
            "model": str(args.model.resolve()),
            "model_identity": adapter.identity,
            "device": mx.device_info(),
            "locks": locks,
            "prompt_tokens": len(prompt),
            "linear_gdn_layers": len(linear_indices),
            "probes": probes,
            "dynamic_accept": args.dynamic_accept,
            "results": results,
            "final_stats": qwen4_fused_gdn_stats(adapter.model),
            "started_at": started_at,
            "finished_at": time.time(),
            "limitations": [
                "Bounded single-lane model qualification, not a serving load test.",
                "Peak deltas are MLX allocator observations, not process RSS.",
                "Selection is scoped to the Flash-Next adapter profile.",
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "qualified": report["qualified"],
                    "observed_used": report["observed_used"],
                }
            )
        )
        return 0 if report["qualified"] else 1
    finally:
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
