#!/usr/bin/env python3
"""Owned, bounded DeepSeek Vision-Exp DSpark support/greedy serving probe.

Root owns the CPG/paired lease and final process group cleanup. This targeted
functional diagnostic does not qualify DSpark or claim packed AST speedup.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from probe_deepseek_v4_packed import atomic, source_receipt, verify_ownership

REQUEST = {
    "messages": [{"role": "user", "content": "Reply with exactly Hello."}],
    "enable_thinking": False,
}


def validate_args(args):
    if not args.run_gpu or args.ownership_window is None:
        raise ValueError("explicit --run-gpu and live ownership window required")
    if args.output.exists():
        raise ValueError("refusing existing evidence directory")
    if not 1 <= args.max_tokens <= 64 or not 1 <= args.prefill_step <= 128:
        raise ValueError("bounded max_tokens1..64 and prefill_step1..128 required")
    if not 96 <= args.memory_limit_gib <= 112:
        raise ValueError(
            "memory-limit-gib96..112 required; external host reserve remains"
        )
    if type(args.execution_policy) is not dict or any(
        type(k) is not str or type(v) is not str
        for k, v in args.execution_policy.items()
    ):
        raise ValueError("execution policy must be a JSON dictionary of string choices")


def selected_adapter_policy(args):
    policy = dict(args.execution_policy)
    if set(policy) & {"draft_model", "num_draft", "deepseek_verifier", "deepseek_draft_confidence"}:
        raise ValueError(
            "use --adapter-external/--verification-mode for bounded sidecar profile"
        )
    if args.adapter_external:
        policy.update(
            draft_model=str(args.sidecar_path),
            num_draft=5,
            deepseek_verifier=args.verification_mode,
        )
    return policy


def sealed_sidecar(path):
    # Acquisition receipt claims are not authority for mutable file bytes.
    # Stream the actual6GB sidecar, check before/after stat, then mint/recheck
    # the mlx2 integrity seal. The already sealed87GB target is not rehashed.
    from mlx2.adapters.deepseek_v4_dspark_artifact import prepare_sidecar

    return prepare_sidecar(path)


def load_request(path):
    if path is None:
        return REQUEST, {"source": "built_in_Hello"}
    stat = path.stat()
    if not 2 <= stat.st_size <= 65536:
        raise ValueError("request JSON file must be2..65536 bytes")
    payload = path.read_bytes()
    value = json.loads(payload)
    if (
        type(value) is not dict
        or type(value.get("messages")) is not list
        or not 1 <= len(value["messages"]) <= 16
        or any(
            type(m) is not dict
            or m.get("role") not in ("system", "user", "assistant")
            or type(m.get("content")) is not str
            for m in value["messages"]
        )
    ):
        raise ValueError("request JSON requires1..16 plain-text chat messages")
    if value.get("enable_thinking", False) is not False:
        raise ValueError("this bounded DSpark probe requires enable_thinking=false")
    value["enable_thinking"] = False
    if any(key in value for key in ("tools", "images", "audio", "video")):
        raise ValueError("this DSpark targeted probe supports text only")
    return value, {
        "source": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def same_values(left, right):
    if left is None or right is None:
        return left is right
    if isinstance(left, (tuple, list)):
        return (
            isinstance(right, (tuple, list))
            and len(left) == len(right)
            and all(same_values(a, b) for a, b in zip(left, right))
        )
    return np.array_equal(np.asarray(left), np.asarray(right))


def same_snapshots(caches, snapshots):
    return all(
        tuple(c.meta_state) == tuple(s.metadata) and same_values(c.state, s.arrays)
        for c, s in zip(caches, snapshots)
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sidecar-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--ownership-window", type=Path)
    parser.add_argument("--owner-path", type=Path, action="append")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--request-json", type=Path)
    parser.add_argument("--expected-text")
    parser.add_argument("--adapter-external", action="store_true")
    parser.add_argument(
        "--verification-mode",
        choices=("sequential", "causal_slab"),
        default="sequential",
    )
    parser.add_argument("--prefill-step", type=int, default=128)
    parser.add_argument("--memory-limit-gib", type=float, default=108)
    parser.add_argument("--execution-policy", type=json.loads, default={})
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    owners = args.owner_path or [
        Path("/Users/Shared/mlxuag/gpu.lock/owner.json"),
        Path("/tmp/gpu.lock/owner.json"),
    ]
    if len(owners) != 2:
        parser.error("exactly two owner paths required")

    def guard():
        result = verify_ownership(args.ownership_window, owners)
        current = args.sidecar_path.stat()
        if (
            current.st_size,
            current.st_mtime_ns,
            current.st_ino,
            current.st_dev,
        ) != side_stat:
            raise RuntimeError("sidecar file changed after SHA-seal binding")
        return result

    request, request_binding = load_request(args.request_json)
    source = source_receipt()
    from mlx2.adapters.deepseek_v4_gguf_artifact import inspect_artifact
    from mlx2.runtime.drafters.deepseek_v4_dspark import (
        TARGET_SHA256,
        DeepSeekV4DSparkConfig,
        verify_runtime_binding,
    )
    from mlx2.runtime.gguf.reader import GGUFFile

    artifact = inspect_artifact(args.model_path)
    side = sealed_sidecar(args.sidecar_path)
    side_stat = tuple(side["stat"][k] for k in ("size", "mtime_ns", "inode", "device"))
    ownership = guard()
    index = GGUFFile(args.sidecar_path)
    config = DeepSeekV4DSparkConfig.from_index(index.metadata, index.tensors)
    index.close()
    if config.target_layers != (40, 41, 42) or config.block_size != 5:
        raise ValueError(
            "actual Vision-Exp sidecar trained capture/depth contract differs"
        )
    snapshot = source.get("source_snapshot_sha256")
    if snapshot is None:
        from mlx2.desktop_runner import digest, source_manifest

        snapshot = digest(source_manifest(ROOT))
    binding = verify_runtime_binding(
        config,
        target_sha256=TARGET_SHA256,
        sidecar_sha256=side["sha256"],
        target_metadata=artifact["metadata"],
        source_snapshot=snapshot,
    )
    args.output.mkdir(parents=True)
    receipt = {
        "schema": "mlx2.deepseek-v4-dspark-live-targeted.v1",
        "status": "running",
        "qualification": False,
        "performance_claim": False,
        "source": source,
        "binding": binding,
        "checkpoint": args.model_path.name + "@" + artifact["identity"]["fingerprint"],
        "artifact": artifact["identity"],
        "sidecar": side,
        "ownership": ownership,
        "request": request,
        "request_binding": request_binding,
        "expected_text": args.expected_text,
        "controls": {
            "sampling": "greedy",
            "thinking": False,
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "prompt_limit": 128,
            "physical_width": 1,
            "trained_proposals": 5,
            "capture_layers": [40, 41, 42],
            "verification": "ordinary_width_one_full_snapshot_replay",
            "packed_ast": False,
            "unsupported": ["temperature>0", "physical_batch>1", "packed_tree_AST"],
            "requested_execution_policy": args.execution_policy,
            "adapter_external": args.adapter_external,
            "verification_mode": args.verification_mode,
        },
        "phases": [],
    }

    def phase(name, **values):
        receipt["phase"] = name
        receipt["phases"].append({"name": name, "at_unix": time.time(), **values})
        atomic(args.output / "results.json", receipt)
        print(json.dumps({"phase": name, **values}), flush=True)

    adapter = weights = backbone = draft = generator = model = cache = None
    try:
        import mlx.core as mx

        from mlx2.process_env import require_process_numerics

        require_process_numerics("DeepSeek V4 DSpark owned targeted probe")
        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(int(args.memory_limit_gib * (1 << 30)))
        mx.set_cache_limit(256 << 20)
        from mlx2.adapters.deepseek_v4_external import (
            DeepSeekV4DSparkDraftModel,
            DeepSeekV4ExternalBatchGenerator,
            DeepSeekV4VerifyRows,
        )
        from mlx2.adapters.deepseek_v4_gguf import DeepSeekV4GGUFAdapter
        from mlx2.runtime.drafters.deepseek_v4_dspark import DeepSeekV4DSparkBackbone
        from mlx2.runtime.gguf.weights import GGUFWeights

        guard()
        phase("before_target_load")
        selected_policy = selected_adapter_policy(args)
        adapter = DeepSeekV4GGUFAdapter(
            str(args.model_path), execution_policy=selected_policy
        )
        model = adapter.model
        expected_target = adapter.identity.get(
            "target_fingerprint", adapter.identity["fingerprint"]
        )
        if expected_target != artifact["identity"]["fingerprint"]:
            raise RuntimeError("target changed during adapter load")
        receipt["route"] = adapter.diagnostics()
        receipt["controls"]["execution_policy"] = dict(adapter.kernel_policy)
        tokens = list(adapter.prompt_tokens(request))
        if not 2 <= len(tokens) <= 128:
            raise ValueError("prompt must contain2..128tokens")
        atomic(
            args.output / "prompt-tokens.json", {"tokens": tokens, "request": request}
        )
        config.validate_target(model.args)
        if args.adapter_external:
            weights = adapter.draft_weights
            backbone = adapter.draft_model.backbone
            draft = adapter.draft_model
            if backbone.identity != binding:
                raise RuntimeError(
                    "adapter sidecar/source binding differs from owned probe"
                )
        else:
            weights = GGUFWeights(
                args.sidecar_path,
                compute_policy=adapter.kernel_policy["deepseek_packed"],
            )
            backbone = DeepSeekV4DSparkBackbone(
                config,
                model.args,
                weights,
                model.weights,
                target_sha256=TARGET_SHA256,
                sidecar_sha256=side["sha256"],
                target_metadata=model.weights.metadata,
                source_snapshot=snapshot,
            )
            backbone.preload()
            draft = DeepSeekV4DSparkDraftModel(
                backbone, verification_mode=args.verification_mode
            ).bind(model)
            # The manual construction lane also needs every sidecar allocation
            # present before wiring; model construction alone remains lazy.
            for name, tensor in weights.tensors.items():
                if tensor.ggml_type in (0, 1):
                    weights.array(name)
                else:
                    weights.packed(name)
            mx.eval(list(weights._dense.values()))
        from deepseek_v4_probe_residency import wire_preloaded_weights
        wire_preloaded_weights(mx, receipt)
        receipt["execution_binding"] = draft.execution_binding
        phase("target_and_sidecar_constructed", active_bytes=mx.get_active_memory())
        cache = model.make_cache()
        features = []
        for start in range(0, len(tokens) - 1, args.prefill_step):
            guard()
            part = tokens[start : min(start + args.prefill_step, len(tokens) - 1)]
            hidden = model.prefill_body(
                mx.array([part], dtype=mx.uint32), cache, config.target_layers
            )
            mx.eval(hidden, [c.state for c in cache])
            features.append(hidden)
        prefix_features = mx.concatenate(features, axis=1)
        base = [c.snapshot() for c in cache]
        ordinary = []
        anchor = tokens[-1]
        eos = set(adapter.tokenizer.eos_token_ids)
        ordinary_times = []
        first_logits = first_state = None
        for _ in range(args.max_tokens):
            guard()
            tic = time.perf_counter()
            logits = model(
                mx.array([[anchor]], dtype=mx.uint32), cache, logits_to_keep=0
            )
            mx.eval(logits, [c.state for c in cache])
            if not bool(mx.all(mx.isfinite(logits)).item()):
                raise RuntimeError("ordinary target logits are nonfinite")
            ordinary_times.append(time.perf_counter() - tic)
            if not ordinary:
                first_logits = mx.asarray(logits, copy=True)
                mx.eval(first_logits)
                first_state = [c.snapshot() for c in cache]
            anchor = int(mx.argmax(logits[0, -1]).item())
            ordinary.append(anchor)
            if anchor in eos:
                break
        receipt["ordinary"] = {
            "tokens": ordinary,
            "text": adapter.tokenizer.decode(ordinary, skip_special_tokens=True),
            "forward_seconds": ordinary_times,
            "timing_scope": "observational; baseline load/prefill/compile order not controlled",
        }
        for c, s in zip(cache, base):
            c.restore(s)
        tx = DeepSeekV4VerifyRows(model, [cache], config.target_layers).begin([3])
        verify_input = [
            tokens[-1],
            ordinary[0],
            (ordinary[0] + 17) % model.args.vocab_size,
        ]
        verified, _ = tx.forward(mx.array([verify_input], dtype=mx.uint32))
        mx.eval(verified, [c.state for c in cache])
        first_equal = same_values(verified[:, :1], first_logits)
        tx.commit([1])
        mx.eval([c.state for c in cache])
        reject_equal = same_snapshots(cache, first_state)
        receipt["rejection_restore"] = {
            "verified_input": verify_input,
            "accepted_inputs": 1,
            "first_logits_bitwise_equal": first_equal,
            "all_state_planes_bitwise_equal": reject_equal,
            "law": "restore_pre_round_then_replay_accepted_prefix_only",
        }
        for c, s in zip(cache, base):
            c.restore(s)
        support = draft.make_cache()
        draft.append_context(prefix_features, support)
        guard()
        tic = time.perf_counter()
        proposed, laws = draft.draft_distributions(
            [tokens[-1]], prefix_features[:, :0], support, 5, [None], [0.0]
        )
        receipt["direct_support"] = {
            "tokens": proposed,
            "law_types": [[type(x).__name__ for x in row] for row in laws],
            "offset": support[0].offset,
            "retained_bytes": support[0].nbytes,
            "seconds": time.perf_counter() - tic,
            "capture_feature_shape": list(prefix_features.shape),
            "settings": draft.receipt_settings,
        }
        phase(
            "direct_support_and_rejection_complete",
            proposals=proposed,
            rejection=receipt["rejection_restore"],
        )
        calls = {
            "target_forward_with_taps": 0,
            "target_input_tokens": 0,
            "prefill_body_calls": 0,
        }
        original = model.forward_with_taps
        original_prefill = model.prefill_body

        def counted(inputs, *a, **kw):
            calls["target_forward_with_taps"] += 1
            calls["target_input_tokens"] += inputs.shape[1]
            return original(inputs, *a, **kw)

        def counted_prefill(*a, **kw):
            calls["prefill_body_calls"] += 1
            return original_prefill(*a, **kw)

        model.forward_with_taps = counted
        model.prefill_body = counted_prefill
        generator_options = {
            "num_draft": 5,
            "completion_batch_size": 1,
            "prefill_step_size": args.prefill_step,
            "cost_per_commit_handoff": False,
            "stop_tokens": tuple((int(t),) for t in eos),
        }
        generator = (
            adapter.create_external_batch(**generator_options)
            if args.adapter_external
            else DeepSeekV4ExternalBatchGenerator(
                model,
                draft_model=draft,
                binding=draft.execution_binding,
                target_artifact_fingerprint=artifact["identity"]["fingerprint"],
                **generator_options,
            )
        )
        generator.insert(
            [tokens],
            max_tokens=[args.max_tokens],
            sampling_configs=[{"sampling_temp": 0.0}],
        )
        external = []
        last = None
        tic = time.perf_counter()
        for round_id in range(args.max_tokens * 3 + 4):
            guard()
            _, responses = generator.next()
            for response in responses:
                external.append(int(response.token))
                last = response
                atomic(
                    args.output / "partial-external.json",
                    {
                        "tokens": external,
                        "round": round_id,
                        "last_receipt": getattr(response, "speculative_receipt", None),
                    },
                )
            if not generator.lanes:
                break
        if generator.lanes:
            raise RuntimeError("external generator exceeded bounded round count")
        receipt["external"] = {
            "tokens": external,
            "text": adapter.tokenizer.decode(external, skip_special_tokens=True),
            "exact_ordinary_tokens": external == ordinary,
            "seconds": time.perf_counter() - tic,
            "calls": calls,
            "final_receipt": None if last is None else last.speculative_receipt,
            "scheduler_stats": dict(generator.scheduler_stats),
        }
        expected_pass = (
            args.expected_text is None
            or receipt["ordinary"]["text"].strip() == args.expected_text
        )
        receipt["text_oracle"] = {
            "expected": args.expected_text,
            "tested": args.expected_text is not None,
            "passed": expected_pass,
        }
        receipt["passed"] = bool(
            first_equal
            and reject_equal
            and external == ordinary
            and last is not None
            and expected_pass
        )
        receipt["status"] = "completed"
        guard()
        phase("completed", passed=receipt["passed"])
    except BaseException as error:
        receipt.update(
            status="failed",
            passed=False,
            error={"type": type(error).__name__, "message": str(error)},
        )
        raise
    finally:
        generator = backbone = draft = cache = model = None
        if weights is not None and not args.adapter_external:
            try:
                weights.close()
            except Exception as error:  # noqa: BLE001 - preserve the original probe failure
                receipt["sidecar_cleanup_error"] = str(error)
        weights = None
        if adapter is not None:
            try:
                adapter.close()
            except Exception as error:  # noqa: BLE001 - preserve the original probe failure
                receipt["target_cleanup_error"] = str(error)
        adapter = None
        gc.collect()
        try:
            mx.clear_cache()
            receipt["cleanup_active_bytes"] = mx.get_active_memory()
        except (NameError, RuntimeError):
            pass
        try:
            from deepseek_v4_probe_residency import restore_probe_wired_limit
            restore_probe_wired_limit(mx, receipt)
        except Exception as error:
            receipt['residency_cleanup_error'] = str(error)
        if "cleanup_error" in receipt or any(
            k.endswith("_cleanup_error") for k in receipt
        ):
            receipt["passed"] = False
            receipt["status"] = "failed"
        atomic(args.output / "results.json", receipt)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
