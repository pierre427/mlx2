"""One-artifact, owned B1 HTTP diagnostic, with an offline tree gate first.

No qualification/default promotion. One cold/warm pair per arm is directional
end-to-end evidence only. Raw requests/responses stay in the external output dir.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.request


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_route(body, arm):
    speculation = body.get("mlx2", {}).get("speculation") or {}
    tree = speculation.get("ast_tree_tiles") or {}
    if arm == "ast":
        if not tree.get("observed_used") or not tree.get("target_rows"):
            raise RuntimeError("HTTP request did not engage the admitted packed AST target")
    elif tree.get("observed_used"):
        raise RuntimeError("ordinary control engaged AST")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "out", "owner", "model"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source", "label", "fingerprint"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--pattern", choices=("3x5", "5x3"), required=True)
    parser.add_argument("--target-stage-trace-only", action="store_true")
    args = parser.parse_args()
    pattern = (5, 5, 5) if args.pattern == "3x5" else (3, 3, 3, 3, 3)

    def source_check():
        if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.root, text=True).strip() != args.source:
            raise RuntimeError("source HEAD changed")
        if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=args.root, text=True).strip():
            raise RuntimeError("tracked source dirty")

    source_check()
    if args.out.exists():
        raise RuntimeError("refusing an existing evidence directory")
    args.out.mkdir(parents=True)
    owned = load("packed_ast_owned", args.root / "scripts/run_single_model_smoke_ladder.py")
    thermal = load("packed_ast_thermal", args.root / "benchmark_results/2026-09-29/scripts/series/thermal_ladder.py")
    ownership = owned.validate_campaign_ownership(
        "drive-mlx2-ast-integration", args.label,
        cpg_task="beff4e634ddb46978a84a1aed315f785", cpg_owner_lock=args.owner,
        cpg_generation=args.generation,
    )
    report = {"schema": "mlx2.flash-next-packed-ast-http-diagnostic.v1",
              "source_revision": args.source, "checkpoint": args.model.name + "@sha256:" + args.fingerprint,
              "pattern": pattern, "owner": ownership, "status": "running",
              "qualified": False, "defaulted": False, "started_at": time.time(), "arms": {}}

    def save():
        temporary = args.out / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, default=str) + "\n")
        temporary.replace(args.out / "report.json")

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(str(signum))

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    before_swap = thermal.swapouts()
    own = thermal.ancestors(os.getpid()) | thermal.process_tree({os.getpid()})
    report["foreign_before"] = thermal.foreign_snapshot(own)
    report["thermal_before"] = owned.thermal_sample()

    def safety():
        if thermal.swapouts() != before_swap:
            raise RuntimeError("new swapout; stop memory-safe diagnostic")
        if owned.thermal_sample()["thermal_state"] >= 2:
            raise RuntimeError("thermal gate")

    adapter = engine = server = None
    actual_close = None
    try:
        safety()
        from mlx2.adapters.registry import inspect_model
        if inspect_model(args.model).artifact["fingerprint"] != args.fingerprint:
            raise RuntimeError("checkpoint fingerprint changed")
        import mlx.core as mx
        mx.set_memory_limit(80 << 30)
        mx.set_cache_limit(256 << 20)
        from mlx2.adapters.flash_next import FlashNextAdapter
        oracle_path = args.out / "oracle.json"
        policy = {"draft_model": str(args.model), "num_draft": max(pattern),
                  "row_exact_verify": False, "progressive_verification_tile": 2,
                  "progressive_alternative_branches": len(pattern),
                  "progressive_alternative_strategy": "confidence_tree", "cost_per_commit_handoff": False,
                  "continuation_pool": {"sources": ["external"], "limit": len(pattern),
                                        "ast_planner": True, "packed_pattern": list(pattern)},
                  "ast_target_tree_oracle": str(oracle_path)}
        adapter = FlashNextAdapter(str(args.model), execution_policy=policy)
        actual_close = adapter.close
        adapter.close = lambda: None  # The harness owns the one resident artifact across both arms.
        if not adapter._tables or os.environ.get("MLX_QWEN4_PLE_NVME") != str(args.model / "ple_rows.bin"):
            raise RuntimeError("PLE offload is not active")
        report["memory"] = {"ple_offload": True, "sidecar_bytes": (args.model / "ple_rows.bin").stat().st_size,
                            "allocator_limit": 80 << 30, "cache_limit": 256 << 20}
        safety()
        import numpy as np
        from mlx.utils import tree_flatten
        from mlx2.adapters.flash_next_tree import FlashNextTargetTreeExecutor, _clone
        from mlx2.adapters.ast_execution_profile import build_execution_profile
        from mlx2.runtime.models.qwen4_tree_gdn import ancestry

        def settle(*items):
            mx.eval(*[item for _, item in tree_flatten(items) if isinstance(item, mx.array)])
            mx.synchronize()

        def equal(lhs, rhs):
            # NumPy cannot import MLX bf16 directly. Widening is lossless and
            # retains a bit-exact comparison of each represented bf16 value.
            return np.array_equal(np.asarray(lhs.astype(mx.float32)),
                                  np.asarray(rhs.astype(mx.float32)))

        def digest(cache):
            value = [(item.state, item.meta_state) for item in cache]
            settle(value)
            result = hashlib.sha256()
            for name, item in tree_flatten(value):
                result.update(name.encode())
                if isinstance(item, mx.array):
                    result.update(str((item.shape, item.dtype)).encode())
                    result.update(np.asarray(item.astype(mx.float32)).tobytes())
                else:
                    result.update(repr(item).encode())
            return result.hexdigest()

        if args.target_stage_trace_only:
            from profile_flash_next_tree import trace_target

            report["scope"] = "Direct target synchronized stage attribution only; no serving or performance qualification."
            report["target_stage_trace"] = trace_target(adapter, pattern, settle, equal, safety)
            report["memory"].update(active_bytes=mx.get_active_memory(), peak_bytes=mx.get_peak_memory())
            source_check()
            report["status"] = "passed"
            return

        def offline_gate():
            # Runs only during cold engine construction, before ready/HTTP requests.
            model, draft = adapter.model, adapter.draft_model.backend
            layers = draft.config.target_layer_ids
            prompt = list(adapter.prompt_tokens({"messages": [{"role": "user", "content": "What is 2 plus 3?"}], "enable_thinking": False}))
            base = model.make_cache()
            seeds = model.prefill_body(mx.array([prompt]), base, layers)
            anchor = int(mx.argmax(model(mx.array([prompt]), cache=model.make_cache())[0, -1]).item())
            slate = draft.draft_packed_paths([anchor], seeds, draft.make_cache(), pattern,
                                             context_tokens=[prompt[1:] + [anchor]])[0]
            tokens, parents = [anchor], [-1]
            for path in slate:
                parent = 0
                for token in path.tokens:
                    parents.append(parent)
                    parent = len(tokens)
                    tokens.append(token)
            if len(tokens) != 16:
                raise RuntimeError("source did not produce the full16-row tile")
            provider = FlashNextTargetTreeExecutor(model)
            initial = digest(base)
            logits, features, transaction = provider.forward(model, tokens, parents, base, layers)
            node_states = []
            checks = {key: True for key in ("logits", "features", "prepared_state", "continuation", "isolation", "commit_abort")}
            try:
                settle(logits, features)
                paths = ancestry(parents)
                for row, parent in enumerate(parents):
                    control = [_clone(item) for item in (base if parent < 0 else node_states[parent])]
                    expected, taps = model.forward_with_taps(mx.array([[tokens[row]]]), control, layers)
                    settle(expected, taps)
                    node_states.append(control)
                    checks["logits"] &= equal(logits[:, row:row+1], expected)
                    checks["features"] &= equal(features[:, row:row+1], taps)
                    checks["prepared_state"] &= digest(transaction.prepared_cache(paths[row])) == digest(control)
                for path in ((), (0,), paths[-1][:3], paths[-1]):
                    successor = transaction.prepared_cache(path)
                    control = [_clone(item) for item in (base if not path else node_states[path[-1]])]
                    lhs, lf = model.forward_with_taps(mx.array([[7]]), successor, layers)
                    rhs, rf = model.forward_with_taps(mx.array([[7]]), control, layers)
                    settle(lhs, lf, rhs, rf)
                    checks["continuation"] &= (equal(lhs, rhs)
                                                 and equal(lf, rf)
                                                 and digest(successor) == digest(control))
                checks["isolation"] &= digest(base) == initial
                checks["commit_abort"] &= digest(transaction.commit_paths([paths[-1]])[0]) == digest(node_states[-1])
            finally:
                if not transaction.closed:
                    transaction.abort()
                node_states.clear()
            checks["commit_abort"] &= digest(base) == initial
            if not all(checks.values()):
                report["direct_checks"] = checks
                raise RuntimeError("offline exact numerical gate declined")
            profile = build_execution_profile(model, adapter.draft_model, binding=adapter.identity["fingerprint"],
                                               tile_cap=2, max_lanes=4, num_draft=max(pattern))
            binding = provider.oracle_binding(artifact=adapter.identity["fingerprint"], capture_layers=layers,
                                               target_binding=adapter._progressive_external_binding, profile=profile)
            direct = {"checks": checks, "metrics": transaction.metrics, "binding": binding}
            direct_path = args.out / "direct.json"
            direct_path.write_text(json.dumps(direct, indent=2) + "\n")
            doc = {"schema": "mlx2.flash-next-tree-oracle.v1", "binding": binding,
                   "input_widths": [16], "checks": checks,
                   "receipt_sha256": hashlib.sha256(direct_path.read_bytes()).hexdigest()}
            oracle_path.write_text(json.dumps(doc, indent=2) + "\n")
            report["direct_checks"] = checks
            report["direct_metrics"] = transaction.metrics
            report["oracle_sha256"] = hashlib.sha256(oracle_path.read_bytes()).hexdigest()
            gc.collect()
            mx.clear_cache()
            safety()
            save()

        from mlx2.serving import ServingEngine
        from mlx2.server import BoundedHTTPServer, handler_for
        draft, external_policy = adapter.draft_model, adapter.external_policy
        request = {"model": "mlx2", "messages": [{"role": "user", "content": "Explain how a database index speeds up queries. Use one short paragraph."}],
                   "temperature": 0, "max_tokens": 64, "stream": False, "enable_thinking": False}
        for arm in ("ordinary", "ast"):
            safety()
            if arm == "ordinary":
                adapter.draft_model, adapter.external_policy = None, {}
            else:
                adapter.draft_model, adapter.external_policy = draft, external_policy
                bind = adapter.bind_external_serving_namespace
                def bind_and_gate(namespace):
                    result = bind(namespace)
                    offline_gate()
                    return result
                adapter.bind_external_serving_namespace = bind_and_gate
            engine = ServingEngine(str(args.model), adapter_factory=lambda _path: adapter,
                mtp=False, max_lanes=1, max_inflight=1, max_context=2048,
                cache_bytes=2 << 30, qualification_mode=True, prefill_step=128,
                lane_matmul="off", int8_prefill="off")
            if not engine.ready.wait(180) or engine.error:
                raise RuntimeError(str(engine.error or "serving readiness timeout"))
            request["model"] = engine.status()["model"]
            server = BoundedHTTPServer(("127.0.0.1", 0), handler_for(engine))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
            rows = []
            report["arms"][arm] = rows
            for phase in ("cold", "warm"):
                safety()
                started = time.perf_counter()
                with urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(request).encode(),
                    headers={"Content-Type": "application/json"}), timeout=180) as response:
                    body = json.load(response)
                elapsed = time.perf_counter() - started
                count = body["usage"]["completion_tokens"]
                require_route(body, arm)
                rows.append({"phase": phase, "seconds": elapsed, "emitted_tokens": count,
                             "emitted_tokens_per_wall_second": count / elapsed, "response": body})
                safety()
                save()
            report.setdefault("status_snapshots", {})[arm] = engine.status()
            server.shutdown()
            server.server_close()
            thread.join(timeout=10)
            server = None
            engine.close()
            engine = None
            gc.collect()
            mx.clear_cache()
        texts = [row["response"]["choices"][0]["message"].get("content")
                 for rows in report["arms"].values() for row in rows]
        if len(set(texts)) != 1:
            raise RuntimeError("ordinary/AST cold/warm text mismatch; preserve for first-divergence diagnosis")
        report["http_text_exact"] = True
        report["memory"].update(active_bytes=mx.get_active_memory(), peak_bytes=mx.get_peak_memory())
        source_check()
        safety()
        report["status"] = "passed"
    except BaseException as exc:
        report["status"], report["error"] = "failed", repr(exc)
        raise
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if engine is not None:
            engine.close()
        if actual_close is not None:
            actual_close()
        gc.collect()
        report["resources_closed"] = True
        report["swapouts"] = {"before": before_swap, "after": thermal.swapouts()}
        report["thermal_after"] = owned.thermal_sample()
        report["foreign_after"] = thermal.foreign_snapshot(own)
        save()


if __name__ == "__main__":
    main()
