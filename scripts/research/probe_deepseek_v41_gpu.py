#!/usr/bin/env python3
"""Bounded, owned V4.1 repair probes; no formal qualification/performance claim."""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from mlx2.desktop_campaign_worker import atomic, check_ownership


def native_resolution(path):
    from mlx2.adapters.registry import inspect_model

    resolution = inspect_model(path)
    if resolution.descriptor.model_type not in ("deepseek41_mlx", "deepseek41_iq1"):
        raise ValueError("native V4.1 probe requires a declared standalone V4.1 artifact")
    return resolution.adapter_type, resolution.artifact


def environment():
    from mlx2.apple_telemetry import TemperatureSampler

    sampler = TemperatureSampler()
    try:
        thermal = sampler.die_summary()
    finally:
        sampler.close()
    result = {"thermal": thermal, "time": time.time()}
    for name, command in {
        "swap": ["sysctl", "vm.swapusage"],
        "thermal_pressure": ["pmset", "-g", "therm"],
        "services": ["launchctl", "list"],
        "processes": ["ps", "-axo", "pid=,rss=,comm="],
    }.items():
        result[name] = subprocess.check_output(command, text=True)
    result["waiters"] = sorted(p.name for p in Path("/Users/Shared/mlxuag/gpu.lock.waiters").iterdir())
    return result


def stream_oracle(mx, out):
    import hashlib

    import numpy as np

    from mlx2.adapters.deepseek_v41_expert_stream import PackedExpertStream
    from mlx2.runtime.gguf.packed_metal import PackedMatrix
    from mlx2.runtime.gguf.reader import GGUFTensor
    from mlx2.runtime.gguf.v41_metal import moe, moe_rows
    from mlx2.runtime.models.deepseek_v41_engram import file_stamp

    rng = np.random.default_rng(4142)
    tensors, data, full = {}, bytearray(), []
    for kind in ("gate", "up", "down"):
        name = "blk.0.ffn_"+kind+"_exps.weight"
        raw = rng.integers(0, 256, (4*256, 50), np.uint8)
        raw[:, :2] = np.frombuffer(np.float16(0.0625).tobytes(), np.uint8)
        tensor = GGUFTensor(name, (256, 256, 4), 19, len(data), raw.nbytes)
        tensors[name] = tensor
        data.extend(raw.tobytes())
        full.append(PackedMatrix(mx.array(raw.reshape(-1)), tensor))
    path = out / "synthetic-experts.bin"
    path.write_bytes(data)
    stream = PackedExpertStream(path, tensors=tensors, data_offset=0, expected_stat=file_stamp(path.stat()),
                                ceiling_bytes=2*3*256*50, top_k=2)
    try:
        x, ids, scores = mx.full((1, 256), .0625), mx.array([[3, 1]], mx.uint32), mx.array([[.5, 1.5]])
        expected = moe(x, ids, scores, *full)
        remapped, selected = stream.gather("blk.0", ids)
        actual = moe(x, remapped, scores, *selected)
        row_ids, rows = stream.gather("blk.0", ids, row_buffers=True)
        direct = moe_rows(x, row_ids, scores, *rows)
        mx.eval(expected, actual, direct)
        exact = bool(np.array_equal(np.array(expected).view(np.uint32), np.array(actual).view(np.uint32)))
        direct_exact = bool(np.array_equal(np.array(expected).view(np.uint32), np.array(direct).view(np.uint32)))
        if not exact or not direct_exact:
            raise RuntimeError("native streamed expert remapping changed GPU output bits")
        return {"checkpoint": "synthetic-IQ1_S@"+hashlib.sha256(data).hexdigest(),
                "bitwise_equal": exact, "row_buffers_bitwise_equal": direct_exact, "source_and_cache": stream.receipt()}
    finally:
        stream.close()


def unit(spec):
    from probe_deepseek_v4_http import command, get_json, pair_check, stream_chat
    from probe_deepseek_v4_ordinary import memory, snapshot_check

    from mlx2.desktop_runner import digest, source_manifest

    out = Path(spec["out"])
    report = {"checkpoint": spec["checkpoint"], "source_revision": spec["source_revision"],
              "source_snapshot_sha256": spec["source_snapshot_sha256"],
              "thermal_controlled": spec.get("thermal_controlled", True),
              "qualification": False, "performance_claim": False, "passed": False, "phases": []}
    adapter = model = cache = logits = server = server_log = None
    compact_capture = compact_frames = None
    admin_directory = None
    admin_token = None
    stop = threading.Event()
    thread = None

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"owned GPU unit received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def phase(name, **values):
        report["phases"].append({"name": name, "time": time.time(), **values})
        atomic(out / "results.json", report)
        print(json.dumps({"phase": name, **values}, default=str), flush=True)

    def sample():
        while not stop.wait(5):
            with (out / "thermal-samples.jsonl").open("a") as stream:
                stream.write(json.dumps(environment()) + "\n")

    try:
        report["ownership"] = check_ownership(spec)
        if digest(source_manifest(ROOT)) != spec["source_snapshot_sha256"]:
            raise RuntimeError("source changed after GPU probe was frozen")
        report["before"] = environment()
        die = report["before"]["thermal"]["die_max_c"]
        if spec.get("thermal_controlled", True) and (die is None or die > 60):
            raise RuntimeError("initial thermal admission requires HID die temperature <=60C")
        if spec.get("thermal_controlled", True) and "CPU_Speed_Limit" in report["before"]["thermal_pressure"]:
            raise RuntimeError("host reports thermal throttling")
        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
        import mlx.core as mx

        if spec.get("native_mlx"):
            TargetAdapter, _ = native_resolution(spec["model"])
            inspect_artifact = TargetAdapter.artifact_inspector
        else:
            from mlx2.adapters.deepseek_v41 import DeepSeekV41Adapter as TargetAdapter
            from mlx2.adapters.deepseek_v41_artifact import inspect_artifact
        from mlx2.runtime.models.deepseek_v41_quant import fused_quantize, fused_status
        from mlx2.runtime.streamed_load import WeightStreamRequest
        from mlx2.serving import preload_admission_limit_bytes

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(112 << 30)
        mx.set_cache_limit(128 << 20)
        report["device"] = mx.device_info()
        artifact = inspect_artifact(spec["model"])
        if artifact["identity"]["fingerprint"] != spec["fingerprint"]:
            raise RuntimeError("artifact identity changed")
        # First-use gates are synthetic and distinct from actual model execution.
        for kind in ("fp8", "index_fp4", "compressed_fp4"):
            value = mx.array([[(i - 64) / 16 for i in range(128)]], mx.float32)
            mx.eval(fused_quantize(value, kind))
        report["synthetic_qat"] = fused_status()
        report["synthetic_stream_oracle"] = stream_oracle(mx, out)
        if spec.get("compact_iq1_ab"):
            from probe_deepseek_v41_iq1_ab import block_oracle
            report["compact_iq1_unpack_oracle"] = block_oracle(mx)
        phase("synthetic_qat_passed")
        if spec.get("ablate"):
            from mlx2.runtime.gguf.v41_row_exact import self_check as rows_check
            from mlx2.runtime.gguf.v41_dense import self_check as dense_check
            from mlx2.runtime.models.deepseek_v41_fused import self_check as pointwise_check
            from mlx2.runtime.models.deepseek_v41_fused import epilogue_check

            for reuse in (1,2,4):
                for broadcast in (False,True):
                    rows_check(reuse,broadcast)
                    phase("row_exact_gate",reuse=reuse,broadcast=broadcast)
            for groups,simds in ((10,1),(6,4),(10,4)):
                rows_check(4,False,groups,simds)
                phase("row_exact_geometry_gate",groups=groups,simds=simds)
            dense_check();phase("dense_fusion_gate")
            pointwise_check();phase("pointwise_gate")
            epilogue_check();phase("attention_hc_epilogue_gate")
        start = time.monotonic()
        streaming = WeightStreamRequest(mode="moe_experts", budget_mode="auto", budget_bytes=0,
                                       admission_limit_bytes=preload_admission_limit_bytes(), bypass_page_cache=True)
        direct_policy = {k: v for k, v in spec.get("policy", {}).items() if k.startswith("deepseek_")} if spec.get("profile") or spec.get("native_layer_ab") or spec.get("native_mlx") else None
        adapter = TargetAdapter(spec["model"], weight_streaming=streaming, execution_policy=direct_policy)
        model = adapter.model
        report["artifact"] = artifact["identity"]
        report["weight_residency"] = adapter.preload_residency
        report["weight_residency"]["active_bytes_after_load"] = mx.get_active_memory()
        phase("target_loaded", seconds=time.monotonic()-start, memory=memory(mx))
        request = {"messages": [{"role": "user", "content": "Reply with exactly this text, including the period: Hello."}], "enable_thinking": False}
        ids = list(adapter.prompt_tokens(request))
        report["request"] = request
        report["prompt_ids"] = ids
        if spec.get("compact_iq1_ab"):
            from probe_deepseek_v41_iq1_ab import capture
            compact_capture = capture(adapter.weights, mx)
            compact_frames = compact_capture.__enter__()
        cache = model.make_cache()
        logits = model.prefill_forward(mx.array([ids], mx.uint32), cache)
        mx.eval(logits, [state.state for state in cache])
        if not bool(mx.all(mx.isfinite(logits)).item()):
            raise RuntimeError("ordinary reference logits nonfinite")
        phase("ordinary_prefill", prompt_tokens=len(ids), memory=memory(mx))
        report["snapshot_restore"] = snapshot_check(model, cache, 42, mx)
        if not all(report["snapshot_restore"][k] for k in (
            "logits_bitwise_equal", "state_arrays_bitwise_equal", "state_metadata_equal", "restored_original_frontier"
        )):
            raise RuntimeError("ordinary snapshot/replay mismatch")
        tokens = []
        for _ in range(16):
            token = int(mx.argmax(logits[0, -1]).item())
            tokens.append(token)
            if token in adapter.tokenizer.eos_token_ids:
                break
            logits = model(mx.array([[token]], mx.uint32), cache)
            mx.eval(logits, [state.state for state in cache])
            if not bool(mx.all(mx.isfinite(logits)).item()):
                raise RuntimeError("ordinary decode logits nonfinite")
        text = adapter.tokenizer.decode(tokens, skip_special_tokens=True)
        report["ordinary_reference"] = {"tokens": tokens, "text": text, "finite": True}
        report["route"] = adapter.diagnostics()
        phase("ordinary_reference_complete", text=text, token_ids=tokens)
        if text.strip() != "Hello.":
            raise RuntimeError("ordinary functional oracle expected exactly Hello.")
        if spec.get("compact_iq1_ab"):
            from probe_deepseek_v41_iq1_ab import run as compact_run
            compact_capture.__exit__(None, None, None)
            compact_capture = None
            report["compact_iq1_ab"] = compact_run(adapter, spec["affine_reference"], compact_frames,
                                                   mx, out, phase, lambda: check_ownership(spec))
            compact_frames = None
        if spec.get("native_layer_ab"):
            from ab_deepseek_v41_native_layer import run as native_layer_run
            report["native_layer_ab"] = native_layer_run(adapter,mx,out,phase,guard=lambda:check_ownership(spec))
            phase("native_layer_ab_complete",passed=report["native_layer_ab"]["passed"])
        if spec.get("ten_token"):
            from analyze_deepseek_v41_10token import direct

            report["ten_token_reference"] = direct(adapter,mx,"reference")
            atomic(out/"ten-token-reference.json",report["ten_token_reference"])
            phase("ten_token_reference_complete")
        if spec.get("profile"):
            from profile_deepseek_v41 import profile_model

            report["profile"] = profile_model(adapter, cache, 42, mx)
            atomic(out / "profile.json", {"checkpoint": spec["checkpoint"], "source_revision": spec["source_revision"],
                                         "source_snapshot_sha256": spec["source_snapshot_sha256"], **report["profile"]})
            phase("profile_complete", shapes={k: report["profile"][k]["profiled_seconds"] for k in ("decode", "prefill")})
            if not all(report["profile"][k]["passed"] for k in ("decode", "prefill", "commits")):
                raise RuntimeError("profile candidate differs from its same-shape reference")
        # Release the direct model before starting one CLI loopback server.
        model = cache = logits = None
        adapter.close()
        adapter = None
        gc.collect()
        mx.clear_cache()
        mx.set_wired_limit(report["weight_residency"]["previous_wired_limit_bytes"])
        phase("direct_model_released", memory=memory(mx))
        if mx.get_active_memory() > 1 << 30:
            raise RuntimeError("direct probe retained model allocations before server")
        if spec.get("ten_token"):
            if spec.get("dense_cache_compare"):
                from page_deepseek_v41 import compare_dense_capacities
                report["dense_cache_compare"]=compare_dense_capacities(spec,mx,out,phase,streaming)
            if spec.get("model_paging"):
                from page_deepseek_v41 import direct as paging_direct
                report["paging_direct"] = paging_direct(spec,mx,out,phase,streaming)
            adapter=TargetAdapter(spec["model"],weight_streaming=streaming,
                                      execution_policy={k:v for k,v in spec["policy"].items() if k.startswith("deepseek_")})
            if spec.get("dense_read_ab"):
                from page_deepseek_v41 import compare_dense_reads
                report["dense_read_ab"]=compare_dense_reads(spec,adapter,out,phase)
            if spec.get("ablate"):
                from ablate_deepseek_v41 import run as run_ablations
                report["ablations"]=run_ablations(adapter,mx,out,phase)
                phase("ablations_complete")
            if spec.get("micro"):
                from micro_deepseek_v41 import run as run_micro
                report["micro"]=run_micro(adapter,mx,out,phase)
            if spec.get("compare_final"):
                from ablate_deepseek_v41 import compare_final
                report["combined_comparison"]=compare_final(adapter,mx,out,phase)
            report["ten_token_optimized"]=direct(adapter,mx,"optimized")
            atomic(out/"ten-token-optimized.json",report["ten_token_optimized"])
            adapter.close();adapter=None
            gc.collect();mx.clear_cache()
            mx.set_wired_limit(report["weight_residency"]["previous_wired_limit_bytes"])
            phase("ten_token_optimized_complete",memory=memory(mx))
            if spec.get("paging_attribution_only"):
                report["passed"] = bool(report["ten_token_optimized"]["passed"])
                report["http_scope"] = "not rerun; focused paging attribution after completed source-bound lifecycle unit"
                return 0 if report["passed"] else 1
        check_ownership(spec)
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", 8285)) == 0:
                raise RuntimeError("loopback port8285 already occupied")
        policy = out / "policy.json"
        atomic(policy, spec.get("policy", {}))
        external = "draft_model" in spec.get("policy", {})
        cmd = command(Path(spec["model"]), 8285, policy, out / "apcv2", external=external,
                      prefill_step=spec.get("prefill_step", 128), cache_bytes=512 << 20, persist=bool(spec.get("model_paging")))
        if spec.get("ladder") or spec.get("model_paging") or spec.get("single_cell"):
            import secrets
            import tempfile

            admin_directory = tempfile.TemporaryDirectory(prefix="mlx2-v41-ladder-admin-")
            admin_file = Path(admin_directory.name)/"token"
            admin_token = secrets.token_urlsafe(32)
            admin_file.write_text(admin_token)
            admin_file.chmod(0o600)
            if spec.get("ladder"):
                cmd[cmd.index("--max-context")+1] = "65536"
            cmd += ["--generation-event-timeout-seconds", "7200", "--admin-token-file", str(admin_file)]
        report["server_command"] = cmd
        server_log = (out / "server.log").open("w")
        server = subprocess.Popen(cmd, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 600
        while True:
            check_ownership(spec)
            if server.poll() is not None:
                raise RuntimeError(f"server exited before ready: {server.returncode}")
            try:
                status = get_json("http://127.0.0.1:8285", "/v1/status")
                if status.get("healthy") is True:
                    break
            except (OSError, ValueError) as error:
                report["last_startup_poll_error"] = str(error)
            if time.monotonic() >= deadline:
                raise TimeoutError("server startup")
            time.sleep(2)
        report["status_before"] = status
        phase("server_ready")
        model_id = get_json("http://127.0.0.1:8285", "/v1/models")["data"][0]["id"]
        if spec.get("single_cell"):
            from bench_deepseek_v41_1k import run as run_single_cell
            report["single_cell"]=run_single_cell(spec,model_id,guard=lambda:check_ownership(spec),
                                                  environment=environment,phase=phase,admin_token=admin_token)
            report["passed"]=report["single_cell"]["passed"]
        elif spec.get("ten_token"):
            from analyze_deepseek_v41_10token import http

            report["ten_token_http"]=http("http://127.0.0.1:8285",model_id,out,lambda:check_ownership(spec))
            report["passed"]=report["ten_token_http"]["passed"]
            if spec.get("model_paging"):
                from page_deepseek_v41 import http as paging_http
                report["paging_http"] = paging_http(spec,model_id,out,phase,admin_token,report["ten_token_http"]["arms"][-1])
                report["passed"] &= report["paging_direct"]["passed"] and report["paging_http"]["passed"]
        elif spec.get("ladder"):
            from deepseek_v41_context_ladder import run_ladder

            report["ladder"] = run_ladder(spec, model_id, guard=lambda: check_ownership(spec),
                                          environment=environment, phase=phase, admin_token=admin_token)
            report["passed"] = report["ladder"]["passed"]
        else:
            body = {**spec.get("http_request", request), "model": model_id, "max_tokens": 64, "temperature": 0,
                    "stream": True, "logprobs": True, "top_logprobs": 1}
            for arm in ("cold", "warm"):
                check_ownership(spec)
                report[arm] = stream_chat("http://127.0.0.1:8285", body, out / (arm+".sse"), timeout=600)
                receipt = report[arm]["receipt"]
                phase(arm+"_complete", text=report[arm]["text"], cached_tokens=receipt.get("cached_tokens"),
                      speculation=receipt.get("speculation"))
            report["APCv2"] = pair_check(report["cold"], report["warm"], external=external, law="exact", expected=spec.get("expected_text", "Hello."))
            report["passed"] = report["APCv2"]["passed"]
        report["status_after"] = get_json("http://127.0.0.1:8285", "/v1/status")
        if not report["passed"]:
            raise RuntimeError("context ladder contains failed or incomplete cells" if spec.get("ladder")
                               else "APCv2 exact functional pair failed")
    except BaseException as error:  # noqa: BLE001 - retain failure and always retire owned resources
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        phase("failed", error=report["error"])
    finally:
        if compact_capture is not None:
            compact_capture.__exit__(None, None, None)
        compact_capture = compact_frames = None
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        if server_log:
            server_log.close()
        if admin_directory is not None:
            admin_directory.cleanup()
        model = cache = logits = None
        if adapter:
            adapter.close()
        adapter = None
        gc.collect()
        if "mx" in locals():
            mx.clear_cache()
            if report.get("weight_residency"):
                mx.set_wired_limit(report["weight_residency"]["previous_wired_limit_bytes"])
            report["cleanup_memory"] = memory(mx)
        stop.set()
        if thread:
            thread.join(timeout=15)
        report["after"] = environment()
        report["finished_at"] = time.time()
        atomic(out / "results.json", report)
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("owned", "unit"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copy-chain", action="store_true")
    parser.add_argument("--optimized", action="store_true")
    parser.add_argument("--long-prefix", action="store_true")
    parser.add_argument("--expert-cache-gib", type=float)
    parser.add_argument("--ladder", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--ten-token", action="store_true")
    parser.add_argument("--ablate", action="store_true")
    parser.add_argument("--micro", action="store_true")
    parser.add_argument("--compare-final", action="store_true")
    parser.add_argument("--model-paging", action="store_true")
    parser.add_argument("--paging-attribution-only", action="store_true")
    parser.add_argument("--dense-read-ab", action="store_true")
    parser.add_argument("--dense-cache-compare", action="store_true")
    parser.add_argument("--dense-cache-mib", type=int, default=1024)
    parser.add_argument("--single-cell", action="store_true")
    parser.add_argument("--native-layer-ab", action="store_true")
    parser.add_argument("--native-layer-path", type=Path)
    parser.add_argument("--native-mlx-path", type=Path)
    parser.add_argument("--compact-iq1-affine-reference", type=Path)
    parser.add_argument("--tiled-arena", action="store_true")
    parser.add_argument("--tiled-completion", choices=("sync", "async"), default="async")
    parser.add_argument("--tiled-decode", choices=("row_exact", "column_f32"), default="row_exact")
    args = parser.parse_args()
    if args.mode == "unit":
        return unit(json.loads((args.output / "spec.json").read_text()))
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.native_mlx_path:
        TargetAdapter, artifact = native_resolution(args.native_mlx_path)
        inspect_artifact = TargetAdapter.artifact_inspector
    else:
        from mlx2.adapters.deepseek_v41_artifact import FILENAME, inspect_artifact
    from mlx2.desktop_campaign_lease import DesktopCPGLease
    from mlx2.desktop_campaign_process import spawn
    from mlx2.desktop_runner import digest, source_manifest

    model = str(args.native_mlx_path.resolve()) if args.native_mlx_path else str(Path.home() / 'mlx-models/DeepSeek-V4.1-Flash-1017-Maya-XS-Abliterated-GGUF')
    artifact = inspect_artifact(model)
    if args.optimized:
        if args.native_mlx_path:
            import importlib
            optimized_policy = importlib.import_module(TargetAdapter.__module__).optimized_policy
        else:
            from mlx2.adapters.deepseek_v41 import optimized_policy

        policy = optimized_policy(copy_chain=args.copy_chain)
    else:
        policy = {"draft_model": "prompt_lookup", "num_draft": 5} if args.copy_chain else {}
    if args.tiled_arena:
        if not args.native_mlx_path or TargetAdapter.descriptor.model_type != "deepseek41_iq1":
            raise ValueError("tiled arena requires the standalone compact IQ1 artifact")
        policy.update(deepseek_expert_layout="tiled64", deepseek_arena_completion=args.tiled_completion,
                      deepseek_tiled_decode=args.tiled_decode)
    if args.expert_cache_gib is not None:
        if not 2 <= args.expert_cache_gib <= 70:
            raise ValueError("expert cache probe ceiling must be2..70GiB")
        policy["moe_expert_streaming"] = {"enabled": True, "cache_gib": args.expert_cache_gib}
    if args.model_paging and args.paging_attribution_only:
        raise ValueError("choose lifecycle paging or focused paging attribution")
    if args.model_paging or args.paging_attribution_only:
        if not args.ten_token or not args.optimized or not 256 <= args.dense_cache_mib <= 16384:
            raise ValueError("model paging requires optimized ten-token unit and dense cache256..16384MiB")
        policy.update(deepseek_dense_storage="paged",deepseek_dense_cache_mib=args.dense_cache_mib,deepseek_residency="elastic")
    ladder = None
    if args.ladder:
        if not args.copy_chain or not args.optimized:
            raise ValueError("this ladder requires its explicit optimized copy-chain profile")
        from deepseek_v41_context_ladder import prepare_ladder

        ladder = prepare_ladder(model, args.output)
    files = source_manifest(ROOT)
    name = Path(model).name if args.native_mlx_path else FILENAME
    spec = {"model": model, "checkpoint": name+"@"+artifact["identity"]["fingerprint"],
            "fingerprint": artifact["identity"]["fingerprint"], "out": str(args.output.resolve()),
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_snapshot_sha256": digest(files), "job_id": args.output.name,
            "mcp_url": "http://127.0.0.1:8766/mcp", "policy": policy}
    if args.native_mlx_path:
        if any((args.native_layer_ab,args.profile,args.ten_token,args.ladder,args.model_paging,args.paging_attribution_only)):
            raise ValueError("standalone native MLX artifact requires its own ordinary/copy diagnostic")
        spec.update(native_mlx=True,thermal_controlled=False,prefill_step=512)
    if args.tiled_arena:
        spec.update(tiled_arena=True, tiled_completion=args.tiled_completion, tiled_decode=args.tiled_decode)
    if args.compact_iq1_affine_reference:
        if not args.native_mlx_path or not args.optimized or args.copy_chain or args.single_cell or args.long_prefix:
            raise ValueError("compact IQ1 A/B requires an optimized standalone ordinary unit")
        if TargetAdapter.descriptor.model_type != "deepseek41_iq1":
            raise ValueError("compact IQ1 A/B requires the compact artifact")
        from mlx2.adapters.deepseek_v41_mlx_artifact import inspect_artifact as inspect_affine
        reference_path = args.compact_iq1_affine_reference.resolve()
        reference = inspect_affine(reference_path)
        spec.update(compact_iq1_ab=True, affine_reference=str(reference_path),
                    affine_reference_fingerprint=reference["identity"]["fingerprint"])
    if args.profile:
        if args.ladder or not args.optimized:
            raise ValueError("profile requires optimized policy and a separate unit from the ladder")
        spec.update(profile=True, thermal_controlled=False, prefill_step=512)
    if args.ten_token:
        if args.profile or args.ladder or not args.optimized:
            raise ValueError("ten-token attribution requires its own optimized unit")
        spec.update(ten_token=True,thermal_controlled=False,prefill_step=512)
    if args.single_cell:
        if not args.optimized or args.copy_chain or args.ladder or args.ten_token or args.profile:
            raise ValueError("single1K cell requires optimized ordinary serving and its own unit")
        from bench_deepseek_v41_1k import prepare
        spec.update(single_cell=prepare(model,args.output),thermal_controlled=False,prefill_step=512,
                    repetitions=3,delay_seconds=10)
    if args.native_layer_ab:
        if not args.optimized or not args.native_layer_path or any((args.copy_chain,args.single_cell,args.ladder,args.ten_token,args.profile)):
            raise ValueError("native layer A/B requires its own optimized ordinary unit and converted artifact path")
        native_path=args.native_layer_path.resolve()
        conversion=json.loads((native_path/"conversion.json").read_text())
        if not conversion.get("converted") or conversion["checkpoint"]!=spec["checkpoint"]:
            raise ValueError("native layer artifact checkpoint mismatch")
        spec.update(native_layer_ab=True,native_layer_path=str(native_path),thermal_controlled=False,prefill_step=512)
    if args.ablate:
        if not args.ten_token:
            raise ValueError("ablations require the final ten-token instrumented unit")
        spec["ablate"]=True
    if args.micro:
        if not args.ten_token:raise ValueError("micro comparison requires ten-token unit")
        spec["micro"]=True
    if args.compare_final:
        if not args.ten_token:raise ValueError("final comparison requires ten-token unit")
        spec["compare_final"]=True
    if args.model_paging:
        spec["model_paging"]=True
    if args.paging_attribution_only:
        spec["paging_attribution_only"]=True
    if args.dense_read_ab:
        if not args.paging_attribution_only:
            raise ValueError("dense-read comparison requires focused paging attribution")
        spec["dense_read_ab"]=True
    if args.dense_cache_compare:
        if not args.paging_attribution_only:
            raise ValueError("dense capacity comparison requires focused paging attribution")
        spec["dense_cache_compare"]=True
    if ladder:
        spec.update(ladder=ladder, thermal_controlled=False, prefill_step=512, repetitions=1,
                    historical_reference="qualification/runs/deepseek-v41-context-ladder-20261024-r3/results.json",
                    historical_comparison="same artifact, requests and serving shape as prior 1K..64K ladder; new row_groups/local_frontier candidates; both single uncontrolled observations, no controlled speedup claim")
    if args.copy_chain:
        text = "APCv2 stores exact state. APCv2 stores exact state. APCv2 stores exact state."
        spec.update(expected_text=text, http_request={"messages": [{"role": "user", "content":
            "Copy the following text exactly. Return only that text, without quotation marks or explanation:\n"+text}], "enable_thinking": False})
    if args.long_prefix:
        text = spec.get("expected_text", "Hello.")
        notes = "Background notes; do not reproduce them:\n" + "".join(
            f"Reference entry {i}: prefix caching saves exact model state and token history.\n" for i in range(32))
        spec["http_request"] = {"messages": [{"role": "user", "content": notes+
            "\nReturn only the following text, including punctuation, without explanation:\n"+text}], "enable_thinking": False}
    atomic(args.output / "source.json", {"files": files, "source_revision": spec["source_revision"], "source_snapshot_sha256": digest(files)})
    atomic(args.output / "spec.json", spec)
    with DesktopCPGLease(args.output, spec["job_id"], checkpoint=spec["checkpoint"], source_revision=spec["source_revision"]) as lease:
        cmd = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"), "--session", lease.owner["cpg_session"],
               "--label", spec["job_id"], "--desktop-cpg-lease", str(lease.path), "--receipt", str(args.output / "paired-locks.json"),
               "--", sys.executable, "-u", str(Path(__file__).resolve()), "unit", "--output", str(args.output.resolve())]
        child = spawn(cmd, start_new_session=True)
        deadline = time.monotonic() + (14400 if ladder else 1800)
        try:
            while child.poll() is None:
                lease.check()
                if time.monotonic() >= deadline:
                    raise TimeoutError("bounded GPU ownership window")
                time.sleep(1)
            lease.success = child.returncode == 0
            return child.returncode
        finally:
            child.stop(timeout=35)


if __name__ == "__main__":
    raise SystemExit(main())
