#!/usr/bin/env python3
"""Bounded functional repair probes; never a qualification/performance gate."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def atomic(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temp.replace(path)


def source():
    h = hashlib.sha256()
    for p in sorted((ROOT / "src").rglob("*.py")):
        h.update(str(p.relative_to(ROOT)).encode()); h.update(p.read_bytes())
    return {"source_revision": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": h.hexdigest(), "source_root": str(ROOT),
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def environment():
    return {name: subprocess.run(command, capture_output=True, text=True, check=False).stdout.strip()
            for name, command in {"swap": ["sysctl", "vm.swapusage"], "thermals": ["pmset", "-g", "therm"],
                                  "vm": ["vm_stat"]}.items()}


def http(base, path, body=None):
    request = urllib.request.Request(base + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def unit(spec):
    from mlx2.desktop_campaign_worker import check_ownership
    check_ownership(spec)
    import mlx.core as mx
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(1 << 30)
    from mlx2.adapters.registry import inspect_model
    out = Path(spec["out"])
    report = {**source(), "checkpoint": spec["checkpoint"], "qualified": False,
              "scope": "targeted functional repair probes", "before": environment(), "passed": False}
    server = None
    try:
        resolution = inspect_model(spec["model"])
        if resolution.artifact_fingerprint != spec["fingerprint"]:
            raise ValueError("artifact changed after inventory")
        adapter = resolution.adapter_type(spec["model"])
        from mlx2.runtime.models import qwen_hc_prefill as hc
        from mlx2.runtime.models.gated_delta import install_prefill_scan
        model = adapter.model
        if spec.get("decode_tile") or spec.get("cold_tensorfold_law"):
            from mlx2.adapters.flash_next_tree import install_projection_law
            report["projection_law"] = install_projection_law(model)
        ids = list(adapter.tokenizer.encode("Explain why a cache key needs a revision. " * 50))[:65]
        def run(capture_position=None, model=model, tokenizer=adapter.tokenizer):
            cache = model.make_cache()
            logits = model(mx.array(ids)[None], cache=cache)
            mx.eval(logits)
            tokens = []
            window = None
            for position in range(8):
                token = int(mx.argmax(logits[:, -1, :]).item())
                if position == capture_position:
                    row = logits[0, -1].astype(mx.float32)
                    top = mx.argsort(row)[-10:][::-1]
                    values = row[top] - mx.logsumexp(row)
                    window = {"position_zero_based": position, "shared_prefix": list(tokens),
                              "top_ids": top.tolist(), "top_logprobs": values.tolist()}
                tokens.append(token)
                logits = model(mx.array([[token]]), cache=cache)
            mx.eval(logits)
            return tokens, tokenizer.decode(tokens), bool(mx.all(mx.isfinite(logits)).item()), window
        reference = run()
        report["ordinary_reference"] = {"tokens": reference[0], "text": reference[1], "finite": reference[2]}
        scan = install_prefill_scan(model, 32)
        if resolution.adapter_type.__name__ == "FlashNextAdapter":
            os.environ[hc.ENV] = "1"
        candidate = run()
        report["candidate"] = {"tokens": candidate[0], "text": candidate[1], "finite": candidate[2],
            "token_equal": reference[0] == candidate[0], "scan": scan, "hc": hc.status()}
        if spec.get("diagnose_prefill") and reference[0] != candidate[0]:
            position = next(i for i, pair in enumerate(zip(reference[0], candidate[0])) if pair[0] != pair[1])
            os.environ[hc.ENV] = "0"
            scan_only = run(capture_position=position)
            layers = [layer for _, layer in model.named_modules() if hasattr(layer, "_prefill_scan_chunk")]
            for layer in layers: object.__setattr__(layer, "_prefill_scan_chunk", 0)
            control = run(capture_position=position)
            os.environ[hc.ENV] = "1"
            hc_only = run(capture_position=position)
            for layer in layers: object.__setattr__(layer, "_prefill_scan_chunk", 32)
            both = run(capture_position=position)
            report["first_divergence"] = {"position_zero_based": position, "ordinary_repeat_equal": control[0] == reference[0],
                "arms": {name: {"tokens": value[0], "text": value[1], "finite": value[2], "window": value[3]}
                         for name, value in (("ordinary", control), ("scan_only", scan_only), ("hc_only", hc_only), ("both", both))},
                "classification": "unclassified_reordered_prefill_arithmetic"}
        if spec.get("tensorfold"):
            from mlx2.runtime.models.tensorfold_prefill import install
            try:
                packing = install(model, backend="native")
            except ValueError as error:
                if str(error) != "prefill candidate found no eligible dense q4/g64 projections":
                    raise
                report["tensorfold_composed"] = {"supported": False, "reason": str(error)}
                spec = {**spec, "tensorfold": False}
            else:
                composed = run()
                report["tensorfold_composed"] = {"supported": True, "packing": packing, "tokens": composed[0],
                    "text": composed[1], "finite": composed[2], "token_equal": reference[0] == composed[0]}
        # The new HC dimensions can be exercised synthetically even when this
        # target has no HC layers. This is explicitly separate from target use.
        cells = []
        os.environ[hc.ENV] = "1"
        for hidden in (2560, 5120):
            for dtype in (mx.bfloat16, mx.float16):
                x = mx.random.normal((1, 17, 4 * hidden)).astype(dtype)
                w = mx.ones((4 * hidden,), dtype=mx.float32)
                branch = (mx.random.normal((1, 17, hidden)) * .07).astype(dtype)
                inject = mx.ones((1, 17, 4), dtype=dtype)
                combined, normed = hc.try_combine_norm(x, branch, inject, w, hidden, 1e-6)
                ref_c = hc.eager_combine(x, branch, inject)
                ref_n = hc.eager_norm(ref_c, w, hidden, 1e-6)
                gate = mx.random.normal(x.shape).astype(dtype)
                mixed = hc.try_gate_mix(normed, gate, hidden)
                ref_m = mx.mean((mx.sigmoid(gate) * normed).reshape(1, 17, 4, hidden), axis=-2)
                mx.eval(combined, normed, mixed, ref_c, ref_n, ref_m)
                cells.append({"hidden": hidden, "dtype": str(dtype),
                    "combine_exact": bool(mx.array_equal(combined.view(mx.uint16), ref_c.view(mx.uint16)).item()),
                    "norm_max_abs": float(mx.max(mx.abs(normed.astype(mx.float32) - ref_n.astype(mx.float32))).item()),
                    "mix_max_abs": float(mx.max(mx.abs(mixed.astype(mx.float32) - ref_m.astype(mx.float32))).item())})
        report["synthetic_hc"] = cells
        report["direct_passed"] = reference[2] and candidate[2] and scan["counters"]["chunk32_calls"] > 0 and all(
            c["combine_exact"] and c["norm_max_abs"] <= .04 and c["mix_max_abs"] <= .04 for c in cells)
        if report.get("tensorfold_composed", {}).get("supported"):
            report["direct_passed"] &= report["tensorfold_composed"]["finite"]
        from mlx2.runtime.models import qwen4_qsa_indexed_merge as merge
        qsa_cells = []
        for partials in (1, 3, 17, 33):
            m = mx.random.normal((1, 2, 1, partials))
            l = mx.exp(mx.random.normal(m.shape))
            o = mx.random.normal((*m.shape, 256))
            os.environ[merge._PREFETCH_ENV_NAME] = "0"
            ordinary = merge._dispatch_candidate(m, l, o, output_dtype=mx.bfloat16, threads=128)
            os.environ[merge._PREFETCH_ENV_NAME] = "1"
            prefetched = merge._dispatch_candidate(m, l, o, output_dtype=mx.bfloat16, threads=128)
            mx.eval(ordinary, prefetched)
            qsa_cells.append({"partials": partials, "bit_exact": bool(mx.array_equal(
                ordinary.view(mx.uint16), prefetched.view(mx.uint16)).item())})
        report["synthetic_qsa_prefetch"] = qsa_cells
        report["direct_passed"] &= all(c["bit_exact"] for c in qsa_cells)
        if spec.get("synthetic_compact_moe"):
            from mlx2.runtime.models import moe_nax_gather as nax

            rows, experts, width, output_width = 65537, 512, 2560, 320
            x = (mx.random.normal((rows, 1, width)) * .01).astype(mx.bfloat16)
            weights = mx.full((experts, output_width, width // 8), 0x76543210, dtype=mx.uint32)
            scales = mx.full((experts, output_width, width // 64), .001, dtype=mx.bfloat16)
            biases = mx.full(scales.shape, -.007, dtype=mx.bfloat16)
            indices = (mx.arange(rows, dtype=mx.uint32) * experts) // rows
            old_mode = nax.MODE
            nax.set_mode("fused")
            os.environ["MLX_QWEN4_COMPACT_MOE_PREFILL"] = "1"
            try:
                plan = nax._plan(rows, experts, width, output_width)
                actual = nax.sorted_gather_qmm(x, weights, scales, biases, indices, group_size=64, bits=4)
                if actual is None: raise RuntimeError("compact MoE canary/admission declined")
                expected = mx.gather_qmm(x, weights, scales, biases, rhs_indices=indices,
                    transpose=True, group_size=64, bits=4, sorted_indices=True)
                mx.eval(actual, expected)
                error = float(mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))).item())
                report["synthetic_compact_moe"] = {"rows": rows, "experts": experts, "K": width, "N": output_width,
                    "plan": plan._asdict(), "max_abs": error, "passed": plan.bm == 64 and error <= .002}
                report["direct_passed"] &= report["synthetic_compact_moe"]["passed"]
            finally:
                nax.set_mode(old_mode)
                os.environ.pop("MLX_QWEN4_COMPACT_MOE_PREFILL", None)
            del x, weights, scales, biases, indices, actual, expected
        if spec.get("synthetic_gdn"):
            import numpy as np

            from mlx2.runtime.models.gdn_chunk32 import chunk32
            gdn_cells = []
            for vector, hk, hv in ((False, 1, 8), (True, 4, 4)):
                for dtype in (mx.bfloat16, mx.float16):
                    mx.random.seed(731)
                    q = (mx.random.normal((1, 33, hk, 128)) / 12).astype(dtype)
                    k = (mx.random.normal(q.shape) / 12).astype(dtype)
                    v = mx.random.normal((1, 33, hv, 128)).astype(dtype)
                    shape = (1, 33, hv, 128) if vector else (1, 33, hv)
                    g = mx.random.uniform(.5, .99, shape=shape)
                    g = mx.where(mx.arange(33)[None, :, None, None] == 16, 0, g) if vector else mx.where(mx.arange(33)[None, :, None] == 16, 0, g)
                    beta = mx.random.uniform(0, .8, shape=(1, 33, hv))
                    state = mx.random.normal((1, hv, 128, 128)) / 10
                    qn, kn, vn, gn, bn, carry = (np.array(a.astype(mx.float32)).astype(np.float64) for a in (q, k, v, g, beta, state))
                    expected = []
                    for t in range(33):
                        qt, kt = (np.repeat(a[:, t], hv // hk, axis=1) for a in (qn, kn))
                        carry *= gn[:, t, :, None, :] if vector else gn[:, t, :, None, None]
                        delta = (vn[:, t] - (carry * kt[:, :, None, :]).sum(-1)) * bn[:, t, :, None]
                        carry += delta[..., None] * kt[:, :, None, :]
                        expected.append((carry * qt[:, :, None, :]).sum(-1))
                    output, final = chunk32(q, k, v, g, beta, state)
                    rounded = mx.array(np.stack(expected, axis=1).astype(np.float32)).astype(dtype)
                    mx.eval(output, final, rounded)
                    output_error = float(mx.max(mx.abs(output.astype(mx.float32) - rounded.astype(mx.float32))).item())
                    state_error = float(np.max(np.abs(np.array(final) - carry)))
                    gdn_cells.append({"vector_gate": vector, "qk_heads": hk, "value_heads": hv,
                        "dtype": str(dtype), "output_max_abs": output_error, "state_max_abs": state_error,
                        "passed": output_error <= .004 and state_error <= 5e-6})
            report["synthetic_gdn_kda"] = gdn_cells
            report["direct_passed"] &= all(c["passed"] for c in gdn_cells)
        if spec.get("decode_tile"):
            from mlx2.runtime.cow_cache import snapshot_prompt_cache_descriptors
            cache = model.make_cache()
            mx.eval(model(mx.array(ids)[None], cache=cache))
            trunk = model.language_model.model
            layer = trunk.layers[0]
            tile_cells = []
            for width in (2, 4, 8, 16):
                tokens = mx.arange(11, 11 + width, dtype=mx.int32)[:, None]
                hidden = mx.tile(trunk.embed_tokens(tokens), (1, 1, trunk.args.hc_count))
                owners = [snapshot_prompt_cache_descriptors([cache[0]], require_committed=False)[0][0] for _ in range(width)]
                reference_rows = []
                expected_caches = []
                for row in range(width):
                    current = snapshot_prompt_cache_descriptors([owners[row]], require_committed=False)[0][0]
                    reference_rows.append(layer(hidden[row:row+1], tokens[row:row+1], cache=current))
                    expected_caches.append(current)
                output, tx = adapter.prepare_tensorfold_decode_block(0, hidden, owners, tokens)
                expected = mx.concatenate(reference_rows, axis=0)
                mx.eval(output, expected)
                cache_exact = all(bool(mx.array_equal(a.view(mx.uint16 if a.dtype.size == 2 else mx.uint32),
                    b.view(mx.uint16 if b.dtype.size == 2 else mx.uint32)).item())
                    for actual, control in zip(tx.successors, expected_caches)
                    for a, b in zip(actual.cache, control.cache) if a is not None and b is not None)
                tile_cells.append({"width": width, "metrics": tx.metrics,
                    "hidden_exact": bool(mx.array_equal(output.view(mx.uint16), expected.view(mx.uint16)).item()),
                    "hidden_max_abs": float(mx.max(mx.abs(output.astype(mx.float32)-expected.astype(mx.float32))).item()),
                    "cache_exact": cache_exact})
                tx.abort()
            report["tensorfold_decode_tiles"] = tile_cells
            report["direct_passed"] &= all(c["hidden_exact"] and c["cache_exact"] for c in tile_cells)
        del run, adapter, model, resolution
        gc.collect(); mx.clear_cache()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
        policy = {"gdn_prefill_chunk": 32}
        if spec["adapter"] == "FlashNextAdapter":
            policy["hc_prefill_kernels"] = True
        if spec.get("tensorfold"):
            policy.update(tensorfold_prefill=True, tensorfold_prefill_backend="native")
        atomic(out / "policy.json", policy)
        command = [sys.executable, "-m", "mlx2.server", "--model", spec["model"],
            "--host", "127.0.0.1", "--port", str(port), "--fabric", "off", "--max-lanes", "1",
            "--max-inflight", "1", "--max-context", "4096", "--prefill-step", "128", "--cache-bytes", str(256 << 20),
            "--cache-dir", str(out / "apcv2"), "--execution-policy", str(out / "policy.json"), "--qualification-mode"]
        with (out / "server.log").open("w") as log:
            server_env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True, env=server_env)
            base = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 600
            while True:
                if server.poll() is not None:
                    raise RuntimeError("server exited; see server.log")
                try:
                    status = http(base, "/v1/status")
                    if status.get("error"):
                        raise RuntimeError(status["error"])
                    if status.get("healthy"):
                        report["status_before"] = status
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError("server startup")
                    time.sleep(.5)
                except Exception:  # noqa: BLE001 -- startup retries are bounded and server errors are retained
                    if time.monotonic() > deadline: raise TimeoutError("server startup")
                    time.sleep(.5)
            model_id = http(base, "/v1/models")["data"][0]["id"]
            request = {"model": model_id, "prompt": "A cache key includes a model revision and token prefix. " * 24 + "\nExplain why:", "max_tokens": 16, "temperature": 0}
            report["cold"] = http(base, "/v1/completions", request)
            report["warm"] = http(base, "/v1/completions", request)
            report["status_after"] = http(base, "/v1/status")
        report["cold_warm_text_equal"] = (report["cold"]["choices"][0]["text"] == report["warm"]["choices"][0]["text"])
        report["cached_tokens"] = report["warm"].get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
        report["passed"] = (report["direct_passed"] and report["cold_warm_text_equal"] and report["cached_tokens"] > 0)
    except BaseException as error:  # noqa: BLE001 -- always seal failure and stop the server group
        report["error"] = str(error); report["traceback"] = traceback.format_exc()
    finally:
        if server and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try: server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL); server.wait()
        report["after"] = environment()
        atomic(out / "result.json", report)
    print(json.dumps({"checkpoint": spec["checkpoint"], "passed": report["passed"], "error": report.get("error")}))
    return 0 if report["passed"] else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("owned", "unit")); ap.add_argument("spec", type=Path)
    args = ap.parse_args(); spec = json.loads(args.spec.read_text())
    if args.mode == "unit": return unit(spec)
    from mlx2.desktop_campaign_lease import DesktopCPGLease
    out = Path(spec["out"])
    if out.exists() and any(out.iterdir()): raise ValueError("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    with DesktopCPGLease(out, spec["job_id"], checkpoint=spec["checkpoint"], source_revision=spec["source_revision"]) as lease:
        cmd = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"), "--session", lease.owner["cpg_session"],
            "--label", spec["job_id"], "--desktop-cpg-lease", str(lease.path), "--receipt", str(out / "paired-locks.json"),
            "--", sys.executable, str(Path(__file__)), "unit", str(args.spec.resolve())]
        child = subprocess.Popen(cmd, start_new_session=True)
        deadline = time.monotonic() + 1800
        try:
            while child.poll() is None:
                lease.check()
                if time.monotonic() > deadline: raise TimeoutError("bounded artifact window")
                time.sleep(1)
            lease.success = child.returncode == 0
            return child.returncode
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=35)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
