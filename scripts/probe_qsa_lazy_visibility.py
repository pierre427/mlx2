"""Bounded GPU correctness/memory smoke; no preflight or qualification.

Run owned SPEC to queue through CPG/radio and both host GPU locks. The unit
checks ownership before importing MLX, records the exact artifact and working
source, and always retires its loopback server process group.
"""

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
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def server_env():
    """Bind every Python child to this checkout before ambient imports."""
    source = str(ROOT / "src")
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        [source] + [part for part in existing.split(os.pathsep)
                    if part and part != source]
    )
    return environment


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")
    tmp.replace(path)


def host():
    commands = {
        "swap": ["sysctl", "vm.swapusage"],
        "vm": ["vm_stat"],
        "processes": ["ps", "-axo", "pid,ppid,etime,command"],
        "services": ["launchctl", "list"],
    }
    return {
        name: subprocess.run(cmd, capture_output=True, text=True, check=False).stdout
        for name, cmd in commands.items()
    }


def thermal():
    from run_qualification_matrix import sample_thermal

    sample = sample_thermal()
    required = (
        "pmset_no_thermal_warning",
        "pmset_no_performance_warning",
        "pmset_no_cpu_power_warning",
    )
    if sample.get("thermal_state") not in (0, 1) or not all(
        sample.get(key) for key in required
    ):
        error = RuntimeError("thermal admission declined; release GPU and retry later")
        error.sample = sample
        raise error
    return sample


def http(base, path, body=None):
    request = urllib.request.Request(
        base + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def stop(process):
    if process is None:
        return
    # Stop the whole group even if its leader has already exited.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def source():
    files = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__)]
    hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    return {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "root": str(ROOT),
        "working_source_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
        "files": hashes,
    }


def components(mx, spec):
    import numpy as np
    from check_qwen4_attn_rows import load_layer

    from mlx2.runtime.models import qwen4_attn_rows as R
    from mlx2.runtime.models import qwen4_exp as Q
    from mlx2.runtime.models.base import create_causal_mask
    from mlx2.runtime.models.qwen4_qsa_visibility import QSAVisibility, status

    def selection(total, length, pads):
        b, bs = len(pads), 4
        padding = mx.array(pads, dtype=mx.int32)
        qp = total - length - padding[:, None] + mx.arange(length)[None]
        logical = mx.arange(total)[None] - padding[:, None]
        blocks = total // bs
        picked = min(blocks, 512)
        ids = mx.broadcast_to(
            mx.arange(blocks - picked, blocks, dtype=mx.uint32)[None, None],
            (b, length, picked),
        )
        valid = ((mx.arange(blocks) + 1) * bs - 1)[None, None] <= qp[..., None]
        proof = QSAVisibility(length, total, b, padding)
        return Q.QSASelection(
            kind="explicit",
            batch=b,
            length=length,
            block_size=bs,
            raw_block_ids=ids,
            valid_blocks=valid,
            q_positions=qp,
            token_positions=logical,
            left_padding=padding,
            offset=qp[:, 0],
            physical_width=total,
            n_blocks=blocks,
            scatter_chosen=True,
            causal_mask=proof,
        )

    correctness = []
    for total, length, pads in (
        (10, 3, [0]),
        (65, 9, [0]),
        (131, 17, [0, 1, 5]),
        (4099, 3, [0]),
    ):
        sel = selection(total, length, pads)
        reference = replace(sel, causal_mask=sel.causal_mask.materialize()).dense_mask()
        if R.qsa_mask_supported(sel) is not None:
            raise RuntimeError("native QSA mask refused candidate geometry")
        candidate = R.qsa_mask(sel)
        mx.eval(reference, candidate)
        exact = bool(mx.array_equal(reference, candidate).item())
        correctness.append(
            {
                "total": total,
                "length": length,
                "padding": pads,
                "native_bit_exact": exact,
            }
        )
        if not exact:
            raise AssertionError("native mask visibility mismatch")
    del sel, reference, candidate
    gc.collect()
    mx.clear_cache()

    memory = []
    for arm in ("reference", "candidate", "candidate", "reference"):
        admitted = thermal()
        sel = selection(65539, 1024, [0])
        mx.eval(
            sel.raw_block_ids,
            sel.valid_blocks,
            sel.q_positions,
            sel.token_positions,
            sel.left_padding,
        )
        gc.collect()
        mx.clear_cache()
        start = mx.get_active_memory()
        mx.reset_peak_memory()
        if arm == "reference":
            mask = create_causal_mask(
                sel.length,
                offset=sel.physical_width - sel.length,
                left_padding=sel.left_padding,
            )
            sel = replace(sel, causal_mask=mask)
        result = R.qsa_mask(sel)
        mx.eval(result)
        memory.append(
            {
                "arm": arm,
                "base_active_bytes": start,
                "peak_bytes": mx.get_peak_memory(),
                "active_bytes": mx.get_active_memory(),
                "output_bytes": result.nbytes,
                "dense_input_bytes": 1024 * 65539,
                "shape": [1, 1, 1024, 65539],
                "thermal": admitted,
            }
        )
        del result, sel
        if arm == "reference":
            del mask
        gc.collect()
        mx.clear_cache()
        atomic(
            Path(spec["out"]) / "components.json",
            {
                "checkpoint": spec["checkpoint"],
                "native_masks": correctness,
                "memory_abba": memory,
            },
        )

    args, attention = load_layer(Path(spec["model"]), 3)
    caches = [Q.QSAKVCache(attention.indexer.summary_identity) for _ in range(2)]
    rng = np.random.default_rng(30965)
    R.set_enabled(True)
    layers = []
    for width in (4093, 3, 1, 9):
        x = mx.array(
            rng.standard_normal((1, width, args.hidden_size)).astype(np.float32)
        ).astype(mx.bfloat16)
        results = []
        for enabled, cache in zip((False, True), caches):
            Q._QSA_LAZY_CAUSAL_MASK = enabled
            out = attention(x, Q.qsa_attention_mask(x, cache), cache)
            mx.eval(out, cache.state)
            results.append(out)
        exact = bool(
            mx.array_equal(
                results[0].view(mx.uint16), results[1].view(mx.uint16)
            ).item()
        )
        state = all(
            bool(
                mx.array_equal(getattr(caches[0], key), getattr(caches[1], key)).item()
            )
            for key in ("keys", "values", "index_keys")
        )
        layers.append(
            {
                "layer": 3,
                "query_width": width,
                "offset_after": caches[0].offset,
                "output_bit_exact": exact,
                "cache_exact": state,
            }
        )
        if not exact or not state:
            raise AssertionError("real attention layer output/cache mismatch")
    Q._QSA_LAZY_CAUSAL_MASK = False
    return {
        "native_masks": correctness,
        "memory_abba": memory,
        "real_layer": layers,
        "counters": status(),
    }


def direct(mx, spec):
    from mlx.utils import tree_flatten

    from mlx2.adapters.flash_next import FlashNextAdapter
    from mlx2.runtime.models import qwen4_exp as Q
    from mlx2.runtime.models.qwen4_qsa_visibility import status

    adapter = FlashNextAdapter(spec["model"])
    prompt = (
        "A revision identifies the exact weights used to interpret cached tensors. "
        * 400
        + "\nIn one sentence, explain why a prefix cache key needs a model revision.\nAnswer:"
    )
    ids = list(adapter.tokenizer.encode(prompt))[:4093]
    results, states = [], []
    for enabled in (False, True):
        Q._QSA_LAZY_CAUSAL_MASK = enabled
        status(reset=True)
        cache = adapter.model.make_cache()
        for start in range(0, len(ids), 1024):
            logits = adapter.model(
                mx.array(ids[start : start + 1024])[None], cache=cache
            )
            mx.eval(logits)
        tokens = []
        for _ in range(12):
            if not bool(mx.all(mx.isfinite(logits)).item()):
                raise AssertionError("nonfinite ordinary output")
            token = int(mx.argmax(logits[0, -1]).item())
            tokens.append(token)
            logits = adapter.model(mx.array([[token]]), cache=cache)
            mx.eval(logits)
        results.append(
            {
                "arm": "candidate" if enabled else "reference",
                "tokens": tokens,
                "text": adapter.tokenizer.decode(tokens),
                "counters": status(),
            }
        )
        states.append(cache)
    first = tree_flatten([c.state for c in states[0]])
    second = tree_flatten([c.state for c in states[1]])
    exact_state = len(first) == len(second) and all(
        k == j
        and (bool(mx.array_equal(a, b).item()) if isinstance(a, mx.array) else a == b)
        for (k, a), (j, b) in zip(first, second)
    )
    exact = results[0]["tokens"] == results[1]["tokens"]
    if not exact or not exact_state:
        raise AssertionError("ordinary A/B output or state mismatch")
    return {
        "prompt_tokens": len(ids),
        "prefill_step": 1024,
        "arms": results,
        "tokens_exact": exact,
        "state_exact": exact_state,
    }


def unit(spec):
    from mlx2.desktop_campaign_worker import check_ownership

    ownership = check_ownership(spec)
    out = Path(spec["out"])
    report = {
        "checkpoint": spec["checkpoint"],
        "source": source(),
        "ownership": ownership,
        "scope": "exploratory GPU correctness and component-memory probe; no preflight/qualification",
        "qualified": False,
        "passed": False,
        "before": host(),
        "thermal": [],
    }
    server = None
    try:
        report["thermal"].append(thermal())
        import mlx.core as mx

        mx.set_default_device(mx.gpu)
        mx.set_cache_limit(256 << 20)
        from mlx2.adapters.registry import inspect_model

        resolution = inspect_model(spec["model"])
        if resolution.artifact_fingerprint != spec["fingerprint"]:
            raise ValueError("artifact changed since inventory")
        from mlx2.adapters.flash_next import configure_environment

        configure_environment(Path(spec["model"]))
        report["device"] = mx.metal.device_info()
        if spec.get("components_evidence"):
            previous_path = Path(spec["components_evidence"])
            previous = json.loads(previous_path.read_text())
            current_files = report["source"]["files"]
            prior_files = previous["source"]["files"]
            if previous["checkpoint"] != spec["checkpoint"] or any(
                prior_files.get(name) != digest
                for name, digest in current_files.items()
                if name.startswith("src/")
            ):
                raise ValueError(
                    "prior component evidence has different artifact/runtime source"
                )
            report["components_evidence"] = {
                "path": str(previous_path),
                "sha256": hashlib.sha256(previous_path.read_bytes()).hexdigest(),
                "runtime_source_unchanged": True,
                "scope": "reuse passing component checks only; prior HTTP contract failed",
            }
        else:
            report["components"] = components(mx, spec)
        atomic(out / "result.json", report)
        gc.collect()
        mx.clear_cache()
        report["thermal"].append(thermal())
        command = [
            sys.executable,
            "-u",
            str(Path(__file__)),
            "direct",
            spec["spec_path"],
        ]
        reference_process = subprocess.Popen(command, cwd=ROOT, env=server_env(), start_new_session=True)
        try:
            if reference_process.wait(timeout=600):
                raise RuntimeError("ordinary A/B child failed")
        finally:
            stop(reference_process)
        report["ordinary_reference_ab"] = json.loads((out / "direct.json").read_text())
        atomic(out / "result.json", report)
        gc.collect()
        mx.clear_cache()
        report["thermal"].append(thermal())
        policy = out / "policy.json"
        atomic(policy, {"qsa_lazy_causal_mask": True})
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        command = [
            sys.executable,
            "-m",
            "mlx2.server",
            "--model",
            spec["model"],
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--fabric",
            "off",
            "--ordinary",
            "--no-auto-route",
            "--max-lanes",
            "1",
            "--max-inflight",
            "1",
            "--max-context",
            "8192",
            "--prefill-step",
            "1024",
            "--cache-bytes",
            str(spec.get("cache_bytes", 512 << 20)),
            "--execution-policy",
            str(policy),
        ]
        with (out / "server.log").open("w") as log:
            server = subprocess.Popen(
                command,
                cwd=ROOT,
                env=server_env(),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + 240
            while True:
                if server.poll() is not None:
                    raise RuntimeError("loopback server exited")
                try:
                    state = http(base, "/v1/status")
                    if state.get("healthy"):
                        break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("loopback startup")
                time.sleep(1)
            report["server_command"] = command
            report["server_before"] = state
            model_id = http(base, "/v1/models")["data"][0]["id"]
            prompt = (
                "This is the QSA visibility smoke. " * 750
                + "\nThe identifier is QSA-7429. Return only this identifier.\nAnswer:"
            )
            request = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 64,
                "temperature": 0,
                "enable_thinking": False,
                "reasoning_effort": "none",
            }
            report["thermal"].append(thermal())
            cold = http(base, "/v1/chat/completions", request)
            report["thermal"].append(thermal())
            warm = http(base, "/v1/chat/completions", request)
            report["server_after"] = http(base, "/v1/status")
            text = lambda response: response["choices"][0]["message"]["content"]
            cached = (
                warm.get("usage", {})
                .get("prompt_tokens_details", {})
                .get("cached_tokens", 0)
            )
            report["http"] = {
                "cold": cold,
                "warm": warm,
                "text_exact": text(cold) == text(warm),
                "cached_tokens": cached,
                "identifier_present": "QSA-7429" in text(cold),
            }
            counts = (
                report["server_after"]
                .get("execution", {})
                .get("qsa_lazy_causal_mask", {})
                .get("counts", {})
            )
            report["served_candidate_counts"] = counts
            report["passed"] = (
                text(cold) == text(warm)
                and cached > 0
                and "QSA-7429" in text(cold)
                and sum(
                    counts.get(name, 0)
                    for name in (
                        "dense_mask_calls",
                        "fused_mask_calls",
                        "compact_calls",
                    )
                )
                > 0
            )
            if not report["passed"]:
                raise AssertionError("HTTP cold/warm functional smoke failed")
        report["thermal"].append(thermal())
    except BaseException as error:  # noqa: BLE001 -- evidence and process cleanup on every failure
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        if hasattr(error, "sample"):
            report["rejected_thermal"] = error.sample
    finally:
        stop(server)
        report["after"] = host()
        report["server_retired"] = server is None or server.poll() is not None
        atomic(out / "result.json", report)
    print(
        json.dumps(
            {
                "checkpoint": spec["checkpoint"],
                "passed": report["passed"],
                "error": report.get("error"),
            }
        ),
        flush=True,
    )
    return 0 if report["passed"] else 1


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("owned", "unit", "direct"))
    parser.add_argument("spec", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    spec["spec_path"] = str(args.spec.resolve())
    if args.mode == "direct":
        from mlx2.desktop_campaign_worker import check_ownership

        check_ownership(spec)
        import mlx.core as mx

        mx.set_default_device(mx.gpu)
        mx.set_cache_limit(256 << 20)
        from mlx2.adapters.flash_next import configure_environment

        configure_environment(Path(spec["model"]))
        atomic(Path(spec["out"]) / "direct.json", direct(mx, spec))
        return 0
    if args.mode == "unit":
        return unit(spec)
    out = Path(spec["out"])
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    from mlx2.desktop_campaign_lease import DesktopCPGLease

    with DesktopCPGLease(
        out,
        spec["job_id"],
        checkpoint=spec["checkpoint"],
        source_revision=spec["source_revision"],
    ) as lease:
        command = [
            sys.executable,
            str(ROOT / "scripts/run_with_gpu_locks.py"),
            "--session",
            lease.owner["cpg_session"],
            "--label",
            spec["job_id"],
            "--desktop-cpg-lease",
            str(lease.path),
            "--receipt",
            str(out / "paired-locks.json"),
            "--",
            sys.executable,
            "-u",
            str(Path(__file__)),
            "unit",
            str(args.spec.resolve()),
        ]
        child = subprocess.Popen(command, cwd=ROOT, env=server_env(), start_new_session=True)
        try:
            while child.poll() is None:
                lease.check()
                time.sleep(1)
            lease.success = child.returncode == 0
            return child.returncode
        finally:
            stop(child)


if __name__ == "__main__":
    raise SystemExit(main())
