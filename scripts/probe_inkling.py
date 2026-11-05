#!/usr/bin/env python3
"""One pinned Inkling artifact/route in one bounded CPG + paired-lock window.

Ordinary reference, one loopback server, cold/warm replay and concurrent lanes.
This writes bounded functional/A-B evidence, never publishable statistics or a
qualification/default mutation. Run each arm in a fresh ownership window.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
COPY_TEXT = "BLUE ABX-739 GREEN ZXQ-284. " * 8
COPY_PROMPT = "Repeat exactly this block, with no introduction: " + COPY_TEXT
CASES = {
    "copy": (COPY_PROMPT, COPY_TEXT.strip()),
    "arithmetic": ("What is 2 + 2? Reply with only the integer.", "4"),
    "sheep": (
        "A farmer has 17 sheep. All but 9 run away. How many remain? Reply with only the integer.",
        "9",
    ),
    "long_identifier": (
        "Ignore this filler: "
        + "blue green yellow red. " * 96
        + " Reply with exactly this identifier and nothing else: ABX-739",
        "ABX-739",
    ),
    "identifier": (
        "Reply with exactly this identifier and nothing else: ABX-739",
        "ABX-739",
    ),
}


def context_cases(args):
    """Size a varied document ledger with the checkpoint tokenizer, not characters."""
    if args.contexts:
        cases = {}
        for length in args.contexts:
            sub = argparse.Namespace(**vars(args))
            sub.contexts, sub.context_tokens = None, length
            cases.update(
                {
                    f"{name}_{length}": value
                    for name, value in context_cases(sub).items()
                }
            )
        return cases
    if not args.context_tokens:
        return dict(CASES)
    from transformers import AutoTokenizer

    from mlx2.runtime.tokenizer_integrity import repair_loaded_tokenizer
    from mlx2.runtime.tokenizer_utils import TokenizerWrapper

    raw_tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    repair_loaded_tokenizer(raw_tokenizer, args.model)
    tokenizer = TokenizerWrapper(raw_tokenizer)
    start = (
        "Read the document ledger below. The archive release identifier is ABX-739.\n"
        "Remember this identifier while reading the unrelated records.\n"
    )
    end = "\nWhat is the archive release identifier? Reply with exactly the identifier and nothing else."
    records = [
        f"Record {i:05d}: warehouse {i % 97}, parcel {i * 17 + 301}, "
        f"destination sector {i % 53}, inventory count {(i * 13) % 211}.\n"
        for i in range(args.context_tokens // 12 + 1)
    ]

    def count(text):
        return len(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
                reasoning_effort="none",
            )
        )

    lo, hi = 0, len(records)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(start + "".join(records[:mid]) + end) <= args.context_tokens:
            lo = mid
        else:
            hi = mid - 1
    text = start + "".join(records[:lo]) + end
    actual = count(text)
    if not args.context_tokens - 64 <= actual <= args.context_tokens:
        raise ValueError(f"context construction missed target: {actual}")
    cases = {"long_retrieval": (text, "ABX-739")}
    if args.generation_tokens:
        prose = (
            text.rsplit("\nWhat is", 1)[0]
            + "\nWrite a detailed technical essay explaining how a warehouse can detect inventory errors, reconcile conflicting records, and maintain a reliable audit trail. Include concrete examples and tradeoffs. Write at least 800 words."
        )
        cases["long_generation"] = (prose, None)
    return cases


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    tmp.replace(path)


def body(text):
    return {
        "messages": [{"role": "user", "content": text}],
        "reasoning_effort": "none",
        "temperature": 0,
        "max_tokens": 40,
        "return_token_trace": True,
    }


def ownership():
    from qualification_gpu_ownership import require_qualification_lease

    lock = Path(os.environ["CPG_OWNER_LOCK"])
    owner = json.loads((lock / "owner.json").read_text())
    return require_qualification_lease(
        task_id=os.environ["CPG_TASK"],
        cpg_owner_lock=lock,
        generation=owner["cpg_generation"],
    )


def host():
    result = {}
    for name, cmd in [
        ("swap", ["sysctl", "vm.swapusage"]),
        ("thermal", ["pmset", "-g", "therm"]),
    ]:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        result[name] = p.stdout.strip() or p.stderr.strip()
    return result


def performance_host():
    from mlx2.system_monitor import thermal_state

    state = {"nominal": 0, "fair": 1, "serious": 2, "critical": 3}.get(thermal_state())
    stats = subprocess.check_output(["vm_stat"], text=True, timeout=5)
    swap_io = [
        line.strip()
        for line in stats.splitlines()
        if line.startswith(("Swapins:", "Swapouts:"))
    ]
    if state not in (0, 1, 2, 3) or len(swap_io) != 2:
        raise RuntimeError("controlled timing telemetry is incomplete")
    return {
        **host(),
        "thermal_state": state,
        "thermal_source": "mlx2.system_monitor.thermal_state/NSProcessInfo",
        "swap_io": swap_io,
    }


def validate_reference_receipt(receipt, *, checkpoint, source_revision, cases):
    if (
        receipt.get("complete") is not True
        or set(receipt.get("rows", {})) != set(cases)
        or receipt.get("checkpoint") != checkpoint
        or receipt.get("source_revision") != source_revision
    ):
        raise RuntimeError(
            "ordinary reference is incomplete or has a different identity"
        )


def reference(args):
    ownership()
    import mlx.core as mx

    from mlx2.adapters.inkling import InklingAdapter

    previous_cache_limit = None
    if args.reference_allocator_cache_bytes is not None:
        previous_cache_limit = mx.set_cache_limit(args.reference_allocator_cache_bytes)
    adapter = InklingAdapter(str(args.model))
    try:
        reference_identity = {
            "allocator_cache_limit_bytes": args.reference_allocator_cache_bytes,
            "previous_allocator_cache_limit_bytes": previous_cache_limit,
            "checkpoint": args.model.name + "@" + adapter.identity["fingerprint"],
            "source_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        }
        print(
            json.dumps(
                {
                    **reference_identity,
                    "reference_loaded": True,
                    "device": str(mx.default_device()),
                }
            ),
            flush=True,
        )
        if args.arm in ("lane", "ast", "ast_fused", "ast_attention"):
            from mlx2.runtime.lane import install

            lane_options = (
                json.loads(args.lane_policy.read_text()) if args.lane_policy else {}
            )
            install(
                adapter.model,
                min_rows=1,
                max_rows=lane_options.get("max_rows", 32),
                chunk_above_max=lane_options.get("chunk_above_max", True),
            )
        rows = {}
        for name, (text, oracle) in CASES.items():
            prompt = adapter.prompt_tokens(body(text))
            prefill_started = time.monotonic()
            print(
                json.dumps(
                    {
                        "case": name,
                        "prompt_tokens": len(prompt),
                        "phase": "prefill",
                        "metal_active_bytes": mx.get_active_memory(),
                        "metal_cached_bytes": mx.get_cache_memory(),
                    }
                ),
                flush=True,
            )
            cache = adapter.model.make_cache()
            for begin in range(0, len(prompt) - 1, args.prefill_step):
                adapter.model(
                    mx.array(
                        [
                            prompt[
                                begin : min(begin + args.prefill_step, len(prompt) - 1)
                            ]
                        ],
                        mx.uint32,
                    ),
                    cache=cache,
                )
                mx.eval([item.state for item in cache])
                print(
                    json.dumps(
                        {
                            "case": name,
                            "prefilled": min(
                                begin + args.prefill_step, len(prompt) - 1
                            ),
                            "seconds": time.monotonic() - prefill_started,
                        }
                    ),
                    flush=True,
                )
            token = prompt[-1]
            generated = []
            logprob_window = None
            started = time.monotonic()
            for _ in range(
                args.generation_tokens
                if name.startswith("long_generation")
                else 128
                if name == "copy"
                else 40
            ):
                logits = adapter.model(mx.array([[token]], mx.uint32), cache=cache)
                if args.logprob_position == len(generated) + 1:
                    lp = logits[0, -1].astype(mx.float32)
                    lp = lp - mx.logsumexp(lp)
                    ids = mx.argsort(-lp)[:20]
                    logprob_window = {
                        "position": len(generated) + 1,
                        "top": list(zip(ids.tolist(), lp[ids].tolist())),
                    }
                token = int(mx.argmax(logits[0, -1]).item())
                generated.append(token)
                if len(generated) % 64 == 0:
                    print(
                        json.dumps(
                            {
                                "case": name,
                                "generated": len(generated),
                                "seconds": time.monotonic() - started,
                            }
                        ),
                        flush=True,
                    )
                if token in adapter.tokenizer.eos_token_ids:
                    break
            raw = adapter.tokenizer.decode(generated, skip_special_tokens=False)
            parser = adapter.output_parser(body(text))
            events = parser.finish(
                raw, "stop" if token in adapter.tokenizer.eos_token_ids else "length"
            )
            content = "".join(e.get("content", "") for e in events)
            rows[name] = {
                "prompt_tokens": prompt,
                "logprob_window": logprob_window,
                "token_ids": generated,
                "raw": raw,
                "content": content,
                "oracle": oracle,
                "oracle_pass": content.strip() == oracle
                if oracle is not None
                else bool(content.strip()),
                "seconds": time.monotonic() - started,
                "prefill_seconds": started - prefill_started,
            }
            del cache
            if args.reference_allocator_cache_bytes is not None:
                gc.collect()
                mx.clear_cache()
            atomic(
                args.out / "reference.json",
                {
                    **reference_identity,
                    "rows": rows,
                    "complete": False,
                    "kernels": adapter.model.kernel_receipt(),
                },
            )
            print(
                json.dumps(
                    {
                        "case": name,
                        "complete": True,
                        "oracle_pass": rows[name]["oracle_pass"],
                    }
                ),
                flush=True,
            )
        atomic(
            args.out / "reference.json",
            {
                **reference_identity,
                "rows": rows,
                "complete": True,
                "kernels": adapter.model.kernel_receipt(),
            },
        )
    finally:
        adapter.close()
        del adapter
        gc.collect()
        mx.clear_cache()
        if previous_cache_limit is not None:
            mx.set_cache_limit(previous_cache_limit)


def request(base, path, payload=None, *, admin=None):
    headers = {"Content-Type": "application/json"}
    if admin is not None:
        headers["Authorization"] = "Bearer " + admin
    req = Request(
        base + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers=headers,
    )
    started = time.monotonic()
    try:
        with urlopen(req, timeout=300) as response:
            result = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(
            f"{path}: HTTP {exc.code}: {exc.read().decode()[:1000]}"
        ) from exc
    return {"response": result, "seconds": time.monotonic() - started}


def tokens(row):
    return row["response"].get("mlx2", {}).get("token_trace", {}).get("token_ids", [])


def first_difference(left, right):
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return i
    return min(len(left), len(right)) if len(left) != len(right) else None


def stop(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--arm",
        choices=(
            "ordinary",
            "fused",
            "lane",
            "ast",
            "ast_fused",
            "attention",
            "ast_attention",
        ),
        default="ordinary",
    )
    ap.add_argument("--port", type=int, default=8288)
    ap.add_argument("--max-context", type=int, default=32768)
    ap.add_argument("--context-tokens", type=int, default=0)
    ap.add_argument("--contexts", type=int, nargs="+")
    ap.add_argument("--cache-bytes", type=int, default=8 << 30)
    ap.add_argument("--prefill-step", type=int, default=128)
    ap.add_argument("--policy", type=Path)
    ap.add_argument("--lane-policy", type=Path)
    ap.add_argument("--generation-tokens", type=int, default=0)
    ap.add_argument("--skip-parallel", action="store_true")
    ap.add_argument("--parallel-generation", action="store_true")
    ap.add_argument("--performance-repeats", type=int, default=0)
    ap.add_argument("--operator-prelude", action="store_true")
    ap.add_argument("--kernel-tests", action="store_true")
    ap.add_argument("--divergence-reports", type=Path, nargs="+")
    ap.add_argument("--reference-child", action="store_true")
    ap.add_argument("--reference-timeout", type=int, default=600)
    ap.add_argument("--reference-allocator-cache-bytes", type=int)
    ap.add_argument("--logprob-position", type=int)
    args = ap.parse_args()
    if args.contexts:
        if len(set(args.contexts)) != len(args.contexts) or min(args.contexts) <= 0:
            raise ValueError("contexts must be unique positive token counts")
        args.context_tokens = max(args.contexts)
    if not 0 <= args.performance_repeats <= 4:
        raise ValueError("performance repeats must be in 0..4")
    if args.performance_repeats and not (
        args.context_tokens and args.generation_tokens
    ):
        raise ValueError("performance repeats require sustained long generation")
    if args.context_tokens and args.context_tokens + 256 > args.max_context:
        raise ValueError("long-context probe needs 256 tokens of output headroom")
    if args.parallel_generation and not (
        args.context_tokens and args.generation_tokens
    ):
        raise ValueError("parallel generation requires context and generation lengths")
    global CASES
    CASES = context_cases(args)
    if args.reference_child:
        reference(args)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    ).strip()
    if dirty:
        raise RuntimeError("probe requires a committed source tree")
    from mlx2.adapters.inkling import inspect_artifact

    artifact = inspect_artifact(args.model)
    report = {
        "checkpoint": args.model.name + "@" + artifact["identity"]["fingerprint"],
        "source_revision": source,
        "artifact": artifact["identity"],
        "arm": args.arm,
        "policy": str(args.policy) if args.policy else None,
        "lane_policy": str(args.lane_policy) if args.lane_policy else None,
        "lane_policy_contents": json.loads(args.lane_policy.read_text())
        if args.lane_policy
        else None,
        "generation_tokens": args.generation_tokens,
        "contexts": args.contexts,
        "parallel_generation_requested": args.parallel_generation,
        "ownership": ownership(),
        "host_before": host(),
        "qualification": "unqualified",
        "performance_scope": (
            "Separate repeated timing cells require thermal and swap-I/O gates; functional cases are not performance qualification"
            if args.performance_repeats
            else "directional functional smoke; no thermal admission or publishable statistics"
        ),
        "shape": {
            "max_context": args.max_context,
            "max_lanes": 2,
            "max_inflight": 4,
            "cache_bytes": args.cache_bytes,
            "prefill_step": args.prefill_step,
        },
        "cases": {},
    }
    atomic(args.out / "report.json", report)
    child = None
    admin = secrets.token_urlsafe(32)
    admin_file = args.out / ".admin-token"
    fd = os.open(admin_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(admin)

    def interrupted(sig, frame):
        raise KeyboardInterrupt()

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        preludes = []
        if args.kernel_tests:
            preludes.append(
                (
                    "kernels",
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        "tests/test_inkling_kernels_gpu.py",
                        f"--junitxml={args.out}/kernels.xml",
                    ],
                )
            )
        if args.divergence_reports:
            preludes.append(
                (
                    "divergences",
                    [
                        sys.executable,
                        str(ROOT / "scripts/probe_inkling_divergence.py"),
                        "--model",
                        str(args.model),
                        "--out",
                        str(args.out / "divergences"),
                        "--reports",
                        *map(str, args.divergence_reports),
                    ],
                )
            )
        for label, command in preludes:
            with (args.out / f"{label}.log").open("w") as log:
                log.write(
                    f"checkpoint {report['checkpoint']}\nsource_revision {source}\n"
                )
                log.flush()
                child = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env={**os.environ, "MLX2_RUN_GPU_TESTS": "1"},
                )
                if child.wait(timeout=900):
                    raise RuntimeError(f"{label} prelude failed; see {label}.log")
                child = None
        if args.operator_prelude:
            with (args.out / "operators.log").open("w") as log:
                child = subprocess.Popen(
                    [
                        sys.executable,
                        str(ROOT / "scripts/bench_inkling_ops.py"),
                        "--model",
                        str(args.model),
                        "--out",
                        str(args.out / "operators"),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                if child.wait(timeout=300):
                    raise RuntimeError(
                        "actual-weight operator prelude failed; see operators.log"
                    )
                child = None
            report["operator_prelude"] = json.loads(
                (args.out / "operators/report.json").read_text()
            )
            atomic(args.out / "report.json", report)
        with (args.out / "reference.log").open("w") as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--reference-child",
                    "--model",
                    str(args.model),
                    "--out",
                    str(args.out),
                    "--arm",
                    args.arm,
                    "--max-context",
                    str(args.max_context),
                    "--context-tokens",
                    str(args.context_tokens),
                    *(
                        ["--contexts", *map(str, args.contexts)]
                        if args.contexts
                        else []
                    ),
                    "--cache-bytes",
                    str(args.cache_bytes),
                    "--prefill-step",
                    str(args.prefill_step),
                    "--generation-tokens",
                    str(args.generation_tokens),
                    *(
                        [
                            "--reference-allocator-cache-bytes",
                            str(args.reference_allocator_cache_bytes),
                        ]
                        if args.reference_allocator_cache_bytes is not None
                        else []
                    ),
                    *(
                        ["--logprob-position", str(args.logprob_position)]
                        if args.logprob_position is not None
                        else []
                    ),
                    *(
                        ["--lane-policy", str(args.lane_policy)]
                        if args.lane_policy
                        else []
                    ),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            if child.wait(timeout=args.reference_timeout):
                raise RuntimeError("ordinary reference failed; see reference.log")
            child = None
        baseline = json.loads((args.out / "reference.json").read_text())
        validate_reference_receipt(
            baseline,
            checkpoint=report["checkpoint"],
            source_revision=source,
            cases=CASES,
        )
        report["ordinary_reference"] = baseline
        policy = {}
        if args.arm in ("fused", "ast_fused", "attention", "ast_attention"):
            policy.update(
                inkling_fused=True,
                inkling_q2_down_combine=True,
                inkling_moe_nax="fused",
            )
        if args.arm in ("attention", "ast_attention"):
            policy["inkling_attention"] = "nax"
        if args.arm in ("ast", "ast_fused", "ast_attention"):
            policy.update(draft_model="prompt_lookup", inkling_tree_executor="packed")
        if args.policy:
            policy.update(json.loads(args.policy.read_text()))
        policy_path = args.out / "policy.json"
        atomic(policy_path, policy)
        lane = (
            "exact"
            if args.arm in ("lane", "ast", "ast_fused", "ast_attention")
            else "off"
        )
        cmd = [
            sys.executable,
            "-m",
            "mlx2.server",
            "--model",
            str(args.model),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--max-context",
            str(args.max_context),
            "--max-lanes",
            "2",
            "--max-inflight",
            "4",
            "--cache-bytes",
            str(args.cache_bytes),
            "--cache-dir",
            str(args.out / "cache"),
            "--prefill-step",
            str(args.prefill_step),
            "--lane-matmul",
            lane,
            "--admin-token-file",
            str(admin_file),
            "--execution-policy",
            str(policy_path),
            "--external-draft"
            if args.arm in ("ast", "ast_fused", "ast_attention")
            else "--ordinary",
        ]
        if args.lane_policy:
            cmd.extend(["--lane-policy", str(args.lane_policy)])
        report["server_command"] = cmd
        with (args.out / "server.log").open("w") as log:
            # The shared venv's editable install points at the primary
            # checkout; pin this tree so source_revision names the code run.
            child = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env={
                    **os.environ,
                    "PYTHONPATH": os.pathsep.join(
                        filter(None, (str(ROOT / "src"), os.environ.get("PYTHONPATH")))
                    ),
                },
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            base = f"http://127.0.0.1:{args.port}"
            deadline = time.monotonic() + 600
            while True:
                if child.poll() is not None:
                    raise RuntimeError("server exited during load; see server.log")
                try:
                    health = request(base, "/health")["response"]
                    if health.get("ready") is True or health.get("status") in (
                        "ok",
                        "ready",
                    ):
                        break
                except (URLError, TimeoutError, RuntimeError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("server did not become ready")
                time.sleep(1)
            report["status_before"] = request(base, "/v1/status")["response"]
            for name, (text, oracle) in CASES.items():
                request(
                    base,
                    "/v1/admin/apc/clear",
                    {"reason": "inkling bounded probe"},
                    admin=admin,
                )
                payload = body(text)
                if name == "copy":
                    payload["max_tokens"] = 128
                if name.startswith("long_generation"):
                    payload["max_tokens"] = args.generation_tokens
                if args.logprob_position:
                    payload.update(
                        logprobs=True,
                        top_logprobs=20,
                        logprobs_start=args.logprob_position,
                        logprobs_end=args.logprob_position,
                    )
                cold = request(base, "/v1/chat/completions", payload)
                warm = request(base, "/v1/chat/completions", payload)
                for row in (cold, warm):
                    content = (
                        row["response"]["choices"][0]["message"].get("content", "")
                        or ""
                    )
                    row["oracle_pass"] = (
                        content.strip() == oracle
                        if oracle is not None
                        else bool(content.strip())
                    )
                ref_ids = baseline["rows"][name]["token_ids"]
                report["cases"][name] = {
                    "cold": cold,
                    "warm": warm,
                    "first_reference_divergence": first_difference(
                        ref_ids, tokens(cold)
                    ),
                    "first_cold_warm_divergence": first_difference(
                        tokens(cold), tokens(warm)
                    ),
                }
                atomic(args.out / "report.json", report)
            request(
                base,
                "/v1/admin/apc/clear",
                {"reason": "inkling bounded batch probe"},
                admin=admin,
            )
            retrieval_key = next(
                (name for name in reversed(CASES) if name.startswith("long_retrieval")),
                None,
            )
            batch_cases = (
                (
                    CASES[retrieval_key],
                    (
                        "Please " + CASES[retrieval_key][0],
                        CASES[retrieval_key][1],
                    ),
                )
                if args.context_tokens
                else (CASES["arithmetic"], CASES["sheep"])
            )
            report["parallel"] = []
            if not args.skip_parallel:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [
                        pool.submit(
                            request,
                            base,
                            "/v1/chat/completions",
                            {
                                **body(text),
                                "max_tokens": 128 if args.context_tokens else 40,
                            },
                        )
                        for text, _ in batch_cases
                    ]
                    report["parallel"] = [future.result() for future in futures]
            if args.arm in ("ast", "ast_fused", "ast_attention"):
                if not args.context_tokens:
                    report["ast_copy"] = report["cases"]["copy"]["cold"]
                request(
                    base,
                    "/v1/admin/apc/clear",
                    {"reason": "inkling speculative batch probe"},
                    admin=admin,
                )
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [
                        pool.submit(
                            request,
                            base,
                            "/v1/chat/completions",
                            {**body(text), "max_tokens": 128},
                        )
                        for text in (COPY_PROMPT, "Please " + COPY_PROMPT)
                    ]
                    report["ast_parallel"] = [future.result() for future in futures]
                report["ast_parallel_pass"] = all(
                    row["response"]["choices"][0]["message"].get("content", "").strip()
                    == COPY_TEXT.strip()
                    for row in report["ast_parallel"]
                )
            if args.parallel_generation:
                request(
                    base,
                    "/v1/admin/apc/clear",
                    {"reason": "inkling sustained batch probe"},
                    admin=admin,
                )
                text = next(
                    value[0]
                    for name, value in reversed(CASES.items())
                    if name.startswith("long_generation")
                )
                started = time.monotonic()
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [
                        pool.submit(
                            request,
                            base,
                            "/v1/chat/completions",
                            {**body(prompt), "max_tokens": args.generation_tokens},
                        )
                        for prompt in (text, "Please " + text)
                    ]
                    report["parallel_generation"] = [
                        future.result() for future in futures
                    ]
                report["parallel_generation_wall_seconds"] = time.monotonic() - started
                started = time.monotonic()
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [
                        pool.submit(
                            request,
                            base,
                            "/v1/chat/completions",
                            {**body(prompt), "max_tokens": args.generation_tokens},
                        )
                        for prompt in (text, "Please " + text)
                    ]
                    report["parallel_generation_warm"] = [
                        future.result() for future in futures
                    ]
                report["parallel_generation_warm_wall_seconds"] = (
                    time.monotonic() - started
                )
                report["parallel_generation_first_replay_divergences"] = [
                    first_difference(tokens(cold), tokens(warm))
                    for cold, warm in zip(
                        report["parallel_generation"],
                        report["parallel_generation_warm"],
                        strict=True,
                    )
                ]
                report["parallel_generation_scope"] = (
                    "Two independent long prompts; sustained throughput observation. "
                    "No semantic oracle or numerical-equivalence verdict for prose."
                )
            if args.performance_repeats:
                report["performance"] = []
                for name, (text, _) in CASES.items():
                    if not name.startswith("long_generation"):
                        continue
                    for repeat in range(args.performance_repeats):
                        before = performance_host()
                        if before["thermal_state"] >= 2:
                            raise RuntimeError(
                                "controlled timing refused serious/critical thermal state"
                            )
                        request(
                            base,
                            "/v1/admin/apc/clear",
                            {"reason": "controlled Inkling A/B"},
                            admin=admin,
                        )
                        payload = {**body(text), "max_tokens": args.generation_tokens}
                        cold = request(base, "/v1/chat/completions", payload)
                        warm = request(base, "/v1/chat/completions", payload)
                        after = performance_host()
                        exact = (
                            tokens(cold)
                            == tokens(warm)
                            == tokens(report["cases"][name]["cold"])
                        )
                        report["performance"].append(
                            {
                                "case": name,
                                "repeat": repeat,
                                "cold": cold,
                                "warm": warm,
                                "before": before,
                                "after": after,
                                "repeat_tokens_exact": exact,
                                "eligible": exact
                                and after["thermal_state"] < 2
                                and before["swap_io"] == after["swap_io"],
                                "scope": "controlled same-shape repeated screen; source/artifact/route bound; no population statistics",
                            }
                        )
                        atomic(args.out / "report.json", report)
            report["status_after"] = request(base, "/v1/status")["response"]
        report["functional_pass"] = all(
            row["oracle_pass"]
            for value in report["cases"].values()
            for row in (value["cold"], value["warm"])
        )
        report["ordinary_reference_pass"] = all(
            row["oracle_pass"] for row in baseline["rows"].values()
        )
        report["reference_exact"] = all(
            value["first_reference_divergence"] is None
            for value in report["cases"].values()
        )
        report["parallel_pass"] = (
            None
            if args.skip_parallel
            else all(
                row["response"]["choices"][0]["message"].get("content", "").strip()
                == oracle
                for row, (_, oracle) in zip(
                    report["parallel"], batch_cases, strict=True
                )
            )
        )
        report["width_one_replay_exact"] = all(
            value["first_cold_warm_divergence"] is None
            for value in report["cases"].values()
        )
    except BaseException as exc:
        report["error"] = str(exc)
        raise
    finally:
        stop(child)
        admin_file.unlink(missing_ok=True)
        report["host_after"] = host()
        report["finished_at"] = time.time()
        atomic(args.out / "report.json", report)
        print(
            json.dumps(
                {
                    key: report.get(key)
                    for key in (
                        "checkpoint",
                        "source_revision",
                        "arm",
                        "functional_pass",
                        "width_one_replay_exact",
                        "error",
                    )
                },
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
