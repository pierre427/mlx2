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
    if not args.context_tokens:
        return dict(CASES)
    from transformers import AutoTokenizer

    from mlx2.runtime.tokenizer_integrity import repair_loaded_tokenizer
    from mlx2.runtime.tokenizer_utils import TokenizerWrapper

    raw_tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    repair_loaded_tokenizer(raw_tokenizer, args.model)
    tokenizer = TokenizerWrapper(raw_tokenizer)
    start = (
        "Read the document ledger below. The saved block is: "
        + COPY_TEXT
        + "\nRemember the saved block while reading the unrelated records.\n"
    )
    end = "\nNow repeat the saved block exactly, with no introduction or extra text."
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
    return {"copy": (text, COPY_TEXT.strip())}


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


def reference(args):
    ownership()
    import mlx.core as mx

    from mlx2.adapters.inkling import InklingAdapter

    adapter = InklingAdapter(str(args.model))
    try:
        if args.arm in ("lane", "ast", "ast_fused"):
            from mlx2.runtime.lane import install

            install(adapter.model, min_rows=1, max_rows=32, chunk_above_max=True)
        rows = {}
        for name, (text, oracle) in CASES.items():
            prompt = adapter.prompt_tokens(body(text))
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
            token = prompt[-1]
            generated = []
            started = time.monotonic()
            for _ in range(128 if name == "copy" else 40):
                logits = adapter.model(mx.array([[token]], mx.uint32), cache=cache)
                token = int(mx.argmax(logits[0, -1]).item())
                generated.append(token)
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
                "token_ids": generated,
                "raw": raw,
                "content": content,
                "oracle": oracle,
                "oracle_pass": content.strip() == oracle,
                "seconds": time.monotonic() - started,
            }
            del cache
        atomic(
            args.out / "reference.json",
            {"rows": rows, "kernels": adapter.model.kernel_receipt()},
        )
    finally:
        adapter.close()
        del adapter
        gc.collect()
        mx.clear_cache()


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
        choices=("ordinary", "fused", "lane", "ast", "ast_fused"),
        default="ordinary",
    )
    ap.add_argument("--port", type=int, default=8288)
    ap.add_argument("--max-context", type=int, default=32768)
    ap.add_argument("--context-tokens", type=int, default=0)
    ap.add_argument("--cache-bytes", type=int, default=8 << 30)
    ap.add_argument("--prefill-step", type=int, default=128)
    ap.add_argument("--reference-child", action="store_true")
    ap.add_argument("--logprob-position", type=int)
    args = ap.parse_args()
    if args.context_tokens and args.context_tokens + 256 > args.max_context:
        raise ValueError("long-context probe needs 256 tokens of output headroom")
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
        "ownership": ownership(),
        "host_before": host(),
        "qualification": "unqualified",
        "performance_scope": "directional functional smoke; no thermal admission or publishable statistics",
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
                    "--cache-bytes",
                    str(args.cache_bytes),
                    "--prefill-step",
                    str(args.prefill_step),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            if child.wait(timeout=600):
                raise RuntimeError("ordinary reference failed; see reference.log")
            child = None
        baseline = json.loads((args.out / "reference.json").read_text())
        report["ordinary_reference"] = baseline
        policy = {}
        if args.arm in ("fused", "ast_fused"):
            policy.update(
                inkling_fused=True,
                inkling_q2_down_combine=True,
                inkling_moe_nax="fused",
            )
        if args.arm in ("ast", "ast_fused"):
            policy.update(draft_model="prompt_lookup", inkling_tree_executor="packed")
        policy_path = args.out / "policy.json"
        atomic(policy_path, policy)
        lane = "exact" if args.arm in ("lane", "ast", "ast_fused") else "off"
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
            "--external-draft" if args.arm in ("ast", "ast_fused") else "--ordinary",
        ]
        report["server_command"] = cmd
        with (args.out / "server.log").open("w") as log:
            child = subprocess.Popen(
                cmd,
                cwd=ROOT,
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
                    row["oracle_pass"] = content.strip() == oracle
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
            batch_cases = (
                (CASES["copy"], ("Please " + CASES["copy"][0], CASES["copy"][1]))
                if args.context_tokens
                else (CASES["arithmetic"], CASES["sheep"])
            )
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
            if args.arm in ("ast", "ast_fused"):
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
        report["parallel_pass"] = all(
            row["response"]["choices"][0]["message"].get("content", "").strip()
            == oracle
            for row, (_, oracle) in zip(report["parallel"], batch_cases, strict=True)
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
