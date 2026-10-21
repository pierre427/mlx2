#!/usr/bin/env python3
"""Bounded GPU smoke of an exported HySparse2 checkpoint through mlx2.server.

One artifact, one ownership window (run it under the GPU lock wrapper):

1. research reference, in process: the research model's own cached greedy
   decode (prefill + decode) for each prompt, and the research sequential
   MTP reference (``GreedyMTPReference``) acceptance and rate;
2. for each route (``--ordinary``, default adaptive prompt lookup, ``--ast``):
   one loopback ``python -m mlx2.server`` process; greedy completions compared
   with the research tokens, a repeat for determinism and APCv2 warm reuse,
   two concurrent requests, a chat request, and streamed decode rates;
   the whole server process group is stopped in ``finally``.

The evidence directory must be new or empty so retries stay distinguishable.
These are exploratory smoke observations, not qualification or controlled
performance numbers.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PROMPTS = [
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "user: Explain what a GPU kernel is in one sentence.\n\nassistant:",
    "The attention mechanism lets a transformer model",
]
CHAT = [{"role": "user", "content": "Write a Python function that adds two numbers."}]
REFERENCE_TOKENS = 32
RATE_TOKENS = 96
# Repetitive source text: where prompt lookup has something to copy.
REPEAT_PROMPT = "".join(
    f"def scale_{i}(values, factor):\n    return [value * factor for value in values]\n\n"
    for i in range(8)
)
ROUTES = {"ordinary": ["--ordinary"], "prompt_lookup": [], "ast": ["--ast"]}


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def host_sample() -> dict:
    sample = {"time": time.time()}
    for name, command in (
        ("swap", ["sysctl", "-n", "vm.swapusage"]),
        ("thermal", ["pmset", "-g", "therm"]),
    ):
        try:
            sample[name] = subprocess.run(
                command, capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError) as error:
            sample[name] = f"unavailable: {error}"
    return sample


NEAR_TIE_NATS = 0.5


def _row(logits):
    import mlx.core as mx

    logprobs = logits.astype(mx.float32) - mx.logsumexp(logits.astype(mx.float32))
    order = mx.argsort(-logprobs)[:3].tolist()
    values = logprobs.tolist()
    return {"top": [[int(t), float(values[t])] for t in order], "logprobs": values}


def _divergence(research_rows, serving_rows, research_tokens, serving_tokens):
    """First divergent position and its bounded logprob window (AGENTS.md)."""
    for index, (a, b) in enumerate(zip(research_tokens, serving_tokens)):
        if a == b:
            continue
        r, s = research_rows[index], serving_rows[index]
        r_margin = r["top"][0][1] - r["top"][1][1]
        s_margin = s["top"][0][1] - s["top"][1][1]
        # Multi-way clause: each arm's selected token within the ceiling of
        # the other arm's top log-probability at this position.
        cross = (
            r["top"][0][1] - r["logprobs"][b] <= NEAR_TIE_NATS
            and s["top"][0][1] - s["logprobs"][a] <= NEAR_TIE_NATS
        )
        same_top2 = {t for t, _ in r["top"][:2]} == {t for t, _ in s["top"][:2]}
        return {
            "index": index,
            "research_token": a,
            "serving_token": b,
            "research_top3": r["top"],
            "serving_top3": s["top"],
            "research_margin_nats": r_margin,
            "serving_margin_nats": s_margin,
            "research_gap_to_serving_token_nats": r["top"][0][1] - r["logprobs"][b],
            "serving_gap_to_research_token_nats": s["top"][0][1] - s["logprobs"][a],
            "same_unordered_top2": same_top2,
            "near_tie_candidate": bool(
                (same_top2 and r_margin <= NEAR_TIE_NATS and s_margin <= NEAR_TIE_NATS) or cross
            ),
            "note": "functional oracle not applied; toy checkpoint has no task oracle",
        }
    return None


def research_reference(model_path: Path) -> dict:
    """Research cached decode vs the serving model, sharing one set of weights."""
    import mlx.core as mx

    from mlx2.adapters.hysparse2 import HySparse2Adapter
    from mlx2.experimental.hysparse2.speculation import GreedyMTPReference

    adapter = HySparse2Adapter(str(model_path))
    tokenizer = adapter.tokenizer
    serving, research = adapter.model, adapter.model.core
    eos = tokenizer.convert_tokens_to_ids("<eos>")
    result = {"load": adapter.load_receipt, "dtype": adapter.dtype_name, "prompts": []}

    def greedy(step, first):
        rows, tokens = [_row(first[0, -1])], []
        start = time.perf_counter()
        for _ in range(REFERENCE_TOKENS):
            token = rows[-1]["top"][0][0]
            tokens.append(token)
            rows.append(_row(step(token)[0, -1]))
        return tokens, rows, REFERENCE_TOKENS / (time.perf_counter() - start)

    for prompt in PROMPTS:
        ids = adapter.prompt_tokens({"prompt": prompt})
        start = time.perf_counter()
        logits, cache = research.prefill(mx.array([ids]))
        research_prefill = time.perf_counter() - start
        r_tokens, r_rows, r_rate = greedy(
            lambda t, cache=cache: research.decode(mx.array([[t]]), cache), logits
        )
        serving_cache = serving.make_cache()
        start = time.perf_counter()
        logits = serving(mx.array([ids]), cache=serving_cache)
        mx.eval(logits)
        serving_prefill = time.perf_counter() - start
        s_tokens, s_rows, s_rate = greedy(
            lambda t, c=serving_cache: serving(mx.array([[t]]), cache=c), logits
        )
        until = s_tokens[: s_tokens.index(eos)] if eos in s_tokens else s_tokens
        result["prompts"].append({
            "prompt": prompt,
            "prompt_tokens": len(ids),
            "research_tokens": r_tokens,
            "serving_tokens": s_tokens,
            "tokens_equal": r_tokens == s_tokens,
            "first_divergence": _divergence(r_rows, s_rows, r_tokens, s_tokens),
            "research_text": tokenizer.decode(r_tokens),
            "serving_text_until_eos": tokenizer.decode(until),
            "research_prefill_seconds": research_prefill,
            "serving_prefill_seconds": serving_prefill,
            "research_decode_tokens_per_second": r_rate,
            "serving_inprocess_decode_tokens_per_second": s_rate,
        })
    ids = adapter.prompt_tokens({"prompt": PROMPTS[0]})
    reference = GreedyMTPReference(research)
    start = time.perf_counter()
    output, _cache, receipt = reference.generate(ids, max_tokens=REFERENCE_TOKENS)
    elapsed = time.perf_counter() - start
    rounds = receipt["rounds"]
    result["mtp_reference"] = {
        "algorithm": receipt["schema"],
        "verification": receipt["verification"],
        "proposal": receipt["proposal"],
        "mtp_history_cache": receipt["mtp_history_cache"],
        "proposed": sum(r["proposed"] is not None for r in rounds),
        "accepted": sum(bool(r["accepted"]) for r in rounds),
        "tokens": len(output),
        "tokens_equal_research_greedy": output == result["prompts"][0]["research_tokens"][: len(output)],
        "wall_tokens_per_second_including_prefill": len(output) / elapsed,
        "serving_route_qualified": False,
    }
    adapter.close()
    del adapter, serving, research, reference, _cache, cache, serving_cache, logits
    gc.collect()
    mx.clear_cache()
    return result


def request(port: int, path: str, body: dict | None = None, timeout: float = 600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def streamed_rate(port: int, prompt: str, max_tokens: int) -> dict:
    body = {
        "prompt": prompt, "max_tokens": max_tokens, "min_tokens": max_tokens,
        "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first = last = None
    usage = None
    receipt = None
    with urllib.request.urlopen(req, timeout=900) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            event = json.loads(line[5:])
            if event.get("choices") and event["choices"][0].get("text"):
                now = time.perf_counter()
                first = now if first is None else first
                last = now
            usage = event.get("usage") or usage
            receipt = event.get("mlx2") or receipt
    tokens = (usage or {}).get("completion_tokens", 0)
    return {
        "completion_tokens": tokens,
        "ttft_seconds": None if first is None else first - start,
        "decode_tokens_per_second": (
            (tokens - 1) / (last - first) if first is not None and last > first and tokens > 1 else None
        ),
        "speculation": (receipt or {}).get("speculation"),
    }


def completion(port: int, prompt: str, max_tokens: int) -> dict:
    response = request(port, "/v1/completions", {
        "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
    })
    receipt = response.get("mlx2") or {}
    return {
        "text": response["choices"][0]["text"],
        "finish_reason": response["choices"][0].get("finish_reason"),
        "usage": response.get("usage"),
        "route": receipt.get("route"),
        "route_receipt": receipt.get("route_receipt"),
        "qualification": receipt.get("qualification"),
        "cached_tokens": receipt.get("cached_tokens"),
    }


def serve_route(args, name: str, flags: list, reference: dict, out: Path) -> dict:
    log = (out / f"server-{name}.log").open("w")
    command = [
        args.python, "-m", "mlx2.server", "--model", str(args.model),
        "--host", "127.0.0.1", "--port", str(args.port), "--fabric", "off",
        "--max-lanes", "2", *flags,
    ]
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    process = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
    )
    result = {"command": command, "host_before": host_sample()}
    try:
        deadline = time.monotonic() + 900
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"server exited with {process.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=5) as health:
                    if health.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("server did not become healthy")
            time.sleep(2)
        result["status"] = request(args.port, "/v1/status")
        comparisons = []
        for prompt, expected in zip(PROMPTS, reference["prompts"]):
            cold = completion(args.port, prompt, REFERENCE_TOKENS)
            warm = completion(args.port, prompt, REFERENCE_TOKENS)
            # Text of the in-process serving model's greedy tokens.  The
            # streaming detokenizer may trim one leading space and withhold
            # an incomplete trailing UTF-8 byte; compare modulo those.
            text = expected["serving_text_until_eos"]
            served = cold["text"]

            def norm(value):
                return value.removeprefix(" ").rstrip("\ufffd")

            shared = next(
                (i for i, (a, b) in enumerate(zip(norm(served), norm(text))) if a != b),
                min(len(norm(served)), len(norm(text))),
            )
            comparisons.append({
                "prompt": prompt, "cold": cold, "warm": warm,
                "inprocess_serving_text": text,
                "matches_inprocess_prefix_chars": shared,
                "equal_to_inprocess_serving": norm(served) == norm(text),
                "deterministic_repeat": served == warm["text"],
                "warm_cached_tokens": warm["cached_tokens"],
            })
        result["comparisons"] = comparisons
        concurrent = {}

        def run(index):
            concurrent[index] = completion(args.port, PROMPTS[index], REFERENCE_TOKENS)

        threads = [threading.Thread(target=run, args=(i,)) for i in (0, 2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        result["concurrent_equal_isolated"] = {
            str(i): concurrent[i]["text"] == comparisons[i]["cold"]["text"] for i in (0, 2)
        }
        chat = request(args.port, "/v1/chat/completions", {
            "messages": CHAT, "max_tokens": 48, "temperature": 0,
        })
        result["chat"] = {
            "content": chat["choices"][0]["message"].get("content"),
            "finish_reason": chat["choices"][0].get("finish_reason"),
        }
        result["rates"] = [streamed_rate(args.port, prompt, RATE_TOKENS) for prompt in [*PROMPTS, REPEAT_PROMPT]]
        result["status_after"] = request(args.port, "/v1/status")
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=60)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        log.close()
        result["host_after"] = host_sample()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--routes", default=",".join(ROUTES))
    args = parser.parse_args(argv)
    args.model = args.model.expanduser().resolve()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty evidence directory {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    from mlx2.adapters.hysparse2 import inspect_artifact

    artifact = inspect_artifact(args.model)
    checkpoint = f"{args.model.name}@{artifact['identity']['fingerprint']}"
    receipt = {
        "schema": "mlx2.hysparse2-serving-smoke.v1",
        "checkpoint": checkpoint,
        "source_revision": args.commit,
        "host": args.host_label,
        "platform": platform.platform(),
        "artifact": {k: artifact[k] for k in ("checkpoint", "manifest")},
        "status": {"implemented": True, "qualified": False, "selected": False},
        "host_start": host_sample(),
    }
    write_json(args.out / "receipt.json", receipt)
    receipt["research_reference"] = research_reference(args.model)
    write_json(args.out / "receipt.json", receipt)
    receipt["routes"] = {}
    for name in args.routes.split(","):
        receipt["routes"][name] = serve_route(args, name, ROUTES[name], receipt["research_reference"], args.out)
        write_json(args.out / "receipt.json", receipt)
        time.sleep(10)
    receipt["host_end"] = host_sample()
    write_json(args.out / "receipt.json", receipt)
    print(json.dumps({"checkpoint": checkpoint, "out": str(args.out)}))


if __name__ == "__main__":
    main()
