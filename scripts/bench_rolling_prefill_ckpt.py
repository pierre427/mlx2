"""GPU A/B driver for rolling disposable prefill checkpoints (item 7).

Against a running mlx2 server, with a long prompt that shares nothing with
earlier traffic:

1. ``cold``      -- one full request (TTFT/wall overhead of rolling capture).
2. ``cancel``    -- a fresh long prompt, abandoned by the client after
                    ``--cancel-after`` seconds (the server cancels on
                    disconnect), then retried: the retry's ``cached_tokens`` is
                    the prefill a rolling checkpoint saved.
3. ``concurrent``-- two identical fresh long prompts, the second sent
                    ``--stagger`` seconds after the first: with rolling on, the
                    second resumes from the first's newest published point.
4. ``repeat``    -- the cold prompt again (committed boundary hit; rolling
                    must not have left entries behind).

Usage:
  python scripts/bench_rolling_prefill_ckpt.py --url http://127.0.0.1:8298 \
      --output out.json [--repeats 2400] [--cancel-after 6] [--stagger 4]

Run once per arm (policy off / on) and compare (see the GPU queue job).
"""

import argparse
import json
import socket
import threading
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def _post(url, body, timeout=1800):
    request = Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _status(base):
    with urlopen(base + "/v1/status", timeout=60) as response:
        status = json.load(response)
    counts, apc = {}, {}

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "apcv2" and isinstance(value, dict):
                    apc.update(value)
                elif key == "counts" and isinstance(value, dict):
                    counts.update(value)
                walk(value)

    walk(status)
    return {
        "rolling_counts": {
            key: value for key, value in counts.items()
            if key.startswith("apc_rolling_") or key.startswith("apc_interior_")
            or key in {"apcv2_store_failures", "cancelled"}
        },
        "apc_lifetime": apc.get("lifetime", {}),
        "settings_rolling": ((status.get("settings") or {}).get(
            "apc_rolling_checkpoints"
        )),
    }


def _body(model, prompt, max_tokens):
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "enable_thinking": False,
    }


def _timed(base, body):
    tic = time.perf_counter()
    response = _post(base + "/v1/chat/completions", body)
    wall = time.perf_counter() - tic
    usage = response.get("usage") or {}
    choice = (response.get("choices") or [{}])[0]
    return {
        "wall_s": round(wall, 4),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get(
            "cached_tokens"
        ),
        "text": (choice.get("message") or {}).get("content"),
        "checkpoint_role": (response.get("mlx2") or {}).get("cache_checkpoint_role"),
    }


def _abandon(base, body, seconds):
    """Send a request and drop the connection after ``seconds``."""
    try:
        _post(base + "/v1/chat/completions", body, timeout=seconds)
    except (socket.timeout, TimeoutError, URLError):
        return True
    return False


def _prompt(tag, repeats):
    return f"Document {tag}.\n" + "\n".join(
        f"Record {tag}-{i}: value {i * 7 % 13}, owner {i % 97}, state {i % 5}."
        for i in range(repeats)
    ) + "\nSummarize the document in one sentence."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=2400)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--cancel-after", type=float, default=6.0)
    parser.add_argument("--stagger", type=float, default=4.0)
    args = parser.parse_args()
    base = args.url.rstrip("/")
    with urlopen(base + "/v1/models", timeout=60) as response:
        model = json.load(response)["data"][0]["id"]
    before = _status(base)
    results = {}

    cold = _body(model, _prompt("cold", args.repeats), args.max_tokens)
    results["cold"] = _timed(base, cold)
    print("cold", json.dumps(results["cold"]), flush=True)

    cancel = _body(model, _prompt("cancel", args.repeats), args.max_tokens)
    results["cancel_abandoned"] = _abandon(base, cancel, args.cancel_after)
    time.sleep(2.0)
    results["cancel_retry"] = _timed(base, cancel)
    print("cancel_retry", json.dumps(results["cancel_retry"]), flush=True)

    shared = _body(model, _prompt("shared", args.repeats), args.max_tokens)
    first = {}
    thread = threading.Thread(
        target=lambda: first.update(_timed(base, shared)), daemon=True
    )
    thread.start()
    time.sleep(args.stagger)
    results["concurrent_second"] = _timed(base, shared)
    thread.join()
    results["concurrent_first"] = first
    print("concurrent", json.dumps(results["concurrent_second"]), flush=True)

    results["repeat"] = _timed(base, cold)
    print("repeat", json.dumps(results["repeat"]), flush=True)
    after = _status(base)
    with open(args.output, "w") as handle:
        json.dump(
            {"model": model, "results": results, "before": before, "after": after},
            handle, indent=2,
        )


if __name__ == "__main__":
    main()
