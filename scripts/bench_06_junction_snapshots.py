"""GPU A/B driver for APCv2 junction snapshots (item 6).

Agent-shaped traffic against a running mlx2 server: several chat requests
share one long system prompt and diverge only in the user turn.  On a hybrid
(GDN + attention) model the second request can only snap back to a recorded
prefill-chunk checkpoint of the first; with ``apc_junction_checkpoints`` it
leaves an exact snapshot at the divergence, and every later request resumes
there.

Usage:
  python scripts/bench_06_junction_snapshots.py --url http://127.0.0.1:8297 \
      --output out.json [--system-repeats 1200] [--turns 5] [--max-tokens 16]

Writes per-request wall time, cached_tokens, prompt_tokens, text and the
server's APC counters before and after.  Run once against an arm with the
policy off and once with it on, then compare (see the GPU queue job).
"""

import argparse
import json
import time
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
    counts = {}
    found = {}

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "apcv2" and isinstance(value, dict):
                    found.update(value)
                elif key == "counts" and isinstance(value, dict):
                    counts.update(value)
                walk(value)

    walk(status)
    return {
        "junction_counts": {
            key: value for key, value in counts.items()
            if key.startswith("apc_junction_") or key.startswith("apc_interior_")
        },
        "apc_lifetime": found.get("lifetime", {}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--system-repeats", type=int, default=1200)
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()
    base = args.url.rstrip("/")
    with urlopen(base + "/v1/models", timeout=60) as response:
        model = json.load(response)["data"][0]["id"]
    # Deterministic tool-catalog-like system prompt (~10 tokens per repeat).
    system = "You are a coding agent. Tools:\n" + "\n".join(
        f"- tool_{i}: returns field {i * 7 % 13} of record {i % 97}."
        for i in range(args.system_repeats)
    )
    tasks = [
        "List three prime numbers.",
        "Name a color of the sky.",
        "What is two plus two?",
        "Spell the word cache backwards.",
        "Give one word that rhymes with tree.",
        "Name a planet.",
    ]
    before = _status(base)
    results = []
    for index in range(args.turns):
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": tasks[index % len(tasks)]},
            ],
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "enable_thinking": False,
        }
        tic = time.perf_counter()
        response = _post(base + "/v1/chat/completions", body)
        wall = time.perf_counter() - tic
        usage = response.get("usage") or {}
        choice = (response.get("choices") or [{}])[0]
        results.append({
            "turn": index,
            "wall_s": round(wall, 4),
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get(
                "cached_tokens"
            ),
            "finish_reason": choice.get("finish_reason"),
            "text": (choice.get("message") or {}).get("content"),
            "checkpoint_role": (response.get("mlx2") or {}).get(
                "cache_checkpoint_role"
            ),
        })
        print(json.dumps(results[-1]), flush=True)
    after = _status(base)
    with open(args.output, "w") as handle:
        json.dump(
            {"model": model, "results": results, "before": before, "after": after},
            handle, indent=2,
        )


if __name__ == "__main__":
    main()
