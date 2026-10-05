#!/usr/bin/env python3
"""Build a replayable multi-user traffic trace and source-bound N20 inputs."""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

from prepare_varlen_skew_inputs import digest, resize_messages

DEFAULT_SEED = 20261004
USERS = ("ada", "ben", "cy", "dee", "eli")
TURNS = 4
WAVE_SIZE = 3
REQUESTS = 18


def traffic_plan(seed: int = DEFAULT_SEED) -> list[dict]:
    """Return the immutable logical trace; no runtime clocks are consulted."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    rng = random.Random(seed)
    lengths = {}
    for user in USERS:
        current = rng.randrange(1050, 2051, 64)
        for turn in range(1, TURNS + 1):
            if turn > 1:
                current = min(7936, current + rng.randrange(640, 1793, 64))
            lengths[(user, turn)] = current
    logical = []
    for turn in range(1, TURNS + 1):
        order = list(USERS)
        rng.shuffle(order)
        for user in order:
            logical.append((user, turn))
    logical = logical[:REQUESTS]
    result = []
    for index, (user, turn) in enumerate(logical):
        result.append(
            {
                "trace_index": index,
                "wave": index // WAVE_SIZE,
                "position": index % WAVE_SIZE,
                "user": user,
                "turn": turn,
                "target_prompt_tokens": lengths[(user, turn)],
                "max_tokens": 4,
                "draft_depth": 2,
                "arrival_offset_ms": rng.randrange(0, 401, 20),
            }
        )
    return result


def derive(inputs, tokenizer, *, seed: int = DEFAULT_SEED, domain_index: int = 0):
    result = deepcopy(inputs)
    domains = result.get("domain_order", [])
    if not 0 <= domain_index < len(domains):
        raise ValueError("domain index outside manifest")
    domain = domains[domain_index]
    source_rows = [row for row in result["rows"] if row["domain"] == domain]
    trace = traffic_plan(seed)
    if len(source_rows) != 20 or len(trace) > len(source_rows):
        raise ValueError("one complete 20-row source domain is required")
    for item, row in zip(trace, source_rows):
        messages, ids = resize_messages(
            tokenizer, row["body"]["messages"], item["target_prompt_tokens"]
        )
        row["body"]["messages"] = messages
        row["body_sha256"] = digest(row["body"])
        row["prompt_token_ids"] = list(ids)
        row["prompt_tokens"] = len(ids)
        row["preparation_receipt"] = {
            "schema": "mlx2.deterministic-arbitrary-traffic-row.v1",
            "seed": seed,
            "user": item["user"],
            "turn": item["turn"],
            "target_prompt_tokens": item["target_prompt_tokens"],
            "actual_prompt_tokens": len(ids),
            "authoritative_notes_preserved": True,
            "qualification_input": False,
        }
        item["case_id"] = row["case_id"]
        item["actual_prompt_tokens"] = len(ids)
    trace_receipt = {
        "schema": "mlx2.deterministic-arbitrary-traffic.v1",
        "seed": seed,
        "domain": domain,
        "users": list(USERS),
        "turns": TURNS,
        "request_count": len(trace),
        "wave_size": WAVE_SIZE,
        "wave_count": len(trace) // WAVE_SIZE,
        "requests": trace,
        "deterministic_fields": [
            "case_id",
            "user",
            "turn",
            "target_prompt_tokens",
            "max_tokens",
            "draft_depth",
            "arrival_offset_ms",
        ],
        "runtime_nondeterminism_excluded": [
            "thread dispatch",
            "host scheduling",
            "device timing",
        ],
        "qualification_input": False,
    }
    result["deterministic_arbitrary_traffic"] = deepcopy(trace_receipt)
    result["qualified"] = False
    result["price_usable"] = False
    result["inputs_sha256"] = digest(
        {key: value for key, value in result.items() if key != "inputs_sha256"}
    )
    trace_receipt["inputs_sha256"] = result["inputs_sha256"]
    return result, trace_receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-root",
        type=Path,
        help="retokenize and bind the derived inputs to this local artifact",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--domain-index", type=int, default=0)
    args = parser.parse_args()
    inputs = json.loads(args.inputs.read_text())
    if args.tokenizer_root is not None:
        inputs["tokenizer_root"] = str(args.tokenizer_root.expanduser().resolve())
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        inputs["tokenizer_root"], trust_remote_code=True
    )
    result, trace = derive(
        inputs, tokenizer, seed=args.seed, domain_index=args.domain_index
    )
    for path, value in ((args.output, result), (args.trace, trace)):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(path)
    print(json.dumps(trace, indent=2))


if __name__ == "__main__":
    main()
