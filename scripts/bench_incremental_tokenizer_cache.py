#!/usr/bin/env python3
"""CPU-only receipt for the production incremental-tokenizer candidate.

Loads tokenizer files only (no weights, MLX, Metal, service, network, or
remote code).  Every result is checked against the cache's independent full
snapshot and the ordinary HF path before any timing is reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoTokenizer

from mlx2.runtime.incremental_tokenizer_cache import IncrementalPromptTokenizerCache


class _Wrapper:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._chat_template = None
        self._v1_encode_worker = None

    def encode(self, text, *, add_special_tokens=False):
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)


class _Adapter:
    incremental_tokenizer_cache_supported = True
    incremental_tokenizer_renderer_revision = "qwen-hf-chat-template-v1"

    def __init__(self, artifact: Path):
        tokenizer = AutoTokenizer.from_pretrained(
            artifact, local_files_only=True, trust_remote_code=False
        )
        self.tokenizer = _Wrapper(tokenizer)
        digest = hashlib.sha256()
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            path = artifact / name
            if path.is_file():
                digest.update(name.encode())
                digest.update(path.read_bytes())
        self.identity = {"fingerprint": digest.hexdigest()}

    @staticmethod
    def render_incremental_prompt(tokenizer, request):
        if "messages" not in request:
            return request["prompt"]
        return tokenizer.apply_chat_template(
            request["messages"],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=request.get("enable_thinking", False),
            tools=request.get("tools")
            if request.get("tool_choice") != "none"
            else None,
        )

    def prompt_tokens(self, request):
        rendered = self.render_incremental_prompt(self.tokenizer._tokenizer, request)
        return self.tokenizer.encode(rendered, add_special_tokens=False)


def _cases(base_characters: int, turns: int):
    unit = "English and CJK 你好; def f(x): return x + 1; punctuation []{}.\n"
    document = (unit * (base_characters // len(unit) + 1))[:base_characters]
    messages = [
        {"role": "system", "content": "Be exact."},
        {"role": "user", "content": document},
    ]
    out = []
    for index in range(turns):
        messages = [
            *messages,
            {"role": "assistant", "content": f"answer {index}"},
            {"role": "user", "content": f"follow-up {index}"},
        ]
        out.append((f"growing-{index}", {"messages": messages}))
    edited = [dict(message) for message in messages]
    edited[0] = {"role": "system", "content": "Be concise."}
    out.append(("edited-earlier-turn", {"messages": edited}))
    out.append(
        (
            "quoted-special-literal",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Quote <|im_end|> and <think> literally.",
                    }
                ]
            },
        )
    )
    return out


def run(artifact: Path, *, base_characters: int, turns: int, repeats: int) -> dict:
    samples = {
        name: {"ordinary_ns": [], "candidate_ns": []}
        for name, _ in _cases(base_characters, turns)
    }
    final_rows = None
    final_status = None
    for _repeat in range(repeats):
        adapter = _Adapter(artifact)
        cache = IncrementalPromptTokenizerCache(
            max_entries=4,
            max_characters=max(8 << 20, base_characters * 8),
            max_tokens=2 << 20,
        )
        if not cache.bind(adapter):
            raise RuntimeError(cache.status()["refusal"])
        rows = []
        for name, request in _cases(base_characters, turns):
            prepared = cache.prepare(request)
            started = time.perf_counter_ns()
            ordinary = adapter.prompt_tokens(request)
            ordinary_ns = time.perf_counter_ns() - started
            started = time.perf_counter_ns()
            actual, receipt = cache.tokenize(
                prepared, lambda ordinary=ordinary: ordinary
            )
            candidate_ns = time.perf_counter_ns() - started
            reference = cache.bound_full_tokens(prepared)
            if actual != reference or ordinary != reference:
                raise RuntimeError(f"{name}: candidate/reference/ordinary IDs differ")
            samples[name]["ordinary_ns"].append(ordinary_ns)
            samples[name]["candidate_ns"].append(candidate_ns)
            rows.append(
                {
                    "name": name,
                    "characters": len(prepared.text),
                    "tokens": len(actual),
                    "action": receipt["action"],
                    "exact": True,
                    "reused_characters": receipt.get("reused_characters", 0),
                    "reused_tokens": receipt.get("reused_tokens", 0),
                }
            )
        final_rows = rows
        final_status = cache.status()
    for row in final_rows:
        timing = samples[row["name"]]
        row["ordinary_median_ns"] = int(statistics.median(timing["ordinary_ns"]))
        row["candidate_median_ns"] = int(statistics.median(timing["candidate_ns"]))
        row["component_speedup"] = row["ordinary_median_ns"] / max(
            row["candidate_median_ns"], 1
        )
    return {
        "schema": "mlx2.incremental-tokenizer-cache-benchmark.v1",
        "status": {
            "implemented": True,
            "qualified": False,
            "selected": False,
            "observed_used": bool(final_status["observed_used"]),
            "serving_qualified": False,
            "selection_note": "benchmark-local selection only; production default remains off",
        },
        "environment": {
            "cpu_only": True,
            "gpu_used": False,
            "model_loaded": False,
            "service_started": False,
            "network_used": False,
            "trust_remote_code": False,
        },
        "artifact": str(artifact.resolve()),
        "tokenizer_revision": final_status["tokenizer_revision"],
        "parameters": {
            "base_characters": base_characters,
            "turns": turns,
            "repeats": repeats,
            "entries": 4,
        },
        "all_ids_exact": all(row["exact"] for row in final_rows),
        "cases": final_rows,
        "source": {
            "repository": "Niko1221/Strata",
            "pull_request": 567,
            "revision": "76b916e2b54584e1b8f169874d71490a74175245",
            "plain_span_review": "Strata PR 931 head afe2706209d534fc67cf89277b18636a893ebff8",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--base-characters", type=int, default=100_000)
    parser.add_argument("--turns", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if min(args.base_characters, args.turns, args.repeats) < 1:
        parser.error("benchmark sizes must be positive")
    report = run(
        args.artifact,
        base_characters=args.base_characters,
        turns=args.turns,
        repeats=args.repeats,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.write_text(encoded + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
