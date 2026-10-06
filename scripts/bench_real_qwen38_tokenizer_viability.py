#!/usr/bin/env python3
"""Local-tokenizer-only Qwen3.8 shared-prefix viability receipt.

No weights, model code, network, remote code, MLX, or service are used.  The
suite compares every candidate result with a fresh full encode across growing,
edited and interleaved rendered chats, then changes the plain-span policy for a
quoted ``<think>`` literal to cover the bug found during Strata #567 review.
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
sys.path.insert(0, str(ROOT))

from scripts.bench_tokenizer_viability import HFMarkedTokenizer, MarkedPrefixEncoder

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "chat_template.json",
    "vocab.json",
    "merges.txt",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_receipt(path: Path) -> dict:
    files = []
    for name in TOKENIZER_FILES:
        file = path / name
        if file.is_file():
            files.append({"name": name, "bytes": file.stat().st_size, "sha256": _sha256(file)})
    if not any(item["name"] == "tokenizer.json" for item in files):
        raise ValueError("artifact has no tokenizer.json")
    return {
        "path": str(path),
        "name": path.name,
        "files": files,
        "model_weights_opened": False,
        "network_used": False,
        "trust_remote_code": False,
    }


def _prompt(turns, *, document: str, system: str = "Be exact.") -> str:
    text = f"<|im_start|>system\n{system}<|im_end|>\n"
    text += f"<|im_start|>user\n{document}<|im_end|>\n"
    for index, (question, answer) in enumerate(turns):
        text += f"<|im_start|>assistant\n<think>step {index}</think>{answer}<|im_end|>\n"
        text += f"<|im_start|>user\n{question}<|im_end|>\n"
    return text + "<|im_start|>assistant\n"


def _cases(base_chars: int, turns: int):
    unit = "English and CJK 你好; def f(x): return x + 1; punctuation []{}.\n"
    document = (unit * (base_chars // len(unit) + 1))[:base_chars]
    growing = []
    history = []
    for index in range(turns):
        history.append((f"follow-up {index}", f"answer {index}"))
        growing.append((f"growing-{index}", _prompt(history, document=document), ()))

    edited_a = _prompt([("continue", "alpha")], document=document, system="constraint ALPHA")
    edited_b = _prompt([("continue", "alpha")], document=document, system="constraint BETA")

    short = document[: max(512, base_chars // 8)]
    a1 = _prompt([("A next", "A one")], document=short, system="conversation A")
    b1 = _prompt([("B next", "B one")], document=short, system="conversation B")
    a2 = _prompt([("A next", "A one"), ("A again", "A two")], document=short, system="conversation A")
    b2 = _prompt([("B next", "B one"), ("B again", "B two")], document=short, system="conversation B")

    quoted = _prompt(
        [("Does the quoted marker stay text?", "yes")],
        document="Quote this literally: <think>not hidden reasoning</think>.",
    )
    start = quoted.index("<think>not hidden reasoning</think>")
    stop = start + len("<think>not hidden reasoning</think>")
    quoted_plain = ((start, stop),)
    quoted_grown = quoted + "<|im_start|>user\ncontinue<|im_end|>\n<|im_start|>assistant\n"

    return [
        *growing,
        ("edited-source", edited_a, ()),
        ("edited-earlier-turn", edited_b, ()),
        ("interleaved-a1", a1, ()),
        ("interleaved-b1", b1, ()),
        ("interleaved-a2", a2, ()),
        ("interleaved-b2", b2, ()),
        ("quoted-special-normal", quoted, ()),
        ("quoted-special-plain-policy-change", quoted, quoted_plain),
        ("quoted-special-plain-growth", quoted_grown, quoted_plain),
    ]


def run_suite(artifact: Path, *, base_chars: int, turns: int, repeats: int) -> dict:
    artifact = artifact.resolve()
    tokenizer = HFMarkedTokenizer(artifact)
    cases = _cases(base_chars, turns)
    samples = {name: {"full_ns": [], "incremental_ns": []} for name, _text, _spans in cases}
    final_rows = None
    for _repeat in range(repeats):
        encoder = MarkedPrefixEncoder(tokenizer, keep=4)
        rows = []
        for name, text, spans in cases:
            started = time.perf_counter_ns()
            expected = tokenizer.encode(text, plain_spans=spans)
            full_ns = time.perf_counter_ns() - started
            started = time.perf_counter_ns()
            actual = encoder.encode(text, plain_spans=spans)
            incremental_ns = time.perf_counter_ns() - started
            if actual != expected:
                raise RuntimeError(f"{name}: incremental IDs differ from a fresh full encode")
            samples[name]["full_ns"].append(full_ns)
            samples[name]["incremental_ns"].append(incremental_ns)
            rows.append(
                {
                    "name": name,
                    "characters": len(text),
                    "tokens": len(expected),
                    "plain_spans": [list(span) for span in spans],
                    "exact": True,
                    "reused_characters": encoder.last_reused_chars,
                    "reused_tokens": encoder.last_reused_tokens,
                }
            )
        final_rows = rows
    for row in final_rows:
        timing = samples[row["name"]]
        row["full_median_ns"] = int(statistics.median(timing["full_ns"]))
        row["incremental_median_ns"] = int(statistics.median(timing["incremental_ns"]))
        row["median_speedup"] = row["full_median_ns"] / max(row["incremental_median_ns"], 1)

    plain = next(row for row in final_rows if row["name"] == "quoted-special-plain-policy-change")
    grown = next(row for row in final_rows if row["name"] == "quoted-special-plain-growth")
    return {
        "schema": "mlx2.real-qwen38-tokenizer-prefix-viability.v1",
        "status": "observed CPU tokenizer-only research; not selected or serving-qualified",
        "gpu_used": False,
        "model_loaded": False,
        "service_started": False,
        "artifact": artifact_receipt(artifact),
        "parameters": {"base_chars": base_chars, "turns": turns, "repeats": repeats, "keep": 4},
        "all_ids_exact": all(row["exact"] for row in final_rows),
        "plain_policy_change_cut_before_span": (
            plain["reused_characters"] <= plain["plain_spans"][0][0]
        ),
        "plain_policy_growth_reused_prefix": grown["reused_characters"] > 0,
        "cases": final_rows,
        "source": {
            "repository": "Niko1221/Strata",
            "pull_request": 567,
            "revision": "f29e527856b85e37bda90626ccc28e002b4989dd",
            "review_followup": "0.1.39 plain spans and quoted special literals",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--base-chars", type=int, default=100_000)
    parser.add_argument("--turns", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if min(args.base_chars, args.turns, args.repeats) < 1:
        parser.error("suite sizes must be positive")
    report = run_suite(
        args.artifact, base_chars=args.base_chars, turns=args.turns, repeats=args.repeats
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(encoded + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
