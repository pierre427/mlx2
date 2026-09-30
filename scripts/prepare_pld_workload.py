#!/usr/bin/env python3
"""Prepare a pinned ragged-PLD workload (CPU and tokenizer only).

Writes the ``--prompt-ids`` file ``scripts/qualify_ragged_pld.py`` reads:
``{"prompts": [[ids]...], "max_tokens": [...], "receipt": {...}}``. It loads a
LOCAL tokenizer only (``AutoTokenizer``, ``local_files_only=True``,
``trust_remote_code=False``), applies mlx2's load-time tokenizer repair, and
renders each lane with the artifact's own chat template. It never imports
MLX, never loads weights, never downloads and never claims the GPU.

The task texts are derived from the task constructor retained in
``qualification/runs/known-limits-20260918/sanity_20x20.py`` (``tasks(rnd)``;
read, NOT imported: that module contacts a server at import). This is a new
controlled direct-model workload, not a reproduction of the 2026-09-18 HTTP
campaign, and nothing guarantees it will propose, accept or reject: the
driver reports ``coverage_refused`` when it does not. The 2026-09-26
warm-prefix prompts survive only as hashes and are not used.

Lanes (repeated ledger/code/sequence/copy structure plus one distinct lane):

  B2: ledger_copy, capital_sentences
  B4: ledger_copy, code, count_lines, capital_sentences

  PYTHONPATH=src .venv/bin/python scripts/prepare_pld_workload.py \\
      --tokenizer ~/mlx-models/North-Mini-Code-1.0-mlx-4bit --lanes 4 --out /tmp/north-b4.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONSTRUCTOR = "qualification/runs/known-limits-20260918/sanity_20x20.py"
SCHEMA = "mlx2.pld-workload.v1"
MAX_LANES_INPUT = 16384
MAX_OUTPUT = 512
TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
    "chat_template.json", "tokenizer.model", "vocab.json", "merges.txt", "added_tokens.json",
)
CAPITALS = [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Egypt", "Cairo"), ("Canada", "Ottawa"),
            ("Spain", "Madrid"), ("Germany", "Berlin"), ("Kenya", "Nairobi"), ("Peru", "Lima"), ("Norway", "Oslo"),
            ("Greece", "Athens"), ("Portugal", "Lisbon"), ("Austria", "Vienna"), ("Ireland", "Dublin"), ("Cuba", "Havana"),
            ("Poland", "Warsaw"), ("Sweden", "Stockholm"), ("Finland", "Helsinki"), ("Hungary", "Budapest"), ("Chile", "Santiago")]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def lane_texts(rnd: int):
    """(name, text, max_tokens) per lane, from the sanity_20x20 templates."""
    ledger = "".join(f"Ledger line {i}: account {(i * 13 + rnd) % 500} moved {(i * 17 + rnd) % 900} credits.\n"
                     for i in range(90))
    country2, _capital2 = CAPITALS[(rnd + 7) % 20]
    return {
        # Derived: the needle task's ledger, as a copy instruction.
        "ledger_copy": ("Copy these ledger lines exactly, one per line, then stop:\n"
                        + "".join(ledger.splitlines(keepends=True)[:24]), 160),
        "code": (f"Write a Python function named add_{rnd} that returns the sum of its two arguments. "
                 "Include a docstring and two example calls in comments.", 120),
        "count_lines": ("Count from 1 to 20, one number per line.", 64),
        "capital_sentences": (f"Write four sentences about the capital city of {country2}, naming the city.", 48),
    }


LAYOUTS = {2: ("ledger_copy", "capital_sentences"),
           4: ("ledger_copy", "code", "count_lines", "capital_sentences")}


def _ids(encoded):
    if isinstance(encoded, dict) or hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    ids = list(encoded)
    if ids and isinstance(ids[0], list):  # a batch of one
        ids = ids[0]
    if not ids or not all(isinstance(t, int) and t >= 0 for t in ids):
        raise SystemExit("refused: chat template did not produce token ids")
    return ids


def build_workload(tokenizer, lanes: int, rnd: int, template_kwargs: dict):
    """Pure: tokenizer -> {"prompts", "max_tokens", "lanes"} (no I/O)."""
    if lanes not in LAYOUTS:
        raise SystemExit("refused: --lanes must be 2 or 4")
    texts = lane_texts(rnd)
    prompts, caps, lane_receipts = [], [], []
    for name in LAYOUTS[lanes]:
        text, cap = texts[name]
        messages = [{"role": "user", "content": text}]
        rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                 **template_kwargs)
        ids = _ids(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                 **template_kwargs))
        if len(ids) > MAX_LANES_INPUT or not 1 <= cap <= MAX_OUTPUT:
            raise SystemExit(f"refused: lane {name} exceeds {MAX_LANES_INPUT} input or {MAX_OUTPUT} output")
        prompts.append(ids)
        caps.append(cap)
        lane_receipts.append({
            "task": name, "max_tokens": cap, "prompt_tokens": len(ids),
            "text_sha256": _sha(text.encode()), "rendered_sha256": _sha(str(rendered).encode()),
            "token_ids_sha256": _sha(json.dumps(ids).encode()),
        })
    if len({len(p) for p in prompts}) != len(prompts) or len(set(caps)) != len(caps):
        raise SystemExit("refused: lanes need unequal prompt lengths and output caps")
    return {"prompts": prompts, "max_tokens": caps, "lanes": lane_receipts}


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def tokenizer_receipt(path: Path, tokenizer):
    files = {name: _sha((path / name).read_bytes()) for name in TOKENIZER_FILES if (path / name).is_file()}
    return {"path": str(path), "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            "files_sha256": files, "local_files_only": True, "trust_remote_code": False}


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokenizer", required=True, help="local artifact directory (tokenizer files only are read)")
    ap.add_argument("--lanes", type=int, choices=(2, 4), default=2)
    ap.add_argument("--round", type=int, default=0, help="sanity_20x20 round index 0..19")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="pass enable_thinking=True to the chat template (default False)")
    ap.add_argument("--out", required=True)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not 0 <= args.round <= 19:
        raise SystemExit("refused: --round 0..19")
    path = Path(args.tokenizer).expanduser().resolve()
    if not (path / "tokenizer_config.json").is_file() and not (path / "tokenizer.json").is_file():
        raise SystemExit(f"refused: no local tokenizer files in {path}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.runtime.tokenizer_integrity import repair_loaded_tokenizer  # MLX-free

    repair = repair_loaded_tokenizer(tokenizer, path)
    template_kwargs = {"enable_thinking": bool(args.enable_thinking)}
    workload = build_workload(tokenizer, args.lanes, args.round, template_kwargs)
    constructor = ROOT / SOURCE_CONSTRUCTOR
    record = {
        "prompts": workload["prompts"],
        "max_tokens": workload["max_tokens"],
        "receipt": {
            "schema": SCHEMA,
            "scope": ("new controlled direct-model PLD workload (CPU/tokenizer only); not a reproduction "
                      "of the 2026-09-18 HTTP campaign; no guarantee of proposals, acceptance or rejection"),
            "lanes": workload["lanes"], "round": args.round, "template_kwargs": template_kwargs,
            "tokenizer": tokenizer_receipt(path, tokenizer), "tokenizer_repair": repair,
            "source": {"commit": _git("rev-parse", "HEAD"),
                       "constructor": SOURCE_CONSTRUCTOR,
                       "constructor_sha256": _sha(constructor.read_bytes()) if constructor.is_file() else None,
                       "preparer_sha256": _sha(Path(__file__).read_bytes())},
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=1, default=str) + "\n")
    print(json.dumps({"out": args.out, "lanes": workload["lanes"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
