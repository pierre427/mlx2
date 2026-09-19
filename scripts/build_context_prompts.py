#!/usr/bin/env python3
"""Build tokenizer-calibrated, hash-frozen context ladder prompts without MLX."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

from run_qualification_matrix import atomic_json, validate_manifest


TAIL = "\nStart with exactly LADDER_READY, then explain compiler optimization in numbered sections."


def rebase_model_paths(model: dict[str, Any], base: Path) -> None:
    model["tokenizer_path"] = str((base / model["tokenizer_path"]).resolve())
    for arm in model.get("arms", []):
        receipt = arm.get("qualification_receipt")
        if receipt and receipt.get("path"):
            receipt["path"] = str((base / receipt["path"]).resolve())
    spomin = model.get("spomin20x20")
    if spomin:
        spomin["corpus_path"] = str((base / spomin["corpus_path"]).resolve())
        spomin["tokenizer_path"] = str((base / spomin["tokenizer_path"]).resolve())
        spomin["source_reconciliation_tokenizer_path"] = str(
            (base / spomin["source_reconciliation_tokenizer_path"]).resolve()
        )


def rendered_tokens(tokenizer: Any, text: str, renderer: str = "qwen_direct") -> int:
    messages = [{"role": "user", "content": text}]
    if renderer == "qwen_direct":
        tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                               tokenize=True, enable_thinking=False, tools=None)
        if isinstance(tokens, Mapping):
            tokens = tokens["input_ids"]
        if tokens and isinstance(tokens[0], list):
            if len(tokens) != 1:
                raise ValueError("prompt calibration expected one rendered conversation")
            tokens = tokens[0]
        return len(tokens)
    if renderer == "muse_direct":
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                               tokenize=False, reasoning_strength="high", tools=None)
        prompt += " to=user<|message|>"
        return len(tokenizer.encode(prompt, add_special_tokens=False))
    if renderer == "north_direct":
        tokens = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            reasoning=False,
            reasoning_effort="none",
            skip_thinking=True,
            tools=None,
        )
        if isinstance(tokens, Mapping):
            tokens = tokens["input_ids"]
        if tokens and isinstance(tokens[0], list):
            if len(tokens) != 1:
                raise ValueError("prompt calibration expected one rendered conversation")
            tokens = tokens[0]
        return len(tokens)
    raise ValueError(f"unknown tokenizer_renderer {renderer!r}")


def calibrate(tokenizer: Any, target: int, renderer: str = "qwen_direct") -> str:
    count = lambda value: rendered_tokens(tokenizer, value, renderer)
    if count(TAIL) > target:
        raise ValueError(f"target {target} is smaller than the chat-template tail")
    low, high = 0, target
    while low < high:
        middle = (low + high + 1) // 2
        if count(" data" * middle + TAIL) <= target:
            low = middle
        else:
            high = middle - 1
    text = " data" * low + TAIL
    candidates = (" x", " 0", " a", " .", "\n", " ")
    while count(text) < target:
        current = count(text)
        options = [(count(text + value), value) for value in candidates]
        options = [(count, value) for count, value in options if current < count <= target]
        if not options:
            raise RuntimeError(f"cannot calibrate exact prompt target {target}; stopped at {current}")
        _, value = max(options)
        text += value
    actual = count(text)
    if actual != target:
        raise AssertionError(f"prompt calibration produced {actual}, expected {target}")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--prompt-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    validate_manifest(manifest)
    manifest["calibration_source"] = {
        "path": str(args.manifest.resolve()),
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    from transformers import AutoTokenizer
    args.prompt_dir.mkdir(parents=True, exist_ok=True)
    for model in manifest["models"]:
        rebase_model_paths(model, args.manifest.parent)
        tokenizer_path = Path(model["tokenizer_path"])
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False, local_files_only=True)
        renderer = model["tokenizer_renderer"]
        for context in model["contexts"]:
            text = calibrate(tokenizer, int(context["tokens"]), renderer)
            path = args.prompt_dir / f"{model['name']}-{context['tokens']}.txt"
            path.write_text(text)
            context["prompt_path"] = str(path.resolve())
            context["prompt_sha256"] = hashlib.sha256(text.encode()).hexdigest()
            context["calibrated_prompt_tokens"] = rendered_tokens(tokenizer, text, renderer)
            context["tokenizer_renderer"] = renderer
    atomic_json(args.output_manifest, manifest)


if __name__ == "__main__":
    main()
