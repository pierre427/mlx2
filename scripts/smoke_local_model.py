#!/usr/bin/env python3
"""One bounded, source-bound candidate smoke. Run each model in a fresh process.

This is direct ordinary text generation, not serving qualification. The caller
owns the CPG GPU lease and host lock; this script never changes services.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DEFAULT_PROMPT = "Say hello in one sentence."


def _git(args: list[str]) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _identity(resolution) -> dict:
    artifact = resolution.artifact
    identity = artifact.get("identity", artifact)
    return {
        "fingerprint": identity.get("fingerprint"),
        "files": identity.get("files"),
        "has_mtp": artifact.get("has_mtp"),
    }


def _ordinary_generate(adapter, prompt: str, max_tokens: int, prompt_mode: str = "auto") -> dict:
    import mlx.core as mx
    from mlx2.runtime.models.cache import make_prompt_cache

    tokenizer = adapter.tokenizer
    request = {"prompt": prompt}
    prompt_format = "plain"
    if prompt_mode != "plain" and getattr(tokenizer, "has_chat_template", False):
        chat = {"messages": [{"role": "user", "content": prompt}], "enable_thinking": False}
        try:
            tokens = list(adapter.prompt_tokens(chat))
            prompt_format = "chat_no_thinking"
        except ValueError:
            # Some candidate adapters explicitly support only plain prompts.
            tokens = list(adapter.prompt_tokens(request))
    else:
        tokens = list(adapter.prompt_tokens(request))
    if not tokens:
        raise ValueError("prompt tokenized to an empty sequence")
    if len(tokens) > 128:
        raise ValueError("smoke prompt exceeds 128 tokens")
    cache = make_prompt_cache(adapter.model)
    stop_ids = set(getattr(tokenizer, "eos_token_ids", ()) or ())
    encoded = []
    stop_reason = "token_cap"
    started = time.perf_counter()
    for step in range(max_tokens):
        model_input = tokens if step == 0 else [encoded[-1]]
        logits = adapter.model(mx.array(model_input)[None], cache=cache)
        if not isinstance(logits, mx.array) or logits.ndim != 3 or logits.shape[0] != 1:
            raise ValueError("ordinary model did not return [1, sequence, vocab] logits")
        row = logits[0, -1]
        finite = mx.all(mx.isfinite(row))
        token = mx.argmax(row)
        mx.eval(finite, token)
        if not bool(finite.item()):
            raise ValueError(f"nonfinite logits at generated step {step}")
        token_id = int(token.item())
        if not 0 <= token_id < int(row.shape[-1]):
            raise ValueError(f"sampled token {token_id} outside model output range")
        if token_id in stop_ids:
            stop_reason = "eos"
            break
        encoded.append(token_id)
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    if not isinstance(decoded, str) or not decoded.strip() or not any(ch.isprintable() for ch in decoded):
        raise ValueError("no viable visible output within bounded token cap")
    if decoded.strip().casefold() == prompt.strip().casefold():
        raise ValueError("model only echoed the smoke prompt")
    return {
        "prompt_tokens": len(tokens), "prompt_format": prompt_format,
        "token_ids": encoded,
        "text": decoded, "stop_reason": stop_reason,
        "decode_seconds": time.perf_counter() - started,
        "model_vocab": int(row.shape[-1]),
    }


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument("--prompt-mode", choices=("auto", "plain"), default="auto")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument(
        "--adapter-class", help="direct candidate as mlx2.adapters.module:Class",
    )
    parser.add_argument("--i-own-gpu", action="store_true")
    parser.add_argument("--cpg-lease", help="CPG lease id recorded in the receipt")
    args = parser.parse_args(argv)
    if not 1 <= args.max_tokens <= 16:
        parser.error("--max-tokens must be 1..16")
    if not args.inspect_only and (not args.i_own_gpu or not args.cpg_lease):
        parser.error("generation requires --i-own-gpu and --cpg-lease under host lock")
    path = Path(args.model).expanduser().resolve()
    record = {
        "schema": "mlx2.local-model-smoke.v1", "status": "failed",
        "model_path": str(path), "source_revision": _git(["rev-parse", "HEAD"]),
        "source_dirty": bool(_git(["status", "--porcelain", "--untracked-files=no"])),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "platform": platform.platform(), "started_at": time.time(),
        "generation_requested": not args.inspect_only,
        "cpg_lease": args.cpg_lease, "prompt": args.prompt,
        "prompt_mode": args.prompt_mode,
        "max_tokens": args.max_tokens,
    }
    try:
        if args.adapter_class:
            module_name, separator, class_name = args.adapter_class.partition(":")
            if not separator or not module_name.startswith("mlx2.adapters."):
                raise ValueError("--adapter-class must be mlx2.adapters.module:Class")
            module = importlib.import_module(module_name)
            adapter_type = getattr(module, class_name)
            inspector = getattr(adapter_type, "artifact_inspector", None)
            if inspector is None:
                inspector = getattr(module, "inspect_artifact")
            resolution = SimpleNamespace(
                adapter_type=adapter_type, descriptor=adapter_type.descriptor,
                artifact=inspector(path),
            )
        else:
            from mlx2.adapters.registry import inspect_model

            resolution = inspect_model(path)
        record["adapter"] = (
            resolution.adapter_type.__module__ + "." + resolution.adapter_type.__name__
        )
        record["family"] = resolution.descriptor.family
        record["artifact"] = _identity(resolution)
        record["status"] = "preflight_passed"
        if args.inspect_only:
            return 0
        from mlx2.contracts import Capability

        if Capability.VISION in resolution.descriptor.capabilities or resolution.descriptor.model_type in {
            "llada", "diffusion_gemma", "nemotron3_diarization",
        }:
            record["status"] = "unsupported_direct_generation"
            record["reason"] = (
                "candidate needs a separate image, denoising, or speech smoke; "
                "no model load or output was attempted"
            )
            return 2
        import mlx.core as mx

        record["mlx_version"] = importlib.metadata.version("mlx")
        record["device"] = str(mx.default_device())
        loaded_at = time.perf_counter()
        adapter = resolution.adapter_type(str(path))
        record["load_seconds"] = time.perf_counter() - loaded_at
        try:
            if not all(hasattr(adapter, name) for name in ("model", "tokenizer", "prompt_tokens")):
                record["status"] = "unsupported_direct_generation"
                record["reason"] = "adapter requires a separate media, denoising, or speech smoke"
                return 2
            record["output"] = _ordinary_generate(adapter, args.prompt, args.max_tokens, args.prompt_mode)
            record["status"] = "smoke_passed"
            return 0
        finally:
            close = getattr(adapter, "close", None)
            if callable(close):
                close()
    except Exception as error:
        record["status"] = "failed"
        record["error_type"] = type(error).__name__
        record["error"] = str(error)
        record["traceback"] = traceback.format_exc(limit=8)
        return 1
    finally:
        record["finished_at"] = time.time()
        record["elapsed_seconds"] = record["finished_at"] - record["started_at"]
        _write(Path(args.out), record)


if __name__ == "__main__":
    raise SystemExit(run())
