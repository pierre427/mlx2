#!/usr/bin/env python3
"""Bounded direct Agnes image-head smoke; preflight never loads model weights."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("~/mlx-models/Agnes-3.0-Flash-Preview-MLX-6bit")
FORK = Path("~/Desktop/mlx-uag/worktrees/agnes-vlm-support")
PROMPT = "What color is the square on the right? Answer with one word."


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
    ).strip()


def _fixture(path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (512, 320), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((40, 80, 200, 240), fill=(220, 20, 30))
    draw.rectangle((312, 80, 472, 240), fill=(20, 60, 220))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _write(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(row, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--fork", type=Path, default=FORK)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--cpg-lease")
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 32:
        parser.error("--max-tokens must be 1..32")
    if not args.preflight and not args.cpg_lease:
        parser.error("--cpg-lease is required for the GPU smoke")

    model = args.model.expanduser().resolve()
    fork = args.fork.expanduser().resolve()
    image = args.image.expanduser().resolve()
    row = {
        "schema": "mlx2.agnes-direct-vision-head.v1",
        "status": "failed", "scope": "CPU artifact preflight" if args.preflight else "direct image prefill and bounded greedy decode",
        "model_path": str(model), "fork_path": str(fork), "image_path": str(image),
        "prompt": PROMPT, "max_tokens": args.max_tokens, "cpg_lease": args.cpg_lease,
        "source_revision": _head(ROOT), "fork_revision": _head(fork),
        "runner_sha256": _sha(Path(__file__)),
        "adapter_sha256": _sha(ROOT / "src/mlx2/adapters/agnes_vision.py"),
        "candidate_sha256": _sha(ROOT / "src/mlx2/adapters/pinned_vlm_candidate.py"),
        "started_at": time.time(),
    }
    try:
        from mlx2.adapters.agnes_vision import (
            AgnesVisionCandidateAdapter,
            inspect_artifact,
        )
        from mlx2.adapters.mlx_vlm_pin import mlx_vlm_runtime
        from mlx2.adapters.pinned_vlm_candidate import SOURCE_REVISION

        if not image.exists():
            _fixture(image)
        row["image_sha256"] = _sha(image)
        row["mlx_version"] = importlib.metadata.version("mlx")
        runtime = mlx_vlm_runtime()
        row["mlx_vlm_runtime"] = runtime
        if row["fork_revision"] != SOURCE_REVISION or runtime is None or runtime["revision"] != SOURCE_REVISION:
            raise RuntimeError(f"Agnes requires pinned mlx-vlm {SOURCE_REVISION}")
        from importlib.util import find_spec

        spec = find_spec("mlx_vlm")
        if spec is None or spec.origin is None or not Path(spec.origin).resolve().is_relative_to(fork):
            raise RuntimeError("mlx_vlm import does not resolve to the pinned fork")
        row["mlx_vlm_module"] = spec.origin
        artifact = inspect_artifact(model)
        row["artifact_fingerprint"] = artifact["fingerprint"]
        row["tensor_count"] = artifact["tensor_count"]
        row["qualification"] = artifact["qualification"]
        if args.preflight:
            row["status"] = "preflight_passed"
            return 0

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import mlx.core as mx

        adapter = AgnesVisionCandidateAdapter(str(model))
        started = time.perf_counter()
        try:
            # Agnes defaults to xhigh thinking and opens an unfinished <think>
            # block. The checkpoint template explicitly supports this direct
            # answer mode, so pin it for the bounded visible-output smoke.
            apply_template = adapter.processor.apply_chat_template
            rendered_templates = []

            def direct_answer_template(*values, **options):
                options["enable_thinking"] = False
                rendered = apply_template(*values, **options)
                rendered_templates.append(rendered)
                return rendered

            adapter.processor.apply_chat_template = direct_answer_template
            request = {"messages": [{"role": "user", "content": [
                {"type": "input_image", "image_url": "data:image/png;base64," +
                 base64.b64encode(image.read_bytes()).decode("ascii")},
                {"type": "text", "text": PROMPT},
            ]}]}
            try:
                prepared = adapter.prepare_multimodal_request(request)
            finally:
                adapter.processor.apply_chat_template = apply_template
            if len(rendered_templates) != 1 or not rendered_templates[0].endswith(
                "<think>\n\n</think>\n\n"
            ):
                raise RuntimeError("Agnes direct-answer chat template was not applied")
            row["chat_template_kwargs"] = {"enable_thinking": False}
            row["template_suffix"] = rendered_templates[0][-24:]
            if not adapter.has_trusted_media_preparation(prepared):
                raise RuntimeError("Agnes media preparation proof is invalid")
            ids = adapter.prompt_tokens(prepared)
            media_end = prepared["_mlx2_media_token_end"]
            if not 0 < media_end <= len(ids):
                raise RuntimeError("Agnes image placeholders are missing")
            cache = adapter.model.make_cache()
            logits = adapter.model(
                mx.array([ids], dtype=mx.int32), cache=cache,
                **prepared["_mlx2_prefill_inputs"],
            )
            mx.eval(logits)
            row["prompt_tokens"] = len(ids)
            row["media_token_end"] = media_end
            row["cache_layers"] = len(cache)
            row["prefill_logits_shape"] = list(logits.shape)
            if tuple(logits.shape[:2]) != (1, len(ids)):
                raise RuntimeError("Agnes image prefill logits shape is wrong")
            generated = []
            eos = set(adapter._eos_ids())
            for step in range(args.max_tokens):
                token = int(mx.argmax(logits[0, -1]).item())
                if token in eos:
                    row["stop_reason"] = "eos"
                    break
                generated.append(token)
                if step + 1 < args.max_tokens:
                    logits = adapter.model(mx.array([[token]], dtype=mx.int32), cache=cache)
                    mx.eval(logits)
            else:
                row["stop_reason"] = "token_cap"
            text = adapter.processor.tokenizer.decode(generated, skip_special_tokens=True)
            visible = text
            if "<think>" in text:
                visible = text.rsplit("</think>", 1)[1] if "</think>" in text else ""
            row["generated_tokens"] = len(generated)
            row["text"] = text
            row["visible_text"] = visible
            row["decode_logits_shape"] = list(logits.shape)
            if (not visible.strip() or "blue" not in visible.lower()
                    or len(visible.split()) > 12):
                raise RuntimeError("Agnes did not visibly identify the blue square")
            row["status"] = "smoke_passed"
            return 0
        finally:
            row["model_elapsed_seconds"] = time.perf_counter() - started
            adapter.close()
    except Exception as error:  # noqa: BLE001 - persist every smoke failure in the receipt
        row["error_type"] = type(error).__name__
        row["error"] = str(error)
        row["traceback"] = traceback.format_exc(limit=8)
        return 1
    finally:
        row["finished_at"] = time.time()
        _write(args.out, row)


if __name__ == "__main__":
    raise SystemExit(run())
