#!/usr/bin/env python3
"""Run one receipt-bound MOSS-Image inspection, load, or generation probe.

GPU execution requires an externally owned CPG/radio lease and paired host
locks. This focused probe never promotes qualification, selection, or defaults.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

from qualification_gpu_ownership import require_qualification_lease

ROOT = Path(__file__).resolve().parents[1]


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _command(*args):
    return subprocess.check_output(args, text=True).strip()


def _environment():
    from mlx2.system_monitor import thermal_state

    return {
        "time": time.time(),
        "thermal_state": thermal_state(),
        "swap": _command("sysctl", "-n", "vm.swapusage"),
    }


def run(args):
    from mlx2.adapters.moss_image import MOSSImageSFTAdapter

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("evidence directory must be empty; choose a new attempt")
    receipt = {
        "kind": "focused_functional_smoke",
        "qualified": False,
        "selected": False,
        "host": platform.node(),
        "source_revision": _command("git", "-C", str(ROOT), "rev-parse", "HEAD"),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__),
                *sorted((ROOT / "src/mlx2/adapters").glob("moss_image*.py")),
            ]
        },
        "device": args.device,
        "parameters": {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "seed": args.seed,
            "guidance_scale": args.guidance_scale,
            "uncond_skip_layers": args.uncond_skip_layers,
        },
    }
    try:
        adapter = MOSSImageSFTAdapter(args.model)
        receipt["route"] = adapter.status()
        if args.inspect_only:
            receipt["success"] = True
            return receipt
        if args.device == "gpu":
            owner = json.loads((args.cpg_owner_lock / "owner.json").read_text())
            receipt["ownership"] = require_qualification_lease(
                task_id=owner["cpg_task"],
                generation=owner["cpg_generation"],
                cpg_owner_lock=args.cpg_owner_lock,
            )
        receipt["environment_before"] = _environment()
        if args.device == "gpu" and receipt["environment_before"][
            "thermal_state"
        ] not in (
            "nominal",
            "fair",
        ):
            raise RuntimeError("thermal gate is not ready; release GPU and retry")
        import mlx.core as mx

        mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
        from mlx2.adapters.moss_image_pipeline import MOSSImagePipeline

        started = time.perf_counter()
        adapter._backend = MOSSImagePipeline(adapter.artifact.path)
        receipt["load_seconds"] = time.perf_counter() - started
        receipt["text_runtime"] = adapter._backend.text_encoder.runtime
        receipt["weights_loaded_strictly"] = True
        if not args.load_only:
            started = time.perf_counter()
            result = adapter.generate_image(
                args.prompt,
                negative_prompt=args.negative_prompt,
                width=args.width,
                height=args.height,
                steps=args.steps,
                seed=args.seed,
                guidance_scale=args.guidance_scale,
                uncond_skip_layers=tuple(args.uncond_skip_layers),
            )
            receipt["generation_seconds"] = time.perf_counter() - started
            temporary = output / "image.png.tmp"
            temporary.write_bytes(result.data)
            os.replace(temporary, output / "image.png")
            receipt["route"] = adapter.last_route_receipt
            receipt["image"] = {
                key: value
                for key, value in dataclasses.asdict(result).items()
                if key != "data"
            }
        receipt["mlx_peak_memory_bytes"] = mx.get_peak_memory()
        receipt["environment_after"] = _environment()
        receipt["success"] = True
        return receipt
    except Exception as exc:
        receipt.update(success=False, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _atomic_json(output / "receipt.json", receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="A red ceramic teapot on a wooden table.")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--uncond-skip-layers", type=int, nargs="*", default=[22])
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument(
        "--cpg-owner-lock", type=Path, default=Path("/tmp/mlx2-moss-image-cpg.lock")
    )
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))
