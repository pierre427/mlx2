#!/usr/bin/env python3
"""Run one artifact-bound FLUX 3 direct generation/edit smoke or CPU load check.

GPU execution requires the existing CPG/radio lease plus paired lock wrapper.
This is a focused functional probe; it never promotes qualification/defaults.
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


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def environment():
    from mlx2.apple_telemetry import TemperatureSampler
    from mlx2.system_monitor import thermal_state

    sample = {
        "time": time.time(),
        "thermal_state": thermal_state(),
        "swap": command("sysctl", "-n", "vm.swapusage"),
        "temperature": None,
    }
    sampler = None
    try:
        sampler = TemperatureSampler()
        sample["temperature"] = sampler.die_summary()
    except Exception as exc:  # noqa: BLE001 - thermal telemetry is optional
        sample["temperature_error"] = str(exc)
    finally:
        if sampler is not None:
            sampler.close()
    return sample


def run(args):
    from mlx2.adapters.flux3_image import Flux3ImageKleinAdapter

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("evidence directory must be empty; choose a new attempt")
    receipt = {
        "kind": "focused_functional_smoke",
        "qualified": False,
        "selected": False,
        "host": platform.node(),
        "source_revision": command("git", "-C", str(ROOT), "rev-parse", "HEAD"),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                *sorted((ROOT / "src/mlx2/adapters").glob("flux3_*.py")),
            ]
        },
        "parameters": {
            "prompt": args.prompt,
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "seed": args.seed,
        },
        "device": args.device,
    }
    try:
        started = time.perf_counter()
        adapter = Flux3ImageKleinAdapter(args.model)
        receipt["artifact_verification_seconds"] = time.perf_counter() - started
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
        receipt["environment_before"] = environment()
        if args.device == "gpu":
            env = receipt["environment_before"]
            maximum = (env["temperature"] or {}).get("die_max_c")
            if env["thermal_state"] not in ("nominal", "fair") or (
                maximum is not None and maximum > 85
            ):
                raise RuntimeError(
                    "thermal gate is not ready; release GPU and retry in a new window"
                )
        import mlx.core as mx

        mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
        from mlx2.adapters.flux3_pipeline import Flux3Pipeline

        started = time.perf_counter()
        adapter._backend = Flux3Pipeline(adapter.artifact.path)
        receipt["load_seconds"] = time.perf_counter() - started
        receipt["text_runtime"] = adapter._backend.text_encoder.runtime
        receipt["weights_loaded_strictly"] = True
        if not args.load_only:
            started = time.perf_counter()
            kwargs = {
                "width": args.width,
                "height": args.height,
                "steps": args.steps,
                "seed": args.seed,
            }
            result = (
                adapter.edit_image(args.prompt, args.reference, **kwargs)
                if args.reference
                else adapter.generate_image(args.prompt, **kwargs)
            )
            receipt["generation_seconds"] = time.perf_counter() - started
            temporary = output / "image.png.tmp"
            temporary.write_bytes(result.data)
            os.replace(temporary, output / "image.png")
            receipt["route"] = adapter.last_route_receipt
            receipt["image"] = {
                k: v for k, v in dataclasses.asdict(result).items() if k != "data"
            }
        receipt["mlx_peak_memory_bytes"] = mx.get_peak_memory()
        receipt["environment_after"] = environment()
        receipt["success"] = True
        return receipt
    except Exception as exc:
        receipt.update(success=False, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        atomic_json(output / "receipt.json", receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="new/empty evidence directory"
    )
    parser.add_argument(
        "--prompt",
        default="A small red ceramic teapot on a wooden table, soft daylight, product photograph.",
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reference", type=Path, action="append", default=[])
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument(
        "--cpg-owner-lock", type=Path, default=Path("/tmp/mlx2-flux3-cpg.lock")
    )
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))
