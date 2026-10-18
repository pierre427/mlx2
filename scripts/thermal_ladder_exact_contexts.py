#!/usr/bin/env python3
"""Load a frozen B4 wrapper with exact contexts and an explicit width hook."""

from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_contexts(value: str) -> tuple[int, ...]:
    try:
        contexts = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise ValueError("MLX2_LADDER_CONTEXTS must be comma-separated integers") from error
    if not contexts or any(item < 1024 for item in contexts):
        raise ValueError("explicit ladder contexts must all be at least 1024")
    if tuple(sorted(set(contexts))) != contexts:
        raise ValueError("explicit ladder contexts must be unique and increasing")
    return contexts


def configure(
    base_wrapper: Path,
    thermal_module_path: Path,
    contexts: tuple[int, ...],
):
    base_wrapper = base_wrapper.expanduser().resolve(strict=True)
    thermal_module_path = thermal_module_path.expanduser().resolve(strict=True)
    campaign_root = Path(os.environ["MLX2_CAMPAIGN_ROOT"]).resolve(strict=True)
    series = campaign_root / "benchmark_results/2026-09-29/scripts/series"
    sys.path.insert(0, str(series.resolve(strict=True)))
    spec = importlib.util.spec_from_file_location("thermal_ladder", thermal_module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["thermal_ladder"] = module
    spec.loader.exec_module(module)
    module.LENGTHS = contexts
    module.EXPLICIT_WIDTHS = (4,)
    namespace = runpy.run_path(
        str(base_wrapper), run_name="mlx2_wrapped_thermal_ladder"
    )
    # The legacy wrapper assigned a fake dict solely to force B4. The staged
    # thermal module has an explicit width hook, so restore the real builtin.
    module.dict = builtins.dict
    assertions = {
        "loaded_module": str(Path(module.__file__).resolve()),
        "expected_module": str(thermal_module_path),
        "contexts": list(module.LENGTHS),
        "widths": list(module.EXPLICIT_WIDTHS),
        "declared_batch_installed": module.Stream.batch
        is namespace.get("declared_batch"),
        "builtin_dict_restored": module.dict is builtins.dict,
        "module_sha256": sha256(thermal_module_path),
        "base_wrapper_sha256": sha256(base_wrapper),
    }
    if (
        assertions["loaded_module"] != assertions["expected_module"]
        or tuple(module.LENGTHS) != contexts
        or tuple(module.EXPLICIT_WIDTHS) != (4,)
        or assertions["declared_batch_installed"] is not True
        or assertions["builtin_dict_restored"] is not True
    ):
        raise RuntimeError(f"exact-context module assertion failed: {assertions}")
    return module, assertions


def main() -> int:
    wrapper = os.environ.get("MLX2_THERMAL_LADDER_WRAPPER")
    module_path = os.environ.get("MLX2_THERMAL_LADDER_MODULE")
    contexts = os.environ.get("MLX2_LADDER_CONTEXTS")
    receipt = os.environ.get("MLX2_CONTEXT_OVERRIDE_RECEIPT")
    if not wrapper or not module_path or not contexts or not receipt:
        raise RuntimeError(
            "MLX2_THERMAL_LADDER_WRAPPER, MLX2_THERMAL_LADDER_MODULE, "
            "MLX2_LADDER_CONTEXTS and MLX2_CONTEXT_OVERRIDE_RECEIPT are required"
        )
    module, assertions = configure(
        Path(wrapper), Path(module_path), parse_contexts(contexts)
    )
    atomic_json(
        Path(receipt),
        {
            "schema": "mlx2.exact-context-override.v1",
            "passed": True,
            "provenance": "frozen canonical thermal ladder with explicit width hook",
            **assertions,
        },
    )
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
