"""Opt-in CLI for CPU weight-program construction and captured-array checks."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

from .asset_arrays import bounded_npz_arrays
from .evaluation import evaluate_projections, write_evaluation
from .programs import (
    MAX_ARTIFACT_BYTES,
    MAX_TOTAL_ELEMENTS,
    _array,
    _read_json,
    analytical_directional_repair,
    compose_programs,
    export_program,
    import_lora,
    inspect_program,
    load_program,
    truncate_program,
)


def _fields(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= set(value) or set(value) - set(required) - set(optional):
        raise ValueError(f"expected fields {sorted(required)} and optional {sorted(optional)}")
    return value


def _arrays(path):
    path = Path(path).expanduser()
    if not path.is_file() or path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError("NPZ input is absent or exceeds size bound")
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_ARTIFACT_BYTES + 1)
        source = bounded_npz_arrays(raw, maximum_bytes=MAX_ARTIFACT_BYTES)
        if sum(value.size for value in source.values()) > MAX_TOTAL_ELEMENTS:
            raise ValueError("NPZ tensors exceed aggregate element bound")
        return {key: _array(value, ndim=value.ndim) for key, value in source.items()}
    except (OSError, zipfile.BadZipFile, TypeError, EOFError, NotImplementedError) as exc:
        raise ValueError("invalid numeric NPZ archive") from exc


def _bound_args(parser):
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--provenance", required=True, help="nonempty provenance JSON object")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="verify hashes and inspect factors")
    inspect.add_argument("program")
    repair = commands.add_parser("repair", help="compile a directional repair from supplied CPU arrays")
    repair.add_argument("--spec", required=True)
    repair.add_argument("--arrays", required=True)
    repair.add_argument("--output", required=True)
    imported = commands.add_parser("import-lora", help="bind an existing mlx2-format LoRA artifact")
    imported.add_argument("directory")
    _bound_args(imported)
    imported.add_argument("--output", required=True)
    composed = commands.add_parser("compose", help="fixed factor concatenation, without rank truncation")
    composed.add_argument("programs", nargs="+")
    composed.add_argument("--scale", action="append", type=float)
    composed.add_argument("--provenance")
    composed.add_argument("--output", required=True)
    truncated = commands.add_parser("truncate", help="explicit approximate SVD rank reduction")
    truncated.add_argument("program")
    truncated.add_argument("--rank", type=int, required=True)
    truncated.add_argument("--provenance")
    truncated.add_argument("--output", required=True)
    memory = commands.add_parser("calibrate-memory", help="fit captured residual pairs; does not publish semantic memory")
    memory.add_argument("--spec", required=True)
    memory.add_argument("--arrays", required=True)
    memory.add_argument("--output", required=True)
    ablation = commands.add_parser("abliterate", help="compile contrastive CPU directions with explicit layer/component gates")
    ablation.add_argument("--spec", required=True)
    ablation.add_argument("--arrays", required=True)
    ablation.add_argument("--output", required=True)
    evaluation = commands.add_parser("evaluate-projections", help="CPU array comparison; no model qualification")
    evaluation.add_argument("program")
    evaluation.add_argument("--spec", required=True)
    evaluation.add_argument("--arrays", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--simulate-bfloat16", action="store_true")
    return result


def _main(args):
    if args.command == "inspect":
        return inspect_program(load_program(args.program))
    if args.command == "import-lora":
        program = import_lora(
            args.directory, checkpoint=args.checkpoint, source_revision=args.source_revision,
            provenance=_read_json(Path(args.provenance)),
        )
    elif args.command == "compose":
        program = compose_programs(
            [load_program(p) for p in args.programs], args.scale,
            provenance=_read_json(Path(args.provenance)) if args.provenance else None,
        )
    elif args.command == "truncate":
        program = truncate_program(
            load_program(args.program), args.rank,
            provenance=_read_json(Path(args.provenance)) if args.provenance else None,
        )
    elif args.command == "repair":
        spec = _fields(_read_json(Path(args.spec)), {"checkpoint", "source_revision", "provenance", "strength", "targets"})
        arrays = _arrays(args.arrays)
        if not isinstance(spec["targets"], dict) or not spec["targets"]:
            raise ValueError("repair targets must be a nonempty mapping")
        weights, directions, selected = {}, {}, {}
        for target, item in spec["targets"].items():
            _fields(item, {"weight", "direction", "selected_columns"})
            weights[target] = arrays[item["weight"]]
            directions[target] = arrays[item["direction"]]
            selected[target] = item["selected_columns"]
        program = analytical_directional_repair(
            weights, directions, selected, strength=spec["strength"], checkpoint=spec["checkpoint"],
            source_revision=spec["source_revision"], provenance=spec["provenance"],
        )
    elif args.command == "calibrate-memory":
        from .memory import (
            CalibrationSplit,
            CapturedActivations,
            MemorySource,
            calibrate_memory_program,
        )

        spec = _fields(_read_json(Path(args.spec)), {"source", "source_revision", "split", "captures"}, {"rank", "ridge", "strength"})
        _fields(spec["split"], {"calibration_ids", "heldout_ids"})
        if not isinstance(spec["captures"], list) or not 1 <= len(spec["captures"]) <= 32:
            raise ValueError("captures must list 1..32 targets")
        arrays = _arrays(args.arrays)
        captures = []
        for item in spec["captures"]:
            _fields(item, {"target", "sample_ids", "inputs", "target_deltas"})
            captures.append(CapturedActivations(
                item["target"], tuple(item["sample_ids"]), arrays[item["inputs"]], arrays[item["target_deltas"]],
            ))
        result = calibrate_memory_program(
            captures, split=CalibrationSplit(**spec["split"]), source=MemorySource.from_dict(spec["source"]),
            source_revision=spec["source_revision"], rank=spec.get("rank", 8),
            ridge=spec.get("ridge", 1e-4), strength=spec.get("strength", 1),
        )
        program = result.program
    elif args.command == "abliterate":
        from .heretic import LayerProfile, compile_abliteration, estimate_direction

        spec = _fields(
            _read_json(Path(args.spec)),
            {"checkpoint", "source_revision", "provenance", "profiles", "targets"},
            {"orthogonalize_reference", "winsorization_quantile", "row_normalization", "full_rank"},
        )
        if not isinstance(spec["profiles"], dict) or not isinstance(spec["targets"], dict):
            raise ValueError("profiles and targets must be explicit mappings")
        profiles = {}
        for component, profile in spec["profiles"].items():
            _fields(profile, {"peak_layer", "radius"}, {"peak_strength", "edge_strength"})
            profiles[component] = LayerProfile(**profile)
        arrays = _arrays(args.arrays)
        weights, directions, targets = {}, {}, {}
        for target, item in spec["targets"].items():
            _fields(item, {"weight", "reference", "contrast", "layer", "component"})
            weights[target] = arrays[item["weight"]]
            directions[target] = estimate_direction(
                arrays[item["reference"]], arrays[item["contrast"]],
                orthogonalize_reference=spec.get("orthogonalize_reference", False),
                winsorization_quantile=spec.get("winsorization_quantile", 1),
            )
            targets[target] = {"layer": item["layer"], "component": item["component"]}
        program = compile_abliteration(
            weights, directions, targets, profiles=profiles, checkpoint=spec["checkpoint"],
            source_revision=spec["source_revision"], provenance=spec["provenance"],
            row_normalization=spec.get("row_normalization", "none"), full_rank=spec.get("full_rank"),
        )
    elif args.command == "evaluate-projections":
        spec = _fields(_read_json(Path(args.spec)), {"targets"})
        if not isinstance(spec["targets"], dict):
            raise ValueError("projection targets must be a mapping")
        arrays = _arrays(args.arrays)
        weights, inputs = {}, {}
        for target, item in spec["targets"].items():
            _fields(item, {"weight", "inputs"})
            weights[target], inputs[target] = arrays[item["weight"]], arrays[item["inputs"]]
        report = evaluate_projections(
            load_program(args.program), weights, inputs, simulate_bfloat16=args.simulate_bfloat16,
        )
        path = write_evaluation(report, args.output)
        return {"receipt": str(path), **report}
    else:
        raise ValueError("unsupported command")
    path = export_program(program, args.output)
    return {"artifact": str(path), **inspect_program(program)}


def main(argv=None):
    arguments = parser().parse_args(argv)
    try:
        result = _main(arguments)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print(json.dumps({"error": str(exc), "command": arguments.command}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0
