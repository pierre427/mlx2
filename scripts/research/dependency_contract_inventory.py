"""Inspect mechanism/source contracts without importing MLX or loading models."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def checked_path(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("source path must be nonempty")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("missing or escaping contract path: " + relative)
    return path


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def known_builds(root):
    tree = ast.parse((root / "src/mlx2/runtime/mlx_build.py").read_text())
    node = next(
        n.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "VERIFIED_MLX_BUILDS" for t in n.targets
        )
    )
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "frozenset"
        and len(node.args) == 1
    ):
        raise ValueError("unrecognized reviewed build declaration")
    builds = ast.literal_eval(node.args[0])
    if not builds or not all(isinstance(x, str) for x in builds):
        raise ValueError("invalid build declaration")
    return sorted(builds)


def inventory(root, catalog, installed_build):
    root = Path(root).resolve()
    if catalog.get("schema") != 1:
        raise ValueError("unsupported contract inventory schema")
    builds = known_builds(root)
    mechanisms = {}
    for name, spec in catalog["mechanisms"].items():
        sources = {p: file_sha(checked_path(root, p)) for p in spec["sources"]}
        tests = {}
        for relative in spec["tests"]:
            path = checked_path(root, relative)
            tree = ast.parse(path.read_text())
            tests[relative] = {
                "sha256": file_sha(path),
                "test_functions": sorted(
                    {
                        n.name
                        for n in ast.walk(tree)
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and n.name.startswith("test_")
                    }
                ),
                "executed_by_inventory": False,
            }
        mechanisms[name] = {
            **spec,
            "source_sha256": sources,
            "test_inventory": tests,
            "compatibility_established_by_this_report": False,
            "hardware_canaries_run": False,
        }
    return {
        "schema": "mlx2.dependency-contract-inventory.v1",
        "scope": "source inventory only; does not qualify builds or authorize optimized capabilities",
        "installed_build_metadata": installed_build,
        "historical_build_list_membership": installed_build in builds,
        "historical_build_list": builds,
        "framework_imported": False,
        "package_constraints_changed": False,
        "mechanisms": mechanisms,
        "required_evidence_axes": [
            "exact package/native-library build",
            "mechanism source and custom-kernel ABI",
            "device family, dtype, bit width and shape",
            "ordinary-reference output/state comparison",
            "selected policy and observed execution",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path(__file__).with_name("mlx_mechanism_contracts.json"),
    )
    parser.add_argument(
        "--build", help="metadata label for an offline compatibility review"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refuse to overwrite evidence")
    build = args.build
    if build is None:
        try:
            build = importlib.metadata.version("mlx")
        except importlib.metadata.PackageNotFoundError:
            build = "not-installed"
    report = inventory(args.root, json.loads(args.catalog.read_text()), build)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "mechanisms": len(report["mechanisms"]),
                "build": build,
                "compatibility_established": False,
            }
        )
    )


if __name__ == "__main__":
    main()
