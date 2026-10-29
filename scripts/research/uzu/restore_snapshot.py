"""Restore historical experiment sources into a clean checkout at their base.

This prepares source only. It never loads a model, claims a GPU, or runs a test.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def restore(snapshot: Path, destination: Path):
    receipt = json.loads((snapshot / "result.json").read_text())
    revision = receipt["source_revision"]
    head = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    if head != revision:
        raise ValueError(f"destination must be at the experiment base {revision}")
    dirty = subprocess.check_output(["git", "-C", str(destination), "status", "--porcelain"], text=True)
    if dirty:
        raise ValueError("destination must be a clean, separate checkout")
    files = {}
    for name, digest in receipt["source_files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("snapshot paths must be repository-relative")
        data = (snapshot / "source" / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"snapshot hash mismatch: {name}")
        files[relative] = data
    # Historical harness imports use these names at scripts/ in the old tree.
    for name in ("bench_uzu_candidates.py", "probe_uzu_checkpoint.py", "probe_uzu_kernels.py"):
        files.setdefault(Path("scripts") / name, (Path(__file__).parent / name).read_bytes())
    files[Path("tests/test_uzu_mlp_tiling.py")] = (Path(__file__).parent / "test_historical_integration.py").read_bytes()
    for relative, data in files.items():
        target = destination / relative
        if target.is_symlink() or any(parent.is_symlink() for parent in target.parents if parent != destination.parent):
            raise ValueError(f"refusing symlink destination: {relative}")
    for relative, data in files.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return {"source_revision": revision, "restored": len(files), "gpu_execution": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(restore(args.snapshot.resolve(), args.destination.resolve()), indent=2))


if __name__ == "__main__":
    main()
