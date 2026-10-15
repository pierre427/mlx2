#!/usr/bin/env python3
"""Evaluate a bounded matched-arm document without loading MLX or a model."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from mlx2.runtime.deliberation_evaluation import evaluate_matched_arms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite matched evaluation evidence")
    document = json.loads(args.input.read_text())
    report = evaluate_matched_arms(document)
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=args.output.parent,
            prefix=f".{args.output.name}.",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            os.chmod(temporary, 0o600)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link publishes the complete file atomically and refuses to
        # replace a path created after the initial existence check.
        os.link(temporary, args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return 0 if report["matched_quality_contrast_reportable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
