"""Reviewed extra inputs for public desktop source snapshots.

The public source profile ships no private campaign fixtures. Public synthetic
tests remain available; missing external qualification evidence never establishes
qualification. Explicit future fixture entries are verified before copying.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

# The public profile has no extra campaign fixtures. Future additions require
# explicit path and content review; changed pins are never discovered implicitly.
PREFLIGHT_INPUT_SHA256 = {}

PREFLIGHT_INPUT_FILES = tuple(PREFLIGHT_INPUT_SHA256)


def preflight_inputs(root: Path):
    root = Path(root).resolve()
    for name, expected in PREFLIGHT_INPUT_SHA256.items():
        path = root / name
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Missing or escaped reviewed preflight input: {name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Reviewed preflight input changed; update its explicit pin after review: {name}")
        yield path
