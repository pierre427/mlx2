#!/usr/bin/env python3
"""Enable Metal Shader Timeline in an Instruments trace template.

Xcode's stock ``Metal System Trace`` template records GPU encoder intervals but
leaves the shader timeline disabled.  This helper changes only the archived
``shaderprofiler`` switches and writes a task-local template for ``xctrace``.
"""

from __future__ import annotations

import argparse
import plistlib
from pathlib import Path

DEFAULT_TEMPLATE = Path(
    "/Applications/Xcode.app/Contents/Applications/Instruments.app/Contents/"
    "Packages/GPU.instrdst/Contents/Templates/Metal System Trace.tracetemplate"
)


def _uid_value(value: object) -> int | None:
    return value.data if isinstance(value, plistlib.UID) else None


def enable_shader_timeline(payload: dict[str, object]) -> int:
    objects = payload.get("$objects")
    if not isinstance(objects, list):
        raise TypeError("not an NSKeyedArchiver template: missing $objects")

    true_indexes = [index for index, value in enumerate(objects) if value is True]
    if len(true_indexes) != 1:
        raise ValueError(f"expected one archived true value, found {true_indexes}")
    true_ref = plistlib.UID(true_indexes[0])

    changed = 0
    for value in objects:
        if not isinstance(value, dict):
            continue
        key_refs = value.get("NS.keys")
        object_refs = value.get("NS.objects")
        if not isinstance(key_refs, list) or not isinstance(object_refs, list):
            continue
        if len(key_refs) != len(object_refs):
            raise ValueError("archived dictionary has mismatched keys and values")
        for offset, key_ref in enumerate(key_refs):
            key_index = _uid_value(key_ref)
            if key_index is None or key_index >= len(objects):
                continue
            if objects[key_index] not in {"shaderprofiler", "shaderprofilerinternal"}:
                continue
            if object_refs[offset] != true_ref:
                object_refs[offset] = true_ref
                changed += 1
    if changed != 2:
        raise ValueError(f"expected to enable two shader profiler switches, changed {changed}")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open("rb") as handle:
        payload = plistlib.load(handle)
    changed = enable_shader_timeline(payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        plistlib.dump(payload, handle, fmt=plistlib.FMT_BINARY, sort_keys=False)
    print(f"enabled {changed} shader timeline switches in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
