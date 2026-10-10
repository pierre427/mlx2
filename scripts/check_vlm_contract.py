"""Reproduce a reviewed VLM contract from an independently identified source tree.

Default: verify only. --write --revision records an explicitly supplied full
reference revision after the caller has reviewed source provenance. Changed
dynamic dispatchers are refused until their exact-byte review is updated in
the manifest; this tool cannot approve unknown import dispatch.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlx2.source_dependencies import add_resources, source_closure


def rebuild(manifest, root):
    result = dict(manifest, files={}, families={})
    for family, contract in manifest["families"].items():
        files = source_closure(
            root, contract["roots"], reviewed_dynamic=contract["reviewed_dynamic"]
        )
        add_resources(root, files)
        result["files"].update(files)
        result["families"][family] = dict(contract, files=sorted(files))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="mlx_vlm package directory"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--revision")
    args = parser.parse_args()
    original = json.loads(args.manifest.read_text())
    result = rebuild(original, args.source)
    if args.write:
        if not args.revision or not re.fullmatch(r"[0-9a-f]{40}", args.revision):
            parser.error("--write needs an independently verified full --revision")
        result["source_revision"] = args.revision
        args.manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    elif result != original:
        raise SystemExit("source does not reproduce the reviewed manifest")
    print(
        json.dumps(
            {family: len(c["files"]) for family, c in result["families"].items()}
        )
    )


if __name__ == "__main__":
    main()
