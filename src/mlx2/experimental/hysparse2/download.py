"""Download explicitly pinned public HF files and verify their complete hashes."""

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_file(repo, revision, item, root, *, open_url=urllib.request.urlopen):
    path = PurePosixPath(item["path"])
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) or not re.fullmatch(
        r"[0-9a-f]{40}", revision
    ):
        raise ValueError("a dataset ID and exact 40-character revision are required")
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("unsafe dataset path")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        or type(item["size"]) is not int
        or item["size"] < 1
    ):
        raise ValueError("expected SHA256 and positive size required")
    target = Path(root) / repo.replace("/", "--") / revision / path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != item["size"] or sha256(target) != item["sha256"]:
            raise ValueError(f"existing file failed verification: {target}")
        return {
            "path": str(target),
            "size": item["size"],
            "sha256": item["sha256"],
            "status": "verified-existing",
        }
    partial = target.with_name(target.name + ".partial")
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{urllib.parse.quote(str(path))}"
    h = hashlib.sha256()
    size = 0
    # Exclusive creation avoids two downloaders trampling the same partial file.
    with partial.open("xb") as f, open_url(url, timeout=60) as response:
        while block := response.read(8 << 20):
            size += len(block)
            if size > item["size"]:
                raise ValueError("download exceeded expected size; partial retained")
            f.write(block)
            h.update(block)
    if size != item["size"] or h.hexdigest() != item["sha256"]:
        raise ValueError("download hash/size mismatch; partial retained")
    partial.rename(target)
    return {
        "path": str(target),
        "size": size,
        "sha256": h.hexdigest(),
        "status": "downloaded-verified",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    data = json.loads(args.manifest.read_text())
    for source in data["sources"]:
        results = []
        for item in source["files"]:
            result = fetch_file(source["id"], source["revision"], item, args.output)
            results.append(result)
            print(json.dumps(result), flush=True)
        receipt = {
            **source,
            "schema": "mlx2.verified-public-dataset.v1",
            "verified_files": results,
        }
        path = (
            args.output
            / source["id"].replace("/", "--")
            / source["revision"]
            / "receipt.json"
        )
        path.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
