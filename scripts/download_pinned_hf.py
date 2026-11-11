#!/usr/bin/env python3
"""Acquire pinned Hugging Face snapshots with exactly five download workers.

`--pin` resolves each repo's current revision and every file's content hash
from the Hub API into a manifest; `download` then fetches that exact revision
and verifies every file (LFS sha256, else git blob sha1) before writing an
atomic `.hf-download-complete.json` receipt in the output directory.

The PAT is read in-process from ~/.hf_pr; it is never placed in argv, URLs,
manifests or receipts.
"""
import argparse
import fcntl
import hashlib
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WORKERS = 5


def _token(token_file):
    return Path(token_file).expanduser().read_text().strip()


def pin(repo, *, token_file):
    request = urllib.request.Request(
        f"https://huggingface.co/api/models/{repo}?blobs=true",
        headers={"Authorization": f"Bearer {_token(token_file)}"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        info = json.load(response)
    files = []
    for sibling in info["siblings"]:
        lfs = sibling.get("lfs") or {}
        files.append({
            "path": sibling["rfilename"],
            "size": lfs.get("size", sibling.get("size")),
            "sha256": lfs.get("sha256"),
            "git_blob": sibling.get("blobId"),
        })
    return {"repo": repo, "revision": info["sha"], "files": files,
            "bytes": sum(f["size"] or 0 for f in files)}


def _verify(output, entry):
    path = output / entry["path"]
    if not path.is_file() or path.stat().st_size != entry["size"]:
        raise ValueError(f"missing/incomplete snapshot file: {entry['path']}")
    digest = hashlib.sha256() if entry["sha256"] else hashlib.sha1()
    if not entry["sha256"]:
        digest.update(f"blob {entry['size']}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != (entry["sha256"] or entry["git_blob"]):
        raise ValueError(f"snapshot content hash differs: {entry['path']}")
    return {"path": entry["path"], "bytes": entry["size"], "content_hash": digest.hexdigest()}


def download(snapshot, output, *, token_file, workers=WORKERS):
    if workers != WORKERS:
        raise ValueError("pinned acquisition uses exactly five workers")
    from huggingface_hub import snapshot_download

    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".mlx2-download.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot_download(snapshot["repo"], revision=snapshot["revision"], local_dir=output,
                          token=_token(token_file), max_workers=WORKERS)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            verified = list(pool.map(lambda e: _verify(output, e), snapshot["files"]))
        receipt = {"repo": snapshot["repo"], "revision": snapshot["revision"],
                   "workers": WORKERS, "bytes": snapshot["bytes"], "verified": verified}
        target = output / ".hf-download-complete.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        os.replace(temporary, target)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, default=Path.home() / ".hf_pr")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--pin", nargs="*", metavar="REPO=DIR",
                        help="resolve current revisions into --manifest, then exit")
    parser.add_argument("--only", nargs="*", help="download only these repos")
    args = parser.parse_args()
    if args.pin is not None:
        snapshots = []
        for item in args.pin:
            repo, directory = item.split("=", 1)
            snapshot = pin(repo, token_file=args.token_file)
            snapshot["output"] = directory
            snapshots.append(snapshot)
        args.manifest.write_text(json.dumps(
            {"schema": "mlx2.pinned-hf-snapshots.v1", "snapshots": snapshots}, indent=2) + "\n")
        for s in snapshots:
            print(s["repo"], s["revision"], round(s["bytes"] / 2**30, 2), "GiB")
        return
    manifest = json.loads(args.manifest.read_text())
    for snapshot in manifest["snapshots"]:
        if args.only and snapshot["repo"] not in args.only:
            continue
        receipt = download(snapshot, Path(snapshot["output"]).expanduser(),
                           token_file=args.token_file, workers=args.workers)
        print(json.dumps({k: v for k, v in receipt.items() if k != "verified"}), flush=True)


if __name__ == "__main__":
    main()
