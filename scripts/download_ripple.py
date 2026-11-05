#!/usr/bin/env python3
"""Acquire the pinned Ripple NVFP4 artifact with exactly five download workers.

The PAT is read in-process from ~/.hf_pr; never placed in argv or receipts.
All Hub files are content-verified before the atomic completion receipt.
"""
import argparse
import fcntl
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse


def resume_file(entry, output, *, url, headers=None):
    """Use one stable partial per content hash, including older Hub partials.

    The enclosing download holds a local exclusive lock. New Hub releases
    intentionally use process-unique partials and no longer resume them.
    Keep authentication out of URLs and let the Hub HTTP helper handle Range
    retries and servers that ignore Range. Final content hashes remain required.
    """
    from huggingface_hub.file_download import http_get

    target = output / entry["path"]
    if target.is_file() and target.stat().st_size == entry["size"]:
        return
    digest = entry["sha256"] or entry["git_blob"]
    cache = output / ".cache/mlx2-ripple-download"
    cache.mkdir(parents=True, exist_ok=True)
    partial = cache / (digest + ".incomplete")
    old = list((output / ".cache/huggingface/download").glob(f"*.{digest}.*.incomplete"))
    candidates = [p for p in [partial, *old] if p.is_file()]
    if any(p.stat().st_size > entry["size"] for p in candidates):
        raise ValueError(f"oversized download partial: {entry['path']}")
    if candidates:
        largest = max(candidates, key=lambda p: p.stat().st_size)
        if largest != partial:
            os.replace(largest, partial)
    target.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("ab") as stream:
        http_get(url, stream, resume_size=stream.tell(), headers=headers,
                 expected_size=entry["size"], displayed_filename=entry["path"])
    os.replace(partial, target)
    # These are duplicate prefixes of this exact content hash, created by the
    # preceding Hub transfer. The completed target is still hash-checked below.
    for path in old:
        path.unlink(missing_ok=True)


def download(output, *, token_file=None, workers=5, variant="nvfp4"):
    if variant not in {"nvfp4", "bf16"}:
        raise ValueError("unknown Ripple variant")
    if workers != 5:
        raise ValueError("Ripple acquisition uses exactly five workers")
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    token_file = Path.home()/".hf_pr" if token_file is None else Path(token_file)
    manifest_name = "ripple-65b-bf16-snapshot.json" if variant == "bf16" else "ripple-65b-snapshot.json"
    manifest = json.loads((Path(__file__).resolve().parents[1]/"provenance"/manifest_name).read_text())
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True,exist_ok=True)
    with (output/".mlx2-download.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        token = Path(token_file).expanduser().read_text().strip()

        def transfer(entry):
            target = output / entry["path"]
            if target.is_file() and target.stat().st_size == entry["size"]:
                return
            url = hf_hub_url(manifest["repo"], entry["path"], revision=manifest["revision"])
            metadata = get_hf_file_metadata(url, token=token)
            expected_hash = entry["sha256"] or entry["git_blob"]
            if (metadata.commit_hash != manifest["revision"] or metadata.size != entry["size"]
                    or metadata.etag != expected_hash):
                raise ValueError(f"Hub metadata differs from pinned manifest: {entry['path']}")
            # Signed CDN URLs carry their own authorization; never send the PAT
            # to a different origin.
            headers = {"authorization": "Bearer " + token} if (
                urlparse(metadata.location).hostname == "huggingface.co") else None
            resume_file(entry, output, url=metadata.location, headers=headers)

        with ThreadPoolExecutor(max_workers=5) as pool:
            list(pool.map(transfer, manifest["files"]))

        def verify(entry):
            path = output/entry["path"]
            if not path.is_file() or path.stat().st_size != entry["size"]:
                raise ValueError(f"missing/incomplete snapshot file: {entry['path']}")
            digest = hashlib.sha256() if entry["sha256"] else hashlib.sha1()
            if not entry["sha256"]:
                digest.update(f"blob {entry['size']}\0".encode())
            with path.open("rb") as stream:
                for block in iter(lambda:stream.read(8<<20),b""):
                    digest.update(block)
            if digest.hexdigest() != (entry["sha256"] or entry["git_blob"]):
                raise ValueError(f"snapshot content hash differs: {entry['path']}")
            return {"path":entry["path"],"bytes":entry["size"],"content_hash":digest.hexdigest()}

        with ThreadPoolExecutor(max_workers=5) as pool:
            verified = list(pool.map(verify,manifest["files"]))
        receipt = {"repo":manifest["repo"],"revision":manifest["revision"],"workers":5,
                   "bytes":sum(e["size"] for e in manifest["files"]),"verified":verified}
        target = output/".hf-download-complete.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt,indent=2)+"\n")
        os.replace(temporary,target)
        return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--token-file",type=Path,default=Path.home()/".hf_pr")
    parser.add_argument("--workers",type=int,default=5)
    parser.add_argument("--variant", choices=("nvfp4", "bf16"), default="nvfp4")
    args = parser.parse_args()
    receipt = download(args.output,token_file=args.token_file,workers=args.workers,variant=args.variant)
    print(json.dumps({k:v for k,v in receipt.items() if k != "verified"},indent=2))
