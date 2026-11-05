#!/usr/bin/env python3
"""Acquire the pinned Inkling mixed-2bit artifact with exactly five download workers.

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


def download(output, *, token_file=None, workers=5):
    if workers != 5:
        raise ValueError("Inkling acquisition uses exactly five workers")
    from huggingface_hub import snapshot_download
    token_file = Path.home()/".hf_pr" if token_file is None else Path(token_file)
    manifest = json.loads((Path(__file__).resolve().parents[1]/"provenance/inkling-small-snapshot.json").read_text())
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True,exist_ok=True)
    with (output/".mlx2-download.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        snapshot_download(manifest["repo"],revision=manifest["revision"],local_dir=output,
                          token=Path(token_file).expanduser().read_text().strip(),max_workers=5)

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
    args = parser.parse_args()
    receipt = download(args.output,token_file=args.token_file,workers=args.workers)
    print(json.dumps({k:v for k,v in receipt.items() if k != "verified"},indent=2))
