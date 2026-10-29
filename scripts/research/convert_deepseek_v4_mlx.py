#!/usr/bin/env python3
"""Losslessly convert the sealed Vision-Exp target and optional DSpark to MLX."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("destination",type=Path)
    parser.add_argument("--sidecar",type=Path)
    parser.add_argument("--receipt",type=Path,required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        parser.error("refusing existing conversion receipt")
    # Configuration/schema helpers import model classes, but conversion itself
    # is CPU file work and must never schedule a Metal allocation.
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from mlx2.adapters.deepseek_v4_mlx_artifact import convert_artifact
    from mlx2.runtime.gguf.mlx_storage import atomic_json
    start=time.time()
    record=dict(status="running",source=str(args.source),destination=str(args.destination),started_unix=start)
    atomic_json(args.receipt,record)
    def progress(value):
        record.update(progress=value,updated_unix=time.time())
        atomic_json(args.receipt,record)
        print(json.dumps(value),flush=True)
    try:
        artifact=convert_artifact(args.source,args.destination,sidecar=args.sidecar,progress=progress)
        record.update(status="verified",artifact=artifact,elapsed_seconds=time.time()-start)
    except BaseException as error:
        record.update(status="failed",error=repr(error),elapsed_seconds=time.time()-start)
        raise
    finally:
        atomic_json(args.receipt,record)
    print(json.dumps(dict(status=record['status'],identity=artifact['identity'],elapsed_seconds=record['elapsed_seconds'])),flush=True)


if __name__ == "__main__":
    main()
