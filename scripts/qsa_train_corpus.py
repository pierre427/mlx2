#!/usr/bin/env python3
"""Build the token stream for QSA indexer distillation.

Two steps, because the serving venv has no parquet reader:

* ``extract`` (any interpreter with pyarrow, e.g. ``uv run --no-project
  --with pyarrow``): the first ``--docs`` documents of a FineWeb-Edu parquet
  shard, in shard order, to JSONL;
* ``tokenize`` (the mlx2 venv): each document encoded with the target's
  tokenizer, EOS-separated and concatenated (the reference ``iter_token_blocks``
  packing), split into ``--seq-len`` training sequences plus a held-out tail
  of ``--holdout`` sequences, saved as one uint32 ``.npy`` [n, seq_len] with a
  JSON receipt (source hashes, tokenizer, counts).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(args) -> int:
    import pyarrow.parquet as pq

    table = pq.ParquetFile(args.parquet)
    written = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for batch in table.iter_batches(batch_size=1024, columns=["text", "id"]):
            for (text, doc_id) in zip(batch.column("text").to_pylist(),
                                      batch.column("id").to_pylist()):
                out.write(json.dumps({"id": doc_id, "text": text}) + "\n")
                written += 1
                if written >= args.docs:
                    print(json.dumps({"docs": written}))
                    return 0
    print(json.dumps({"docs": written}))
    return 0


def tokenize(args) -> int:
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from transformers import PreTrainedTokenizerFast

    from mlx2.runtime.tokenizer_integrity import repair_loaded_tokenizer

    # The serving adapter's tokenizer law (fast tokenizer, local files, the
    # pretokenizer repair), not AutoTokenizer's remote-code path.
    tok = PreTrainedTokenizerFast.from_pretrained(args.model, local_files_only=True)
    repair_loaded_tokenizer(tok, Path(args.model))
    eos = tok.eos_token_id
    need = (args.sequences + args.holdout) * args.seq_len
    stream: list[int] = []
    docs = 0
    with open(args.jsonl, encoding="utf-8") as handle:
        for line in handle:
            stream.extend(tok.encode(json.loads(line)["text"], add_special_tokens=False))
            stream.append(eos)
            docs += 1
            if len(stream) >= need:
                break
    if len(stream) < need:
        raise SystemExit(f"{len(stream)} tokens from {docs} docs, need {need}")
    arr = np.asarray(stream[:need], dtype=np.uint32).reshape(-1, args.seq_len)
    np.save(args.out, arr)
    receipt = {
        "jsonl": str(args.jsonl), "jsonl_sha256": sha256(Path(args.jsonl)),
        "parquet": args.parquet, "parquet_sha256": args.parquet_sha256,
        "tokenizer": args.model, "eos": eos, "docs_used": docs,
        "seq_len": args.seq_len, "train_sequences": args.sequences,
        "holdout_sequences": args.holdout, "holdout_rows": "last",
        "npy_sha256": sha256(Path(args.out)),
    }
    Path(str(args.out) + ".json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("parquet")
    ex.add_argument("out")
    ex.add_argument("--docs", type=int, default=8000)
    tk = sub.add_parser("tokenize")
    tk.add_argument("jsonl")
    tk.add_argument("out")
    tk.add_argument("--model", required=True)
    tk.add_argument("--parquet", default=None)
    tk.add_argument("--parquet-sha256", default=None)
    tk.add_argument("--seq-len", type=int, default=16384)
    tk.add_argument("--sequences", type=int, default=150)
    tk.add_argument("--holdout", type=int, default=2)
    args = parser.parse_args()
    return extract(args) if args.cmd == "extract" else tokenize(args)


if __name__ == "__main__":
    raise SystemExit(main())
