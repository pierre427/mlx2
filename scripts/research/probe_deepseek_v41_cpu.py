#!/usr/bin/env python3
"""CPU-only V4.1 admission, hash and selected-row oracle against pinned ds4 C.

Reads only the GGUF header and a small set of actual Engram rows. Never imports
MLX, loads target weights, starts a server, or requests GPU ownership.
"""

import argparse
import ctypes as ct
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mlx2.adapters.deepseek_v41_artifact import FILENAME, inspect_artifact
from mlx2.runtime.gguf.reader import GGUFFile
from mlx2.runtime.models.deepseek_v41_engram import EngramHasher, EngramRows

PIN = "0dff39f91721bcf3472813c6d2bebf9353df2a1a"


class Layout(ct.Structure):
    _fields_ = [
        ("token_map", ct.POINTER(ct.c_uint32)),
        ("vocab_size", ct.c_uint32),
        ("compressed_vocab_size", ct.c_uint32),
        ("pad_id", ct.c_uint32),
        ("rows", ct.c_uint32 * 2),
        ("multipliers", (ct.c_uint64 * 4) * 2),
        ("primes", (ct.c_uint32 * 24) * 2),
    ]


class History(ct.Structure):
    _fields_ = [("tail", ct.c_int32 * 3)]


class Table(ct.Structure):
    _fields_ = [("fd", ct.c_int), ("offset", ct.c_uint64), ("rows", ct.c_uint32)]


def run(path, reference):
    revision = subprocess.check_output(
        ["git", "-C", str(reference), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != PIN:
        raise ValueError("ds4 oracle source differs from pinned revision")
    if subprocess.check_output(
        [
            "git",
            "-C",
            str(reference),
            "status",
            "--porcelain",
            "--",
            "ds4_engram.c",
            "ds4_engram.h",
        ],
        text=True,
    ).strip():
        raise ValueError("ds4 oracle source is dirty")
    artifact = inspect_artifact(path)
    metadata = artifact["metadata"]
    hasher = EngramHasher(metadata)
    ids = np.array([0, 120, 211, 421, 1139, 352, 120, 211, 421, 1], np.int32)
    with tempfile.TemporaryDirectory(prefix="mlx2-v41-cpu-") as tmp:
        library = Path(tmp) / "engram.dylib"
        subprocess.run(
            [
                "clang",
                "-dynamiclib",
                "-O2",
                str(reference / "ds4_engram.c"),
                "-o",
                str(library),
            ],
            check=True,
            capture_output=True,
        )
        lib = ct.CDLL(str(library))
        token_map = np.asarray(hasher.token_map, np.uint32)
        layout = Layout(
            token_map.ctypes.data_as(ct.POINTER(ct.c_uint32)),
            len(token_map),
            hasher.vocab,
            hasher.pad_id,
            (ct.c_uint32 * 2)(*hasher.rows),
            ((ct.c_uint64 * 4) * 2)(
                *[(ct.c_uint64 * 4)(*map(int, r)) for r in hasher.multipliers]
            ),
            ((ct.c_uint32 * 24) * 2)(
                *[(ct.c_uint32 * 24)(*map(int, r)) for r in hasher.primes]
            ),
        )
        lib.ds4_engram_layout_valid.argtypes = [ct.POINTER(Layout)]
        lib.ds4_engram_layout_valid.restype = ct.c_bool
        assert lib.ds4_engram_layout_valid(ct.byref(layout))
        lib.ds4_engram_hash.argtypes = [
            ct.POINTER(Layout),
            ct.POINTER(History),
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_uint8),
            ct.c_size_t,
            ct.POINTER(ct.c_uint32),
        ]
        lib.ds4_engram_hash.restype = ct.c_bool
        checks = {}
        for masked in (False, True):
            mask = np.ones(len(ids), np.uint8)
            if masked:
                mask[3] = 0
            history = History((ct.c_int32 * 3)(-1, -1, -1))
            expected = np.zeros((len(ids), 2, 24), np.uint32)
            assert lib.ds4_engram_hash(
                ct.byref(layout),
                ct.byref(history),
                ids.ctypes.data_as(ct.POINTER(ct.c_int)),
                mask.ctypes.data_as(ct.POINTER(ct.c_uint8)),
                len(ids),
                expected.ctypes.data_as(ct.POINTER(ct.c_uint32)),
            )
            actual, tail = hasher(ids, np.full(3, -1, np.int32), mask.astype(bool))
            checks["hash_masked" if masked else "hash_text"] = bool(
                np.array_equal(actual, expected)
                and np.array_equal(tail, np.asarray(history.tail))
            )
        hashes, _ = hasher(ids, np.full(3, -1, np.int32))
        lib.ds4_engram_table_open.argtypes = [
            ct.POINTER(Table),
            ct.c_char_p,
            ct.c_uint64,
            ct.c_uint32,
        ]
        lib.ds4_engram_table_open.restype = ct.c_bool
        lib.ds4_engram_read.argtypes = [
            ct.POINTER(Table),
            ct.POINTER(ct.c_uint32),
            ct.c_size_t,
            ct.POINTER(ct.c_float),
        ]
        lib.ds4_engram_read.restype = ct.c_bool
        lib.ds4_engram_table_close.argtypes = [ct.POINTER(Table)]
        row_stats = {}
        with GGUFFile(path / FILENAME) as file:
            for index, layer in enumerate(metadata["deepseek41.engram.layer_ids"]):
                tensor = file.tensors[f"blk.{layer}.engram_embd.weight"]
                offset = file.data_offset + tensor.offset
                rows = np.ascontiguousarray(hashes[:, index].reshape(-1), np.uint32)
                expected = np.empty((len(rows), 256), np.float32)
                table = Table()
                assert lib.ds4_engram_table_open(
                    ct.byref(table),
                    os.fsencode(path / FILENAME),
                    offset,
                    hasher.rows[index],
                )
                reader = EngramRows(
                    path / FILENAME,
                    offset=offset,
                    rows=hasher.rows[index],
                    expected_stat=artifact["weight_stat"],
                )
                try:
                    assert lib.ds4_engram_read(
                        ct.byref(table),
                        rows.ctypes.data_as(ct.POINTER(ct.c_uint32)),
                        len(rows),
                        expected.ctypes.data_as(ct.POINTER(ct.c_float)),
                    )
                    actual = reader.read(rows)
                    checks[f"layer{layer}_row_bits"] = bool(
                        np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
                    )
                    reader.read(rows)
                    row_stats[str(layer)] = reader.diagnostics()
                finally:
                    reader.close()
                    lib.ds4_engram_table_close(ct.byref(table))
    if "mlx.core" in sys.modules:
        raise RuntimeError("CPU admission oracle unexpectedly imported MLX")
    files = [p for p in ROOT.glob("src/mlx2/**/*v41*.py")] + [Path(__file__)]
    source = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(files)
    }
    return {
        "schema": "mlx2.deepseek-v41-cpu-admission-oracle.v1",
        "checkpoint": FILENAME + "@sha256:" + artifact["identity"]["weight_sha256"],
        "artifact_fingerprint": artifact["identity"]["fingerprint"],
        "source_revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_files": source,
        "source_snapshot_sha256": hashlib.sha256(
            json.dumps(source, sort_keys=True).encode()
        ).hexdigest(),
        "oracle_source_revision": revision,
        "checks": checks,
        "all_passed": all(checks.values()),
        "row_io": row_stats,
        "artifact_loaded": False,
        "GPU_used": False,
        "qualification": False,
        "scope": "header, tokenizer binding, compressed IDs, selected Engram row bits; no decode or performance evidence",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ds4-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing existing evidence file")
    result = run(args.model, args.ds4_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(
        json.dumps(
            {
                "all_passed": result["all_passed"],
                "checks": result["checks"],
                "GPU_used": False,
            }
        )
    )
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
