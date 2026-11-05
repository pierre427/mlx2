# SPDX-License-Identifier: Apache-2.0
"""Immutable arithmetic identities, independent of device libraries.

A fingerprint identifies a declared graph and its architecture/runtime scope.
It does not prove the graph's lowering, native equality, or qualification.
Batch size, padding and scheduling are excluded from per-output dimensions;
changing a primitive, rounding boundary or chunk split creates another class.
See provenance/flashinfer-arithmetic-class-20261104.json.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

SCHEMA = "mlx2.arithmetic-class.v1"
_FINGERPRINT = re.compile(r"[0-9a-f]{64}")
_DTYPES = {"float16", "bfloat16", "float32", "mixed-affine", "mixed-supported"}
_PLACEMENT_DIMENSIONS = {"m", "batch", "batch_size", "padding", "sm_count", "physical_row", "stream"}


def _text(value, name):
    if type(value) is not str or not value or len(value) > 2048:
        raise ValueError(f"{name} must be bounded nonempty text")


def _integer(value, name, *, zero=False):
    if type(value) is not int or not (0 if zero else 1) <= value <= (1 << 63) - 1:
        raise ValueError(f"{name} is outside signed-64-bit nonnegative bounds")


def _pairs(value, name, kind):
    if type(value) is not tuple or len(value) > 32:
        raise ValueError(f"{name} must be a bounded immutable tuple")
    seen = set()
    for pair in value:
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError(f"{name} entries must be immutable pairs")
        key, item = pair
        _text(key, name)
        if key in seen:
            raise ValueError(f"duplicate {name} key")
        seen.add(key)
        if kind is int:
            _integer(item, key)
            if key.lower() in _PLACEMENT_DIMENSIONS:
                raise ValueError("batch/placement is not a per-output arithmetic dimension")
        else:
            _text(item, key)


@dataclass(frozen=True)
class ArithmeticClass:
    operation: str
    algorithm_version: int
    architecture: str
    runtime_revision: str
    scope: str
    input_dtype: str
    output_dtype: str
    accumulator_dtype: str
    dimensions: tuple[tuple[str, int], ...]
    dataflow: tuple[tuple[str, str], ...]

    def __post_init__(self):
        for name in ("operation", "architecture", "runtime_revision", "scope"):
            _text(getattr(self, name), name)
        _integer(self.algorithm_version, "algorithm_version")
        for name in ("input_dtype", "output_dtype", "accumulator_dtype"):
            if getattr(self, name) not in _DTYPES:
                raise ValueError(f"unsupported {name}")
        _pairs(self.dimensions, "dimensions", int)
        _pairs(self.dataflow, "dataflow", str)
        if not self.dataflow:
            raise ValueError("dataflow must describe numerical operations")
        object.__setattr__(self, "dimensions", tuple(sorted(self.dimensions)))

    @property
    def fingerprint(self) -> str:
        document = {"schema": SCHEMA, **asdict(self)}
        encoded = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(encoded.encode("ascii")).hexdigest()

    def metadata(self) -> dict:
        return {"schema": SCHEMA, **asdict(self), "fingerprint": self.fingerprint}

    @classmethod
    def from_metadata(cls, metadata: dict) -> ArithmeticClass:
        if type(metadata) is not dict:
            raise ValueError("arithmetic metadata must be an object")
        expected = {"schema", "fingerprint", *cls.__dataclass_fields__}
        if set(metadata) != expected or metadata.get("schema") != SCHEMA:
            raise ValueError("unknown or incomplete arithmetic metadata schema")
        fingerprint = metadata["fingerprint"]
        if type(fingerprint) is not str or not _FINGERPRINT.fullmatch(fingerprint):
            raise ValueError("invalid arithmetic fingerprint")
        values = {name: metadata[name] for name in cls.__dataclass_fields__}
        for name in ("dimensions", "dataflow"):
            try:
                values[name] = tuple(tuple(pair) for pair in values[name])
            except TypeError as error:
                raise ValueError("invalid arithmetic metadata pairs") from error
        result = cls(**values)
        if result.fingerprint != fingerprint:
            raise ValueError("arithmetic metadata does not match its fingerprint")
        return result

    def require_compatible(self, metadata: dict) -> None:
        if self.fingerprint != self.from_metadata(metadata).fingerprint:
            raise ValueError("arithmetic class/architecture/runtime mismatch")


def attention_class(head_dim: int, *, architecture: str, runtime_revision: str,
                    tile_keys: int = 64, chunk_keys: int = 512) -> ArithmeticClass:
    """The existing dense lane attention graph, not NVIDIA's attention class."""
    for value, name in ((head_dim, "head_dim"), (tile_keys, "tile_keys"), (chunk_keys, "chunk_keys")):
        _integer(value, name)
    if head_dim not in (128, 256) or tile_keys != 64 or chunk_keys != 512:
        raise ValueError("existing lane attention takes D128/256, tile64, chunk512")
    return ArithmeticClass(
        "tensorfold-lane-attention", 1, architecture, runtime_revision,
        "same-architecture-and-installed-runtime; native validation separate",
        "bfloat16", "bfloat16", "float32",
        (("head_dim", head_dim), ("tile_keys", tile_keys), ("chunk_keys", chunk_keys)),
        (("qk", "apple-mpp-matmul2d-bf16-score-fixed-D"),
         ("tiles", "absolute-logical-key-origin-zero; masked-key-zero-contribution"),
         ("statistics", "fixed-simd-shuffle-max-and-pairwise-tile-sum-fp32"),
         ("probability", "metal-fast-exp; store-float16-before-PV"),
         ("pv", "apple-mpp-half-times-bf16; fp32-accumulator"),
         ("merge", "ascending-absolute-chunk-index; fp32-rescale-and-add"),
         ("output", "fp32-division-then-bfloat16-cast")),
    )


def gemm_class(n: int, k: int, *, bits: int, group_size: int, split_k: int,
               architecture: str, runtime_revision: str) -> ArithmeticClass:
    """Explicit instance of the existing affine TensorFold MPP row law.

    The caller supplies the installed, weight-shape-only split. This function
    cannot infer or authorize that split from a batch/dispatcher configuration.
    """
    for value, name in ((n, "n"), (k, "k"), (group_size, "group_size"), (split_k, "split_k")):
        _integer(value, name)
    if (type(bits) is not int or bits not in (2, 3, 4, 5, 6, 8)
            or group_size not in ((32, 64) if bits == 4 else (64,))
            or k % group_size or n % 4 or split_k not in (1, 2, 4, 8)
            or k // group_size < split_k):
        raise ValueError("unsupported existing affine lane geometry")
    return ArithmeticClass(
        "tensorfold-lane-affine-gemm", 1, architecture, runtime_revision,
        "same-architecture-and-installed-runtime; native validation separate",
        "bfloat16", "bfloat16", "float32",
        (("n", n), ("k", k), ("bits", bits), ("group_size", group_size), ("split_k", split_k)),
        (("products", "apple-mpp-bf16-times-exact-integer-codes-per-group"),
         ("activation_sum", "ascending-column-fp32-fold-per-group"),
         ("affine", "fma(scale,integer-product,fma(bias,activation-sum,acc))"),
         ("partitions", "weight-shape-selected-fixed-group-boundaries"),
         ("merge", "ascending-split-index-fp32-add"),
         ("output", "bfloat16-cast-after-complete-row-fold")),
    )


def lane_family_identity(backend: str, *, architecture: str, runtime_revision: str,
                         implementation_sha256: str | None = None) -> dict:
    """Recipe family supplements the existing artifact/shape/policy identity.

    It covers the installed mixed-format projection family, not a claim that
    affine, unquantized, MX and SIMD primitives agree. Those remain distinct
    formats/backends and the pre-existing source/shape gates still apply.
    """
    if backend not in {"mpp", "simd"}:
        raise ValueError("unknown lane arithmetic backend")
    implementation = ()
    if implementation_sha256 is not None:
        if type(implementation_sha256) is not str or not _FINGERPRINT.fullmatch(implementation_sha256):
            raise ValueError("invalid lane implementation digest")
        implementation = (("implementation_sha256", implementation_sha256),)
    return ArithmeticClass(
        "installed-lane-projection-family", 1, architecture, runtime_revision,
        "family-recipe; loaded-artifact-and-installed-format-bindings-required",
        "mixed-supported", "mixed-supported", "float32", (),
        (("backend", "apple-mpp-tensor-ops" if backend == "mpp" else "apple-simdgroup-matrix"),
         ("formats", "existing-per-format-primitive-and-rounding; no-cross-format-equality"),
         ("split", "fixed-by-stacked-weight-shape-and-format; independent-of-M"),
         ("merge", "ascending-split-order-fp32-add"),
         ("epilogue", "existing-per-format-output-cast-after-complete-fold")) + implementation,
    ).metadata()


@lru_cache(maxsize=4)
def lane_implementation_digest(runtime_root: str | Path | None = None) -> str:
    """Cold-process source snapshot; never perform I/O on a request's hot path.

    A process retains its imported numerical code. A new process or explicit
    reviewed reload must refresh this snapshot together with its loaded code.
    Names, lengths and contents are bound independently of checkout location.
    """
    root = Path(__file__).resolve().parent if runtime_root is None else Path(runtime_root)
    digest = hashlib.sha256()
    for name in ("arithmetic_class.py", "lane/matmul.py", "lane/simd.py",
                 "lane/installer.py", "lane/tiled_prefill.py"):
        path = root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 1 << 20:
            raise RuntimeError(f"lane implementation source is missing, indirect or unbounded: {name}")
        content = path.read_bytes()
        if len(content) > 1 << 20:
            raise RuntimeError("lane implementation source changed beyond the byte bound")
        digest.update(name.encode("ascii"))
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def tensorfold_family_identity(*, runtime_revision: str) -> dict:
    """Source-family token; exact native architecture remains adapter-bound."""
    return ArithmeticClass(
        "tensorfold-source-family", 1, "apple-metal-mpp-family", runtime_revision,
        "source-family-only; not-a-cross-device-equality-or-qualification-claim",
        "mixed-supported", "bfloat16", "float32",
        (("attention_tile_keys", 64), ("attention_chunk_keys", 512)),
        (("attention", "absolute-key-tiles; half-P; fast-exp; ascending-chunk-merge"),
         ("projection", "weight-shape-only-split; per-group-affine-FMA; ordered-split-merge"),
         ("tree", "selected-tree-law-and-owned-GDN-remain-separately-bound")),
    ).metadata()


def lane_cache_fingerprint(base, receipt):
    if not receipt or not receipt.get("covered"):
        return base
    law = receipt.get("law_id")
    _text(law, "lane law")
    metadata = receipt.get("arithmetic_class")
    # Readable legacy receipts cannot alias either old three-field namespaces
    # or newly installed arithmetic-tagged state.
    token = "legacy-unversioned" if metadata is None else ArithmeticClass.from_metadata(metadata).fingerprint
    return base, "lane-matmul", law, token


def flattened_offset(row: int, stride: int, column: int, *, total_elements: int) -> int:
    """Validate in Python integers before native lowering to signed int64."""
    for value, name in ((row, "row"), (column, "column")):
        _integer(value, name, zero=True)
    _integer(stride, "stride")
    _integer(total_elements, "total_elements")
    offset = row * stride + column
    if offset >= total_elements or offset > (1 << 63) - 1:
        raise ValueError("flattened address exceeds buffer or signed int64")
    return offset


def vector_address(base_address: int, element_offset: int, *, element_bytes: int,
                   vector_elements: int, total_elements: int) -> int:
    """Check an aligned power-of-two packet of at most 16 bytes plus its tail.

    This does not infer arbitrary native vector layouts (e.g. Metal float3).
    Only 1/2/4/8-byte elements and power-of-two packet lengths are admitted.
    """
    for value, name in ((base_address, "base_address"), (element_offset, "element_offset")):
        _integer(value, name, zero=True)
    for value, name in ((element_bytes, "element_bytes"), (vector_elements, "vector_elements"),
                        (total_elements, "total_elements")):
        _integer(value, name)
    address = base_address + element_offset * element_bytes
    width = element_bytes * vector_elements
    if (element_bytes not in (1, 2, 4, 8) or vector_elements & (vector_elements - 1)
            or width > 16):
        raise ValueError("only power-of-two scalar/vector packets through 16 bytes are supported")
    if (element_offset + vector_elements > total_elements or address > (1 << 63) - 1
            or width > (1 << 63) - 1 or address + width - 1 > (1 << 63) - 1
            or address % width):
        raise ValueError("vector access has invalid alignment, extent, or int64 address")
    return address
