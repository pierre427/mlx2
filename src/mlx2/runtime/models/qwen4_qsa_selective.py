# SPDX-License-Identifier: Apache-2.0
"""Producer-side QSA candidates; explicit unqualified scalar-F32 score law.

This module imports no MLX until ``select_metal`` is explicitly invoked.
CPU execution owns immutable byte snapshots. Native stages run inside one
row-owned kernel invocation over the same retained input graph roots. Neither
backend materializes a dense score row. This first native candidate is serial
within each query row; it establishes an executable slice, not a speed claim.
See provenance/flashinfer-selective-qsa-20261104.json.
"""
from __future__ import annotations

import hashlib
import heapq
import math
import threading
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

PROFILE = "qsa-selective-scalar-f32-v1"
_SOURCE_BYTES = Path(__file__).read_bytes()
_METAL_TEXT = Path(__file__).with_suffix(".metal").read_text()
_LOCK = threading.Lock()
_COUNTS = Counter()


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be integer in [{low}, {high}]")


@dataclass(frozen=True)
class SelectiveConfig:
    topk: int
    ratio: int
    capacity: int = 2048
    samples: int = 64
    banks: int = 4
    max_input_bytes: int = 256 << 20
    native_window_rows: int = 32
    max_workspace_bytes: int = 64 << 20

    def __post_init__(self):
        _integer(self.topk, "topk", 1, 512)
        _integer(self.ratio, "ratio", 1, 2**31 - 1)
        _integer(self.capacity, "capacity", self.topk, 4096)
        _integer(self.samples, "samples", 1, 256)
        if self.banks != 4 or type(self.banks) is not int or self.capacity % self.banks:
            raise ValueError("four equal candidate banks are required")
        _integer(self.max_input_bytes, "max_input_bytes", 1, 2**40)
        _integer(self.native_window_rows, "native_window_rows", 1, 32)
        _integer(self.max_workspace_bytes, "max_workspace_bytes", 1, 2**40)


def arithmetic_contract():
    """Content-bound law for execution receipts and APCv2 arithmetic identity."""
    digest = hashlib.sha256()
    for name, data in (("qwen4_qsa_selective.py", _SOURCE_BYTES),
                       ("qwen4_qsa_selective.metal", _METAL_TEXT.encode())):
        digest.update(name.encode())
        digest.update(hashlib.sha256(data).digest())
    return {"profile": PROFILE, "source_sha256": digest.hexdigest(),
            "selection": "raw-f32-composite-larger-id-ascending-tail-fill-v1",
            "qualified": False, "stock_gemm_parity": "unvalidated"}


def _metadata(qshape, pshape, positions_shape, config):
    if (len(qshape) != 4 or len(pshape) != 3 or len(positions_shape) != 2
            or qshape[:2] != positions_shape or qshape[0] != pshape[0]
            or qshape[2:] != (4, 128) or pshape[2] != 128
            or not 1 <= qshape[0] <= 64 or not 1 <= qshape[1] <= 4096
            or qshape[0] * qshape[1] > 4096
            or not config.topk <= pshape[1] < 2**31):
        raise ValueError("selective QSA requires B<=64,L<=4096,H4,D128,N>=K and <=4096 rows")
    # Conservative FP32 input/snapshot ceiling, even for lower-precision
    # native inputs. Refuse before contiguous copies or output allocation.
    if (math.prod(qshape) * 4 + math.prod(pshape) * 4
            + math.prod(positions_shape) * 8 > config.max_input_bytes):
        raise ValueError("immutable score input budget exceeded")
    if declared_workspace_bytes(qshape, config) > config.max_workspace_bytes:
        raise ValueError("selective QSA declared workspace budget exceeded")


def declared_workspace_bytes(qshape, config):
    """Conservative named-buffer allowance, not a measured allocator peak.

    Two copies of the per-row private candidate/heap/sort/sample arrays allow
    room for compiler temporaries. At most native_window_rows are dispatched
    and synchronously retired before the next window. Concatenated IDs and
    retained window IDs are charged twice; only one diagnostic window lives.
    Compiler/GPU allocator overhead still needs hardware measurements.
    """
    rows = qshape[0] * qshape[1]
    window = min(rows, config.native_window_rows)
    private = 8 * config.capacity + 12 * config.topk + 4 * config.samples + 128
    return 2 * private * window + 8 * rows * config.topk + 16 * window


@dataclass(frozen=True)
class ScoreSnapshot:
    query: np.ndarray
    pooled: np.ndarray
    positions: np.ndarray
    artifact_revision: str
    source_revision: str
    binding: str

    @classmethod
    def capture(cls, query, pooled, positions, *, artifact_revision, source_revision, config):
        if not isinstance(config, SelectiveConfig):
            raise TypeError("validated selective config required")
        # Device __array__ methods must never execute on the CPU path.
        if any(type(v) is not np.ndarray for v in (query, pooled, positions)):
            raise TypeError("CPU score inputs must be NumPy ndarrays")
        if query.dtype != np.float32 or pooled.dtype != np.float32:
            raise ValueError("CPU score operands must be float32")
        if positions.dtype not in (np.dtype("int32"), np.dtype("int64")):
            raise ValueError("query positions must be signed integer")
        if not all(isinstance(v, str) and v.strip() for v in (artifact_revision, source_revision)):
            raise ValueError("artifact and source revisions are required")
        _metadata(query.shape, pooled.shape, positions.shape, config)
        if sum(v.nbytes for v in (query, pooled, positions)) > config.max_input_bytes:
            raise ValueError("immutable score input budget exceeded")
        if np.any(positions < -(2**31)) or np.any(positions >= 2**31):
            raise ValueError("query positions exceed native int32 contract")
        # frombuffer(bytes) cannot be made writable by setflags(write=True).
        frozen = tuple(np.frombuffer(v.tobytes(), dtype=v.dtype).reshape(v.shape)
                       for v in (query, pooled, positions))
        digest = hashlib.sha256()
        digest.update(repr((query.shape, pooled.shape, positions.shape,
                            artifact_revision, source_revision, PROFILE)).encode())
        for v in frozen:
            digest.update(v.dtype.str.encode())
            digest.update(memoryview(v).cast("B"))
        return cls(*frozen, artifact_revision, source_revision, digest.hexdigest())

    def verify(self):
        # Reject forged or mutable snapshots rather than accepting a frozen
        # dataclass whose ndarray contents can still change between passes.
        for value in (self.query, self.pooled, self.positions):
            if (type(value) is not np.ndarray or value.flags.writeable
                    or not value.flags.c_contiguous):
                raise ValueError("score snapshot is not immutable")
            root = value
            while type(root) is np.ndarray:
                root = root.base
            if type(root) is not bytes:
                raise ValueError("score snapshot must own immutable byte storage")
        if (self.query.dtype != np.float32 or self.pooled.dtype != np.float32
                or self.positions.dtype not in (np.dtype("int32"), np.dtype("int64"))):
            raise ValueError("score snapshot dtype contract changed")
        if (not all(isinstance(v, str) and v.strip()
                    for v in (self.artifact_revision, self.source_revision))
                or np.any(self.positions < -(2**31)) or np.any(self.positions >= 2**31)):
            raise ValueError("score snapshot revision or position contract changed")
        digest = hashlib.sha256()
        digest.update(repr((self.query.shape, self.pooled.shape, self.positions.shape,
                            self.artifact_revision, self.source_revision, PROFILE)).encode())
        for value in (self.query, self.pooled, self.positions):
            digest.update(value.dtype.str.encode())
            digest.update(memoryview(value).cast("B"))
        if digest.hexdigest() != self.binding:
            raise ValueError("score snapshot source identity changed")


def _score(snapshot, batch, row, block):
    q, p = snapshot.query, snapshot.pooled
    total = np.float32(0)
    # Explicit separate FP32 multiply and add, then head-order ReLU sum.
    with np.errstate(invalid="ignore", over="ignore"):
        for head in range(4):
            dot = np.float32(0)
            for dim in range(128):
                dot = np.float32(dot + np.float32(q[batch, row, head, dim] * p[batch, block, dim]))
            total = np.float32(total + np.maximum(dot, np.float32(0)))
        return np.float32(total / np.float32(11.313708498984761))


def _key(value, index):
    bits = int(value.view(np.uint32))
    ordered = (~bits & 0xFFFFFFFF) if bits & 0x80000000 else bits ^ 0x80000000
    return (ordered << 32) | index


def _row_select(score, blocks, position, config):
    valid = min(blocks, max(0, (int(position) + 1) // config.ratio))
    take = min(config.topk, valid)
    if not take:
        return list(range(blocks - config.topk, blocks)), (0, 0, 0, valid)
    sample_count = min(valid, config.samples)
    sampled = sorted((_key(score(i * valid // sample_count), 0) >> 32
                      for i in range(sample_count)), reverse=True)
    rank = min(sample_count, max(1, math.ceil(1.5 * take * sample_count / valid)))
    threshold = sampled[rank - 1]
    banks = [[] for _ in range(4)]
    counts = [0] * 4
    span = config.capacity // 4
    for index in range(valid):
        key = _key(score(index), index)
        if key >> 32 >= threshold:
            bank = index % 4
            counts[bank] += 1
            if len(banks[bank]) < span:
                banks[bank].append(key)
    n = sum(counts)
    stored = sum(map(len, banks))
    if n >= take and max(counts) <= span:
        chosen = heapq.nlargest(take, (key for bank in banks for key in bank))
        path = 1
    else:
        path = 2 if n > config.capacity else 3 if max(counts) > span else 4
        chosen = []
        for index in range(valid):
            key = _key(score(index), index)
            if len(chosen) < take:
                heapq.heappush(chosen, key)
            elif key > chosen[0]:
                heapq.heapreplace(chosen, key)
    ids = sorted(key & 0xFFFFFFFF for key in chosen)
    invalid = config.topk - take
    return ids + list(range(blocks - invalid, blocks)), (path, n, stored, valid)


def select_cpu(snapshot, config, *, expected_binding=None):
    if not isinstance(snapshot, ScoreSnapshot) or not isinstance(config, SelectiveConfig):
        raise TypeError("validated score snapshot and selective config required")
    if any(type(v) is not np.ndarray for v in (snapshot.query, snapshot.pooled, snapshot.positions)):
        raise TypeError("CPU score inputs must be NumPy ndarrays")
    _metadata(snapshot.query.shape, snapshot.pooled.shape, snapshot.positions.shape, config)
    snapshot.verify()
    if expected_binding is not None and expected_binding != snapshot.binding:
        raise ValueError("score stage binding differs from expected source")
    batch, rows = snapshot.query.shape[:2]
    blocks = snapshot.pooled.shape[1]
    ids = np.empty((batch, rows, config.topk), dtype=np.uint32)
    diagnostics = np.empty((batch, rows, 4), dtype=np.uint32)
    for b in range(batch):
        for row in range(rows):
            ids[b, row], diagnostics[b, row] = _row_select(
                lambda index, b=b, row=row: _score(snapshot, b, row, index), blocks,
                snapshot.positions[b, row], config)
    snapshot.verify()
    with _LOCK:
        _COUNTS["cpu_dispatches"] += 1
        for path, name in enumerate(("empty", "bounded", "overflow", "bank_overflow", "underfill")):
            _COUNTS[f"cpu_{name}_rows"] += int(np.count_nonzero(diagnostics[..., 0] == path))
    return ids, diagnostics, {"backend": "numpy_cpu", "binding": snapshot.binding,
                              "artifact_revision": snapshot.artifact_revision,
                              "source_revision": snapshot.source_revision,
                              "arithmetic": arithmetic_contract(), "config": asdict(config),
                              "qualified": False, "gpu_used": False}


def status(*, reset=False):
    with _LOCK:
        counts = dict(_COUNTS)
        if reset:
            _COUNTS.clear()
    return {"implemented": True, "qualified": False, "defaulted": False,
            "arithmetic": arithmetic_contract(), "counts": counts,
            "native_execution_observed": counts.get("native_completed_windows", 0) > 0}


def metal_source(config, *, input_type="float"):
    if input_type not in ("float", "half", "bfloat"):
        raise ValueError("unsupported Metal input type")
    defines = f"#define TOPK {config.topk}\n#define CAP {config.capacity}\n#define SAMPLES {config.samples}\n#define RATIO {config.ratio}\n#define INPUT_T {input_type}\n"
    return defines + _METAL_TEXT


def native_supported_metadata(qshape, pshape, positions_shape, config):
    _metadata(tuple(qshape), tuple(pshape), tuple(positions_shape), config)


@lru_cache(maxsize=8)
def _metal_kernel(config):
    import mlx.core as mx

    source = metal_source(config)
    # The standalone source compiles with a fixed INPUT_T. MLX substitutes T
    # through the same template ABI while retaining score/candidate code.
    source = "#define MLX_WRAPPED 1\n" + source.replace("#define INPUT_T float", "#define INPUT_T T")
    header, body = source.split("// MLX_BODY\n", 1)
    body = body.split("// MLX_END", 1)[0]
    return mx.fast.metal_kernel(
        name=f"mlx2_qsa_selective_scalar_k{config.topk}_c{config.capacity}_s{config.samples}_r{config.ratio}",
        input_names=["q", "pooled", "positions", "dims"],
        output_names=["block_ids", "diagnostics"], header=header, source=body,
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def select_metal(q, pooled, positions, *, block_topk, compress_ratio):
    """Explicit adapter backend; calling this schedules an unqualified kernel.

    No CPU/MLX fallback is allowed for an unsupported explicit request.
    Native completion and numerical parity require later owned GPU validation.
    """
    import mlx.core as mx

    config = SelectiveConfig(topk=block_topk, ratio=compress_ratio)
    _metadata(q.shape, pooled.shape, positions.shape, config)
    if (q.dtype != pooled.dtype or q.dtype not in (mx.float32, mx.float16, mx.bfloat16)
            or positions.dtype != mx.int32 or mx.default_device() != mx.gpu
            or not mx.metal.is_available()):
        raise ValueError("selective QSA native dtype/device capability unavailable")
    # Each invocation owns all stages; retained contiguous graph roots are
    # inputs to one kernel, so sampling/emission/recovery cannot bind new roots.
    q, pooled, positions = map(mx.contiguous, (q, pooled, positions))
    batch, rows = q.shape[:2]
    with _LOCK:
        _COUNTS["native_scheduled_dispatches"] += 1
    batches = []
    for b in range(batch):
        windows = []
        keys = mx.contiguous(pooled[b:b + 1])
        for start in range(0, rows, config.native_window_rows):
            stop = min(start + config.native_window_rows, rows)
            shape = mx.array([1, stop - start, pooled.shape[1]], dtype=mx.uint32)
            ids, diagnostics = _metal_kernel(config)(
                inputs=[mx.contiguous(q[b:b + 1, start:stop]), keys,
                        mx.contiguous(positions[b:b + 1, start:stop]), shape],
                template=[("T", q.dtype)],
                grid=((stop - start) * 32, 1, 1), threadgroup=(32, 1, 1),
                output_shapes=[(1, stop - start, config.topk), (1, stop - start, 4)],
                output_dtypes=[mx.uint32, mx.uint32],
            )
            # Explicit retirement bounds private/spilled workspace to one
            # row window. This reference backend favors boundedness over
            # host pipelining; no performance benefit has been established.
            mx.eval(ids, diagnostics)
            # Only a bounded four-integer diagnostic row crosses to the host;
            # score rows remain private to the producer. Reference-mode
            # synchronous readback is deliberate and included in its caveat.
            row_diagnostics = diagnostics.tolist()[0]
            del diagnostics
            windows.append(ids)
            with _LOCK:
                _COUNTS["native_completed_windows"] += 1
                for path, count, stored, _valid in row_diagnostics:
                    name = ("empty", "bounded", "overflow", "bank_overflow", "underfill")[path]
                    _COUNTS[f"native_{name}_rows"] += 1
                    _COUNTS["native_emitted_candidates"] += count
                    _COUNTS["native_stored_candidates"] += stored
                    _COUNTS["native_max_stored_candidates"] = max(
                        _COUNTS["native_max_stored_candidates"], stored)
        batches.append(mx.concatenate(windows, axis=1) if len(windows) > 1 else windows[0])
    return mx.concatenate(batches, axis=0) if len(batches) > 1 else batches[0]
