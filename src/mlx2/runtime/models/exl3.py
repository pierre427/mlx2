"""EXL3 trellis linears and expert banks on the Sushi-derived kernels.

MIT Sushi ad562c4 Metal bodies; adapter-owned model math, no scheduler/cache
changes. See provenance/sushi-integration-20261012.json. The explicit Sushi
pack marker stays 2.5-bit MCG; upstream checkpoints (runtime/trellis) reuse
these modules at any even tile word count. Unqualified format.
"""

from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu

FORMAT_REVISION = "mlx2-exl3-mcg-w16-n40-v1"
ASSETS = Path(__file__).with_name("exl3_kernels")


def validate_metadata(meta):
    if not isinstance(meta, dict) or set(meta) != {
        "version",
        "codebook",
        "window_bits",
        "bits_per_weight",
        "execution",
    }:
        raise ValueError(
            "EXL3 requires explicit version/codebook/window_bits/bits_per_weight/execution"
        )
    if (
        type(meta["version"]) is not int
        or meta["version"] != 1
        or meta["codebook"] != "mcg"
        or type(meta["window_bits"]) is not int
        or meta["window_bits"] != 16
        or type(meta["bits_per_weight"]) not in (float, int)
        or meta["bits_per_weight"] != 2.5
        or meta["execution"] not in ("reference", "fused")
    ):
        raise ValueError(
            "only version-1 2.5-bit MCG/w16 EXL3 reference or fused is implemented"
        )
    return dict(meta)


@lru_cache(maxsize=1)
def _kernels():
    def kernel(name, inputs, outputs, file, header=""):
        return mx.fast.metal_kernel(
            name="mlx2_exl3_" + name,
            input_names=inputs,
            output_names=outputs,
            source=(ASSETS / file).read_text(),
            header=header,
        )

    header = (ASSETS / "header_mcg_w16.metal").read_text()
    return (
        kernel("prepare", ["x", "suh", "slots"], ["y"], "PREPARE_SOURCE.metal"),
        kernel(
            "inner",
            ["x", "trellis", "slots"],
            ["y"],
            "INDEXED_COOP_SOURCE.metal",
            header,
        ),
        kernel("finish", ["inner", "svh", "slots"], ["y"], "FINISH_SOURCE.metal"),
        kernel(
            "pair",
            ["x", "suhg", "suhu", "tg", "tu", "slots"],
            ["yg", "yu"],
            "PAIR_GEMV_SOURCE.metal",
            header,
        ),
        kernel("rows", ["x", "trellis", "order", "eids"], ["y"], "ROWS_GEMV_SOURCE.metal", header),
        kernel("fused_linear", ["x", "trellis", "suh", "svh"], ["y"], "FUSED_LINEAR_SOURCE.metal", header),
        kernel("reconstruct", ["trellis", "experts"], ["inner"], "RECONSTRUCT_SOURCE.metal", header),
    )


# Row-blocked law (ROWS_GEMV_SOURCE): block b covers rows order[b*R..b*R+R)
# sorted by expert; a row's arithmetic is the same at every R and width.
# Four rows per block: eight measured slower on an M5 Max (register
# pressure; lm_head at 512 rows 377 vs 185 ms), same per-row arithmetic.
ROWS_PER_BLOCK = 4
# Bank calls this wide use eight-row blocks (rows of one expert share a tile
# decode); narrower bank calls, mostly distinct experts, use one-row blocks.
GROUP_MIN_ROWS = 32
# fused_nax: dense calls this wide reconstruct fp16 weights and run
# mx.matmul (the M5 tensor units); narrower calls keep the fused kernel.
NAX_MIN_ROWS = 64
# Columns reconstructed per matmul: bounds the fp16 weight transient
# (lm_head: 2048 x 16384 x 2 B = 64 MiB per slice).
NAX_SLICE_COLUMNS = 16384
# rows_nax: bank calls this wide (128 tokens at top-8) reconstruct their
# experts in chunks and run sorted gather_mm on the tensor units.
NAX_BANK_MIN_ROWS = 1024
# Experts reconstructed per gather_mm (64 x 2048 x 512 x 2 B = 128 MiB).
NAX_BANK_CHUNK = 64


class NaxLayout(NamedTuple):
    order: mx.array
    inverse: mx.array
    sorted_ids: mx.array
    bounds: tuple


def nax_layout(ids, num_experts):
    """Expert-sorted rows and each expert's segment (one host sync)."""
    import numpy as np

    host = np.asarray(ids.astype(mx.uint32))
    order = np.argsort(host, kind="stable").astype(np.uint32)
    ranked = host[order]
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size, dtype=np.uint32)
    bounds = tuple(int(b) for b in np.searchsorted(ranked, np.arange(num_experts + 1)))
    return NaxLayout(mx.array(order), mx.array(inverse), mx.array(ranked), bounds)


def nax_bank(proj, rows, ids, layout):
    """One bank projection at prefill width: prepare, per expert chunk
    reconstruct + sorted gather_mm, finish (rows in caller order)."""
    prepared = proj.prepare(rows, ids)[layout.order]
    cpu = mx.default_device() == mx.cpu
    parts = []
    for e0 in range(0, proj.num_experts, NAX_BANK_CHUNK):
        e1 = min(e0 + NAX_BANK_CHUNK, proj.num_experts)
        s0, s1 = layout.bounds[e0], layout.bounds[e1]
        if s0 == s1:
            continue
        weights = proj.reconstruct(mx.arange(e0, e1, dtype=mx.uint32))
        lhs = prepared[s0:s1][:, None, :]
        if cpu:  # MLX's CPU gather GEMM has no half kernel
            lhs, weights = lhs.astype(mx.float32), weights.astype(mx.float32)
        part = mx.gather_mm(lhs, weights, rhs_indices=layout.sorted_ids[s0:s1] - e0,
                            sorted_indices=True)
        parts.append(part.squeeze(-2).astype(mx.float16))
    inner = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)
    return proj.finish(inner[layout.inverse], ids)
# Sushi's fused gate/up pair (PAIR_GEMV_SOURCE) was validated at 40-word tiles
# only; at 32 and 64 words it measured 82-86% of output scale off the reference
# switch on Qwen3.6-35B-A3B layer-10 experts (exl3-phase3-metal-20261104).
PAIR_VALIDATED_WORDS = 40
# Revision of the row law; part of the install receipt and identity.
ROWS_LAW_REVISION = "mlx2-exl3-rows-v4"


class RowPlan(NamedTuple):
    order: mx.array
    eids: mx.array
    rows: int
    dense: bool = False


_CONSTANTS = {}


def _constant(kind, count):
    """Immutable per-width index arrays, built once (they are MLX constants)."""
    key = (kind, count)
    if key not in _CONSTANTS:
        _CONSTANTS[key] = (mx.zeros if kind == "zeros" else mx.arange)(count, dtype=mx.uint32)
    return _CONSTANTS[key]


def row_plan(ids, num_experts, *, rows=None):
    """Device-side plan: rows sorted by expert, block size by call width.

    ``rows`` overrides the block size (timing studies); a row's arithmetic
    is the same at every block size.
    """
    if rows is not None and not 1 <= rows <= ROWS_PER_BLOCK:
        raise ValueError(f"row blocks hold 1..{ROWS_PER_BLOCK} rows")
    ids = ids.astype(mx.uint32)
    count = ids.size
    if num_experts == 1:
        size = rows or (ROWS_PER_BLOCK if count > 1 else 1)
        return RowPlan(_constant("arange", count), ids, size, True)
    size = rows or (ROWS_PER_BLOCK if count >= GROUP_MIN_ROWS else 1)
    order = mx.argsort(ids).astype(mx.uint32) if size > 1 else _constant("arange", count)
    return RowPlan(order, ids, size)


def _h128(x):
    shape = x.shape
    if mx.default_device() == mx.cpu:
        import numpy as np

        a = np.asarray(x, dtype=np.float32).reshape(-1, 128).copy()
        for bit in (1, 2, 4, 8, 16, 32, 64):
            block = a.reshape(-1, 128 // (2 * bit), 2, bit)
            lo, hi = block[:, :, 0, :].copy(), block[:, :, 1, :].copy()
            block[:, :, 0, :], block[:, :, 1, :] = lo + hi, lo - hi
        return mx.array((a * np.float32(1 / (128**0.5))).reshape(shape))
    return mx.hadamard_transform(x.reshape(-1, 128), scale=1 / (128**0.5)).reshape(
        shape
    )


LAWS = ("reference", "rows")


class Exl3Linear(nn.Module):
    def __init__(self, trellis, suh, svh, *, law="reference"):
        super().__init__()
        if law not in LAWS:
            raise ValueError(f"EXL3 law must be one of {LAWS}")
        if (
            trellis.dtype != mx.uint16
            or trellis.ndim != 4
            or trellis.shape[-1] % 2
            or not 16 <= trellis.shape[-1] <= 128
            or suh.dtype != mx.float16
            or svh.dtype != mx.float16
            or suh.shape != (trellis.shape[0], trellis.shape[1] * 16)
            or svh.shape != (trellis.shape[0], trellis.shape[2] * 16)
            or trellis.shape[0] < 1
            or suh.shape[-1] % 128
            or svh.shape[-1] % 128
        ):
            raise ValueError(
                "EXL3 requires [E,I/16,O/16,n] U16 (even n in 16..128) "
                "and matching H128 F16 vectors"
            )
        if not bool(mx.all(mx.isfinite(suh)).item()) or not bool(
            mx.all(mx.isfinite(svh)).item()
        ):
            raise ValueError("EXL3 scale vectors must be finite")
        self.trellis, self.suh, self.svh = trellis, suh, svh
        self.law = law
        self.freeze()

    @property
    def input_dims(self):
        return self.suh.shape[-1]

    @property
    def output_dims(self):
        return self.svh.shape[-1]

    @property
    def num_experts(self):
        return self.trellis.shape[0]

    @property
    def words(self):
        return self.trellis.shape[-1]

    @property
    def packed_nbytes(self):
        return self.trellis.nbytes + self.suh.nbytes + self.svh.nbytes

    def prepare(self, x, ids):
        if mx.default_device() == mx.cpu:
            return _h128(
                x.astype(mx.float32) * self.suh[ids].astype(mx.float32)
            ).astype(mx.float16)
        return _kernels()[0](
            inputs=[x, self.suh, ids],
            template=[("IDIM", self.input_dims)],
            grid=(self.input_dims // 128 * 32, ids.size, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(ids.size, self.input_dims)],
            output_dtypes=[mx.float16],
        )[0]

    def finish(self, inner, ids):
        if mx.default_device() == mx.cpu:
            return (
                _h128(inner.astype(mx.float32)) * self.svh[ids].astype(mx.float32)
            ).astype(mx.float16)
        return _kernels()[2](
            inputs=[inner, self.svh, ids],
            template=[("ODIM", self.output_dims)],
            grid=(self.output_dims // 128 * 32, ids.size, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(ids.size, self.output_dims)],
            output_dtypes=[mx.float16],
        )[0]

    def reconstruct(self, experts, ot0=0, columns=None, *, transposed=False):
        """Decoded fp16 inner weights [U, I, columns] ([U, columns, I] when
        transposed) from output tile ot0."""
        columns = self.output_dims - ot0 * 16 if columns is None else columns
        if columns % 128 or ot0 * 16 + columns > self.output_dims:
            raise ValueError("EXL3 reconstruction slices whole 128-column blocks")
        experts = experts.astype(mx.uint32)
        if mx.default_device() == mx.cpu:
            import numpy as np

            from .exl3_reference import decode_inner

            tiles = np.asarray(self.trellis[:, :, ot0 : ot0 + columns // 16])
            out = np.stack([decode_inner(tiles[int(e)]) for e in np.asarray(experts)])
            return mx.array(np.ascontiguousarray(out.transpose(0, 2, 1)) if transposed else out)
        return _kernels()[6](
            inputs=[self.trellis, experts],
            template=[
                ("IDIM", self.input_dims),
                ("ODIM", self.output_dims),
                ("NHW", self.words),
                ("OT0", ot0),
                ("OSLICE", columns),
                ("TRANS", transposed),
            ],
            grid=(columns // 16 * 32, self.input_dims // 16, experts.size),
            threadgroup=(32, 1, 1),
            output_shapes=[(experts.size, columns, self.input_dims) if transposed
                           else (experts.size, self.input_dims, columns)],
            output_dtypes=[mx.float16],
        )[0]

    def _rows(self, prepared, plan):
        """Row-blocked inner product; float32 rows in the caller's order."""
        rows = prepared.shape[0]
        if mx.default_device() == mx.cpu:
            import numpy as np

            from .exl3_reference import decode_inner

            host = np.asarray(prepared.astype(mx.float32))
            eids = np.asarray(plan.eids)
            value = np.zeros((rows, self.output_dims), np.float32)
            for eid in np.unique(eids):
                inner = decode_inner(np.asarray(self.trellis[int(eid)])).astype(np.float32)
                value[eids == eid] = host[eids == eid] @ inner
            return mx.array(value)
        if rows == 0:
            return mx.zeros((0, self.output_dims), mx.float32)
        return _kernels()[4](
            inputs=[prepared, self.trellis, plan.order, plan.eids],
            template=[
                ("IDIM", self.input_dims),
                ("ODIM", self.output_dims),
                ("NHW", self.words),
                ("ROWS", plan.rows),
                ("DENSE", plan.dense),
            ],
            grid=(self.output_dims // 16 * 128, -(-rows // plan.rows), 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(rows, self.output_dims)],
            output_dtypes=[mx.float32],
        )[0]

    def __call__(self, x, ids, *, trusted_indices=False, plan=None):
        if x.ndim != 2 or x.shape != (ids.size, self.input_dims) or ids.ndim != 1:
            raise ValueError(
                "EXL3 projection requires one input vector per routed slot"
            )
        if ids.dtype != mx.uint32:
            raise ValueError("EXL3 slot IDs must be uint32")
        if (
            not trusted_indices
            and ids.size
            and int(mx.max(ids).item()) >= self.num_experts
        ):
            raise ValueError("EXL3 expert index outside the bound bank")
        prepared = self.prepare(x, ids)
        if self.law == "rows":
            return self.finish(self._rows(prepared, plan or row_plan(ids, self.num_experts)), ids)
        if mx.default_device() == mx.cpu:
            import numpy as np
            from .exl3_reference import decode_inner

            inner = mx.array(decode_inner(np.asarray(self.trellis)))
            # MLX's CPU gather backend has no half GEMM. Accumulate float32
            # then round at the packed projection's F16 boundary.
            value = (
                mx.gather_mm(
                    prepared.astype(mx.float32)[:, None, :],
                    inner.astype(mx.float32),
                    rhs_indices=ids,
                )
                .squeeze(-2)
                .astype(mx.float16)
            )
        else:
            value = _kernels()[1](
                inputs=[prepared, self.trellis, ids],
                template=[
                    ("IDIM", self.input_dims),
                    ("ODIM", self.output_dims),
                    ("NHW", self.words),
                ],
                grid=(self.output_dims // 16 * 128, ids.size, 1),
                threadgroup=(128, 1, 1),
                output_shapes=[(ids.size, self.output_dims)],
                output_dtypes=[mx.float16],
            )[0]
        return self.finish(value, ids)


class Exl3SwitchGLU(nn.Module):
    def __init__(self, gate, up, down, *, execution, trusted_router=False):
        super().__init__()
        if (
            gate.input_dims != up.input_dims
            or gate.output_dims != up.output_dims
            or down.input_dims != gate.output_dims
            or down.output_dims != gate.input_dims
            or len({gate.num_experts, up.num_experts, down.num_experts}) != 1
        ):
            raise ValueError("EXL3 gate/up/down geometry mismatch")
        if execution not in ("reference", "fused", "rows", "rows_nax"):
            raise ValueError("EXL3 switch execution must be reference, fused, rows or rows_nax")
        row_law = execution in ("rows", "rows_nax")
        if row_law != all(p.law == "rows" for p in (gate, up, down)):
            raise ValueError("rows execution needs row-law projections, and only it")
        self.gate_proj, self.up_proj, self.down_proj = gate, up, down
        self.execution = execution
        self.trusted_router = trusted_router
        # The Qwen adapter's one-token dispatcher reads this switch contract.
        # Affine routed kernels cannot consume trellis storage.
        self.routed_decode_mode = "off"
        # Current adapter diagnostics read the ordinary switch counter too.
        object.__setattr__(self, "routed_decode_calls", 0)
        object.__setattr__(self, "fused_calls", 0)
        object.__setattr__(self, "reference_calls", 0)
        object.__setattr__(self, "rows_calls", 0)
        object.__setattr__(self, "nax_calls", 0)
        self.freeze()

    def __call__(self, x, indices):
        if self.training:
            raise ValueError("EXL3 experts are inference-only")
        if (
            indices.ndim < 1
            or x.ndim < 2
            or x.shape[-1] != self.gate_proj.input_dims
            or x.shape[:-1] != indices.shape[:-1]
            or not 1 <= indices.shape[-1] <= 32
            or indices.size == 0
        ):
            raise ValueError("EXL3 expects matching token rows and top-k 1..32")
        ids = indices.reshape(-1).astype(mx.uint32)
        if ids.size > 65535:
            raise ValueError("EXL3 call exceeds the implemented kernel grid bound")
        if (
            not self.trusted_router
            and ids.size
            and int(mx.max(ids).item()) >= self.gate_proj.num_experts
        ):
            raise ValueError("EXL3 expert index outside the bound bank")
        topk = indices.shape[-1]
        flat = x.reshape(-1, x.shape[-1])
        rows = mx.repeat(flat, topk, axis=0)
        fused = self.execution == "fused" and mx.default_device() != mx.cpu
        if fused and not self.gate_proj.words == self.up_proj.words == PAIR_VALIDATED_WORDS:
            raise ValueError(
                f"the Sushi fused gate/up pair is validated only at {PAIR_VALIDATED_WORDS}-word tiles"
            )
        if fused:
            i, o = self.gate_proj.input_dims, self.gate_proj.output_dims
            splits = 2 if i % 256 == 0 else 1
            partials = _kernels()[3](
                inputs=[
                    flat,
                    self.gate_proj.suh,
                    self.up_proj.suh,
                    self.gate_proj.trellis,
                    self.up_proj.trellis,
                    ids,
                ],
                template=[
                    ("IDIM", i),
                    ("ODIM", o),
                    ("NSPLIT", splits),
                    ("NHW", self.gate_proj.words),
                    ("TOPK", topk),
                    ("FUNNEL", True),
                    ("OTPT", 2),
                ],
                grid=(o // 32 * 128, ids.size, splits),
                threadgroup=(128, 1, 1),
                output_shapes=[(ids.size * splits, o)] * 2,
                output_dtypes=[mx.float32] * 2,
            )
            g, u = [
                proj.finish(value.reshape(ids.size, splits, o).sum(axis=1), ids)
                for value, proj in zip(partials, (self.gate_proj, self.up_proj))
            ]
            object.__setattr__(self, "fused_calls", self.fused_calls + 1)
        elif self.execution == "rows_nax" and ids.size >= NAX_BANK_MIN_ROWS:
            # Prefill widths: expert-sorted reconstruction + gather_mm, one
            # layout for all three projections.
            layout = nax_layout(ids, self.gate_proj.num_experts)
            g = nax_bank(self.gate_proj, rows, ids, layout)
            u = nax_bank(self.up_proj, rows, ids, layout)
            out = nax_bank(self.down_proj, swiglu(g, u), ids, layout).astype(x.dtype)
            object.__setattr__(self, "nax_calls", self.nax_calls + 1)
            return out.reshape(*indices.shape, self.down_proj.output_dims)
        elif self.execution in ("rows", "rows_nax"):
            # One plan for all three projections: they share the routed rows.
            plan = row_plan(ids, self.gate_proj.num_experts)
            g, u = (
                self.gate_proj(rows, ids, trusted_indices=True, plan=plan),
                self.up_proj(rows, ids, trusted_indices=True, plan=plan),
            )
            object.__setattr__(self, "rows_calls", self.rows_calls + 1)
        else:
            plan = None
            g, u = (
                self.gate_proj(rows, ids, trusted_indices=True),
                self.up_proj(rows, ids, trusted_indices=True),
            )
            object.__setattr__(self, "reference_calls", self.reference_calls + 1)
        # Keep output in the trunk activation dtype so the existing router's
        # weighted tail and shared expert retain their dtype contract.
        down_plan = plan if self.execution in ("rows", "rows_nax") else None
        out = self.down_proj(
            swiglu(g, u), ids, trusted_indices=True, plan=down_plan
        ).astype(x.dtype)
        return out.reshape(*indices.shape, self.down_proj.output_dims)

    def receipt(self):
        return {
            "format_revision": FORMAT_REVISION,
            "execution": self.execution,
            "qualification": "unqualified",
            "fused_calls": self.fused_calls,
            "reference_calls": self.reference_calls,
            "rows_calls": self.rows_calls,
            "nax_calls": self.nax_calls,
            "packed_weight_bytes": sum(
                p.packed_nbytes for p in (self.gate_proj, self.up_proj, self.down_proj)
            ),
        }


def install_exl3_experts(model, weights, metadata):
    """Adapter load hook: admit all split expert banks before replacing any.

    Keys use Sushi's split ``switch_mlp.{gate,up,down}_proj.{trellis,suh,svh}``.
    Unmarked trellis tensors and mixed affine/EXL3 blocks fail closed.
    """
    meta = validate_metadata(metadata)
    # The marker declares one rate; every bank must carry exactly its tiles.
    words = int(meta["bits_per_weight"] * 16)
    if any(v.shape[-1] != words for k, v in weights.items() if k.endswith(".trellis")):
        raise ValueError(f"EXL3 pack marker declares {words}-word tiles")
    replacements = install_banks(model, weights, execution=meta["execution"])
    return {
        "format_revision": FORMAT_REVISION,
        **meta,
        "qualification": "unqualified",
        "blocks": len(replacements),
        "packed_weight_bytes": sum(m.receipt()["packed_weight_bytes"] for m in replacements),
    }


def install_banks(model, weights, *, execution):
    """Replace every routed ``switch_mlp`` with an ``Exl3SwitchGLU``.

    All banks are admitted before any module is swapped; returns the new
    switches. Callers own format metadata and its identity.
    """
    modules = dict(model.named_modules())
    roots = {key.rsplit(".", 2)[0] for key in weights if key.endswith(".trellis")}
    if not roots:
        raise ValueError("EXL3 metadata has no packed expert tensors")
    replacements = []
    for root in sorted(roots):
        old = modules.get(root)
        if old is None or not root.endswith(".switch_mlp") or "down_proj" not in old:
            raise ValueError(f"EXL3 tensor has no supported switch module: {root}")
        parent = modules.get(root.rpartition(".")[0])
        if parent is None or getattr(parent, "shared_folded", False):
            raise ValueError(
                "EXL3 does not support shared experts folded into routed tables"
            )
        down = old.down_proj
        allowed = {
            f"{root}.{name}.{part}"
            for name in ("gate_proj", "up_proj", "down_proj")
            for part in ("trellis", "suh", "svh")
        }
        if any(key.startswith(root + ".") and key not in allowed for key in weights):
            raise ValueError("mixed EXL3/affine expert tensors are unsupported")
        projs = []
        for name in ("gate_proj", "up_proj", "down_proj"):
            keys = [f"{root}.{name}.{k}" for k in ("trellis", "suh", "svh")]
            if any(key not in weights for key in keys):
                raise ValueError(f"incomplete EXL3 block: {root}")
            proj = Exl3Linear(
                *(weights[key] for key in keys),
                law="rows" if execution in ("rows", "rows_nax") else "reference",
            )
            expected = (
                (down.input_dims, down.output_dims)
                if name == "down_proj"
                else (down.output_dims, down.input_dims)
            )
            if (
                proj.input_dims,
                proj.output_dims,
            ) != expected or proj.num_experts != down.num_experts:
                raise ValueError(
                    f"EXL3 pack disagrees with model topology: {root}.{name}"
                )
            if any(
                key.startswith(f"{root}.{name}.") and key not in keys for key in weights
            ):
                raise ValueError("mixed EXL3/affine expert tensors are unsupported")
            projs.append(proj)
        replacements.append(
            (
                root,
                parent,
                Exl3SwitchGLU(*projs, execution=execution, trusted_router=True),
            )
        )
    # Refuse partial trunks: loading a mixed model must be an explicit future
    # format, not an accidental omission of one packed layer.
    supported = {
        p for p, m in modules.items() if p.endswith(".switch_mlp") and "down_proj" in m
    }
    if roots != supported:
        raise ValueError("EXL3 metadata must cover every routed switch block")
    from ..weight_stream import _swap_module

    for root, parent, replacement in replacements:
        _swap_module(model, root, replacement)
        parent.fused_gate_up = False
        parent.fused_expert_kernel_mode = "stock"
        parent.moe_weighted_sum = False
    return [replacement for _, _, replacement in replacements]


class Exl3DenseLinear(nn.Module):
    """One stored EXL3 linear, run on the indexed kernels as a one-expert bank.

    Replaces an ``nn.Linear`` of the same geometry; output keeps the input
    dtype. Ordinary per-row projection only (no wide-row reconstruct yet).
    """

    # One kernel launch addresses at most this many rows (grid y).
    ROWS_PER_LAUNCH = 65535

    LAWS = ("reference", "rows", "fused", "fused_nax", "fused_w8a8")

    def __init__(self, trellis, suh, svh, bias=None, *, law="reference"):
        super().__init__()
        if law not in self.LAWS:
            raise ValueError(f"EXL3 dense law must be one of {self.LAWS}")
        # "fused": FUSED_LINEAR_SOURCE, one launch per call at any width.
        # "fused_nax": fused below NAX_MIN_ROWS rows, fp16 reconstruction and
        # mx.matmul at or above (two laws split by width, like affine qmv/qmm).
        # "fused_w8a8": fused below NAX_MIN_ROWS, approximate W8A8 above: the
        # prepared rows and the reconstructed weights quantized per row and per
        # output channel to int8, int8 GEMM on the tensor units (int8_prefill).
        self.law = law
        self.bank = Exl3Linear(trellis[None], suh[None], svh[None],
                               law="rows" if law == "rows" else "reference")
        if bias is not None:
            if bias.shape != (self.bank.output_dims,):
                raise ValueError("EXL3 bias must match the output width")
            self.bias = bias
        object.__setattr__(self, "calls", 0)
        object.__setattr__(self, "rows", 0)
        self.freeze()

    @property
    def input_dims(self):
        return self.bank.input_dims

    @property
    def output_dims(self):
        return self.bank.output_dims

    def _nax(self, flat):
        """prepare, then per column slice reconstruct + matmul, then finish."""
        bank = self.bank
        ids = _constant("zeros", flat.shape[0])
        prepared = bank.prepare(flat, ids)
        zero = _constant("zeros", 1)
        inner = []
        for lo in range(0, bank.output_dims, NAX_SLICE_COLUMNS):
            columns = min(NAX_SLICE_COLUMNS, bank.output_dims - lo)
            weight = bank.reconstruct(zero, lo // 16, columns)[0]
            inner.append(mx.matmul(prepared, weight))
        inner = inner[0] if len(inner) == 1 else mx.concatenate(inner, axis=1)
        return bank.finish(inner, ids)

    def _w8a8(self, flat):
        """Approximate: int8 rows x int8 output channels, then finish."""
        from ..int8_prefill import (
            MPP_SLICE_OFFSET_LIMIT, _int8_gemm, _quantize_rows, max_tensor_slice_offset,
        )

        bank = self.bank
        width = min(NAX_SLICE_COLUMNS, bank.output_dims)
        if max_tensor_slice_offset(rows=flat.shape[0], k=bank.input_dims, n=width) >= MPP_SLICE_OFFSET_LIMIT:
            # MPP tensor slices past 2 GiB land low on M5 (see int8_prefill).
            object.__setattr__(self, "w8a8_fallbacks", getattr(self, "w8a8_fallbacks", 0) + 1)
            return self._nax(flat)
        ids = _constant("zeros", flat.shape[0])
        xq, xs = _quantize_rows(bank.prepare(flat, ids))
        zero = _constant("zeros", 1)
        inner = []
        for lo in range(0, bank.output_dims, NAX_SLICE_COLUMNS):
            columns = min(NAX_SLICE_COLUMNS, bank.output_dims - lo)
            wq, ws = _quantize_rows(bank.reconstruct(zero, lo // 16, columns, transposed=True)[0])
            inner.append(_int8_gemm(xq, xs, wq, ws))
        inner = inner[0] if len(inner) == 1 else mx.concatenate(inner, axis=1)
        return bank.finish(inner, ids)

    def _fused(self, flat):
        bank = self.bank
        return _kernels()[5](
            inputs=[flat, bank.trellis, bank.suh, bank.svh],
            template=[
                ("IDIM", bank.input_dims),
                ("ODIM", bank.output_dims),
                ("NHW", bank.words),
                ("OUT", flat.dtype),
            ],
            grid=(bank.output_dims // 128 * 256, flat.shape[0], 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(flat.shape[0], bank.output_dims)],
            output_dtypes=[flat.dtype],
        )[0]

    @property
    def words(self):
        return self.bank.words

    def __call__(self, x):
        if self.training:
            raise ValueError("EXL3 linears are inference-only")
        if x.shape[-1] != self.input_dims:
            raise ValueError("EXL3 linear input width mismatch")
        flat = x.reshape(-1, self.input_dims)
        rows = flat.shape[0]
        if self.law in ("fused_nax", "fused_w8a8") and rows >= NAX_MIN_ROWS:
            wide = self._nax(flat) if self.law == "fused_nax" else self._w8a8(flat)
            out = wide.astype(x.dtype).reshape(*x.shape[:-1], self.output_dims)
            if "bias" in self:
                out = out + self.bias.astype(x.dtype)
            object.__setattr__(self, "calls", self.calls + 1)
            object.__setattr__(self, "rows", self.rows + rows)
            object.__setattr__(self, "nax_calls", getattr(self, "nax_calls", 0) + 1)
            return out
        if self.law in ("fused", "fused_nax", "fused_w8a8") and mx.default_device() != mx.cpu and rows:
            out = self._fused(flat).reshape(*x.shape[:-1], self.output_dims)
            if "bias" in self:
                out = out + self.bias.astype(x.dtype)
            object.__setattr__(self, "calls", self.calls + 1)
            object.__setattr__(self, "rows", self.rows + rows)
            return out
        outs = []
        for lo in range(0, rows, self.ROWS_PER_LAUNCH):
            part = flat[lo : lo + self.ROWS_PER_LAUNCH]
            ids = _constant("zeros", part.shape[0])
            outs.append(self.bank(part, ids, trusted_indices=True))
        out = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
        out = out.astype(x.dtype).reshape(*x.shape[:-1], self.output_dims)
        if "bias" in self:
            out = out + self.bias.astype(x.dtype)
        object.__setattr__(self, "calls", self.calls + 1)
        object.__setattr__(self, "rows", self.rows + rows)
        return out
