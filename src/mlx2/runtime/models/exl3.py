"""Explicit 2.5-bit MCG EXL3 expert format; ordinary projection retained.

MIT Sushi ad562c4 Metal bodies; adapter-owned model math, no scheduler/cache
changes. See provenance/sushi-integration-20261012.json. Unqualified format.
"""

from functools import lru_cache
from pathlib import Path

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
    )


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


class Exl3Linear(nn.Module):
    def __init__(self, trellis, suh, svh):
        super().__init__()
        if (
            trellis.dtype != mx.uint16
            or trellis.ndim != 4
            or trellis.shape[-1] != 40
            or suh.dtype != mx.float16
            or svh.dtype != mx.float16
            or suh.shape != (trellis.shape[0], trellis.shape[1] * 16)
            or svh.shape != (trellis.shape[0], trellis.shape[2] * 16)
            or trellis.shape[0] < 1
            or suh.shape[-1] % 128
            or svh.shape[-1] % 128
        ):
            raise ValueError(
                "EXL3 requires [E,I/16,O/16,40] U16 and matching H128 F16 vectors"
            )
        if not bool(mx.all(mx.isfinite(suh)).item()) or not bool(
            mx.all(mx.isfinite(svh)).item()
        ):
            raise ValueError("EXL3 scale vectors must be finite")
        self.trellis, self.suh, self.svh = trellis, suh, svh
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

    def __call__(self, x, ids, *, trusted_indices=False):
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
                    ("NHW", 40),
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
                    ("NHW", 40),
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
        else:
            g, u = (
                self.gate_proj(rows, ids, trusted_indices=True),
                self.up_proj(rows, ids, trusted_indices=True),
            )
            object.__setattr__(self, "reference_calls", self.reference_calls + 1)
        # Keep output in the trunk activation dtype so the existing router's
        # weighted tail and shared expert retain their dtype contract.
        out = self.down_proj(swiglu(g, u), ids, trusted_indices=True).astype(x.dtype)
        return out.reshape(*indices.shape, self.down_proj.output_dims)

    def receipt(self):
        return {
            "format_revision": FORMAT_REVISION,
            "execution": self.execution,
            "qualification": "unqualified",
            "fused_calls": self.fused_calls,
            "reference_calls": self.reference_calls,
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
            proj = Exl3Linear(*(weights[key] for key in keys))
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
                Exl3SwitchGLU(*projs, execution=meta["execution"], trusted_router=True),
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
    return {
        "format_revision": FORMAT_REVISION,
        **meta,
        "qualification": "unqualified",
        "blocks": len(replacements),
        "packed_weight_bytes": sum(
            m.receipt()["packed_weight_bytes"] for _, _, m in replacements
        ),
    }
