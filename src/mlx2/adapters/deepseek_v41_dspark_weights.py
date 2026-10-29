# SPDX-License-Identifier: MIT
"""Admitted resident Source DSpark tensors; target embedding/head stay shared."""

from contextlib import nullcontext

import numpy as np

from .deepseek_v41_dspark_artifact import aliases, dspark_admission_plan
from .deepseek_v41_source_artifact import checked_file, read_tensors


class SourceDSparkWeights:
    def __init__(self, artifact, target_weights, *, admitted_bytes, load=True):
        from .deepseek_v41_mlx_weights import _refuse_quarantined_uploads
        _refuse_quarantined_uploads()
        self.plan = dspark_admission_plan(artifact)
        if (
            type(admitted_bytes) is not int
            or admitted_bytes < self.plan["reservation_bytes"]
        ):
            raise MemoryError(
                "embedded DSpark requires pre-materialization reservation"
            )
        self.artifact, self.target = artifact, target_weights
        self._dense, self._quant, self._experts = {}, {}, {}
        self._closed = False
        self.scheduled_expert_calls = {}
        self.scheduled_dense_calls = {}
        self.expert_stream = None
        self._loaded = False
        self._pending_host = None
        self._pending_arrays = []
        if load:
            self.load()

    def load(self):
        if self._closed or self._loaded or self._dense or self._quant or self._experts:
            raise ValueError("DSpark weight owner cannot load twice or after retirement")
        try:
            mapping = aliases()
            for alias, name in mapping.items():
                self._load(alias, name)
            for stage in range(3):
                for expert in range(128):
                    for part in ("w1", "w3", "w2"):
                        name = f"mtp.{stage}.ffn.experts.{expert}.{part}"
                        raw = read_tensors(
                            self.artifact, (name + ".weight", name + ".scale")
                        )
                        import mlx.core as mx

                        self._pending_host = raw
                        q = mx.array(raw[name + ".weight"].view("<u4"))
                        self._pending_arrays.append(q)
                        scale = mx.array(raw[name + ".scale"])
                        self._pending_arrays.append(scale)
                        mx.eval(q, scale)
                        self._experts[(stage, expert, part)] = (q, scale)
                        self._complete_upload()
            self.verify_source()
            self._loaded = True
            return self
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:  # noqa: BLE001 - quarantine remains and original failure is re-raised
                error.add_note("DSpark failed-load owner retained after cleanup fence failure: " + repr(cleanup))
            raise

    def _complete_upload(self):
        self._pending_host = None
        self._pending_arrays.clear()

    def _load(self, alias, name):
        import mlx.core as mx

        row = self.artifact["tensors"][name]
        scale = name.removesuffix("weight") + "scale"
        names = (name, scale) if row["dtype"] == "F8_E4M3" else (name,)
        raw = read_tensors(self.artifact, names)
        self._pending_host = raw
        if len(names) == 2:
            q = mx.array(raw[name].view("<u4"))
            self._pending_arrays.append(q)
            scales = mx.repeat(mx.array(raw[scale]), 32, axis=0)[: row["shape"][0]]
            self._pending_arrays.append(scales)
            if alias.endswith("attn_output_a.weight"):
                value = mx.dequantize(q, scales=scales, mode="mxfp8", dtype=mx.bfloat16)
                self._pending_arrays.append(value)
                mx.eval(value)
                self._dense[alias] = value
            else:
                mx.eval(q, scales)
                self._quant[alias] = (q, scales)
        else:
            value = mx.array(raw[name])
            self._pending_arrays.append(value)
            if row["dtype"] == "BF16":
                value = value.view(mx.bfloat16)
                self._pending_arrays.append(value)
            mx.eval(value)
            self._dense[alias] = value
        self._complete_upload()

    def verify_source(self):
        if self._closed:
            raise ValueError("DSpark weight owner is closed")
        for name in {
            r["file"]
            for n, r in self.artifact["tensors"].items()
            if n.startswith("mtp.")
        }:
            checked_file(self.artifact["root"], name, self.artifact["seals"][name])
        return True

    def array(self, name):
        if self._closed:
            raise ValueError("DSpark weight owner is closed")
        return self._dense[name]

    def linear(self, name):
        import mlx.core as mx

        from ..runtime.models.deepseek_v41_quant import quantize

        def apply(x):
            if self._closed:
                raise ValueError("DSpark weight owner is closed")
            if name == "output.weight":
                return self.target.linear(name)(x)
            if name in self._quant:
                q, s = self._quant[name]
                return mx.quantized_matmul(
                    quantize(x, "fp8").astype(mx.float32), q, scales=s, mode="mxfp8"
                )
            return x.astype(mx.float32) @ self.array(name).astype(mx.float32).T

        return apply

    def embedding(self, tokens, name="token_embd.weight"):
        if self._closed:
            raise ValueError("DSpark weight owner is closed")
        if name == "token_embd.weight":
            return self.target.embedding(tokens)
        return self.array(name)[tokens]

    def grouped(self, x, name, *, groups=8):
        import mlx.core as mx

        w = self.array(name)
        return mx.einsum(
            "btgi,goi->btgo",
            x.astype(mx.float32),
            w.astype(mx.float32).reshape(groups, w.shape[0] // groups, w.shape[1]),
        )

    def gate_up(self, x, a, b, **kwargs):
        return self.linear(a)(x), self.linear(b)(x)

    def moe(self, x, ids, scores, prefix, **kwargs):
        import mlx.core as mx

        from ..runtime.models.deepseek_v4 import limited_swiglu
        from ..runtime.models.deepseek_v41_quant import quantize

        if self._closed:
            raise ValueError("DSpark weight owner is closed")
        stage = int(prefix.split(".")[1])
        indices = np.asarray(ids)
        result = mx.zeros(x.shape, mx.float32)
        for expert in sorted({int(v) for v in indices.flat}):
            rows, slots = np.where(indices == expert)
            positions = mx.array(rows, mx.uint32)
            value = x[positions]

            def project(v, part, expert=expert):
                q, s = self._experts[(stage, expert, part)]
                return (
                    mx.quantized_matmul(
                        quantize(v, "fp8").astype(mx.float32), q, scales=s, mode="mxfp4"
                    )
                    .astype(mx.bfloat16)
                    .astype(mx.float32)
                )

            gate, up = project(value, "w1"), project(value, "w3")
            middle = limited_swiglu(gate, up, 10)
            # Training weights the intermediate before the down projection.
            middle = (
                (middle * scores[positions, mx.array(slots, mx.uint32), None])
                .astype(mx.bfloat16)
                .astype(mx.float32)
            )
            result = result.at[positions].add(project(middle, "w2"))
        return result

    def dense_window(self):
        return nullcontext()

    def evaluate_dense_window(self, *values):
        import mlx.core as mx

        mx.eval(values)

    def close(self):
        if self._closed:
            return
        import mlx.core as mx

        from .deepseek_v41_mlx_weights import _quarantine_upload, _release_upload
        try:
            mx.synchronize()
        except BaseException:
            # Includes pending host/device buffers and the borrowed target.
            # A failed constructor can become unreachable outside its adapter.
            _quarantine_upload(self)
            raise
        self._dense.clear()
        self._quant.clear()
        self._experts.clear()
        self._complete_upload()
        self._closed = True
        self._loaded = False
        _release_upload(self)
