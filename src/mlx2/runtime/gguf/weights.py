"""GGUF weight factory with explicit packed/native allocation boundaries."""

from __future__ import annotations

import math
from dataclasses import replace

from .blocks import decode_blocks
from .packed_metal import PackedMatrix, embedding, grouped_linear, packed_moe
from .reader import GGUFError, GGUFFile


class GGUFWeights:
    """Lazy cached packed/native uploads; use only inside GPU ownership.

    Native F16/F32/BF16 tensors retain their dtype. Quantized tensors cannot be
    silently expanded: array() has a bounded reference limit, linear() uses a
    packed kernel. No arbitrary whole-model floating conversion occurs.
    """

    def __init__(
        self,
        path,
        *,
        max_dense_elements=2_000_000,
        embedding_policy="resident",
        expected_stat=None,
        compute_policy="reference",
        expert_prefill_policy="reference",
        dense_prefill_policy="reference",
    ):
        if embedding_policy not in ("resident", "file_rows") or compute_policy not in (
            "reference",
            "cooperative",
        ):
            raise GGUFError("unsupported packed GGUF storage/compute policy")
        if expert_prefill_policy not in ("reference", "segmented_nax_fp16"):
            raise GGUFError("unsupported packed expert prefill policy")
        if dense_prefill_policy not in ("reference", "nax_fp16"):
            raise GGUFError("unsupported packed dense prefill policy")
        from pathlib import Path
        if Path(path).is_dir():
            from .mlx_storage import MLXPackedFile

            self.file = MLXPackedFile(path)
        else:
            self.file = GGUFFile(path)
        self.metadata = self.file.metadata
        self.tensors = self.file.tensors
        self.max_dense_elements = max_dense_elements
        self._packed = {}
        self._dense = {}
        self._grouped = {}
        self._native_linear_fp32 = {}
        self.embedding_policy = embedding_policy
        self.compute_policy = compute_policy
        self.expert_prefill_policy = expert_prefill_policy
        self.dense_prefill_policy = dense_prefill_policy
        self._embedding_rows = {}
        self._expected_stat = expected_stat
        self.scheduled_expert_calls = {}
        self.scheduled_grouped_calls = {}
        self.scheduled_dense_calls = {}

    def packed(self, name):
        if name not in self._packed:
            direct = getattr(self.file, "device_array", None)
            self._packed[name] = (
                PackedMatrix(direct(name), self.tensors[name])
                if callable(direct) else PackedMatrix.upload(self.file, name)
            )
        return self._packed[name]

    def _native(self, name):
        import mlx.core as mx
        import numpy as np

        tensor = self.tensors[name]
        if tensor.ggml_type not in (0, 1, 24, 25, 26, 27, 30):
            raise GGUFError(
                "native upload only permits supported unquantized scalar types"
            )
        direct = getattr(self.file, "device_array", None)
        if callable(direct):
            return direct(name)
        view = self.file.packed_view(name)
        try:
            dtype = {
                0: "<f4",
                1: "<f2",
                24: "i1",
                25: "<i2",
                26: "<i4",
                27: "<i8",
                30: "<u2",
            }[tensor.ggml_type]
            data = mx.array(
                np.frombuffer(view, dtype=dtype).reshape(tensor.numpy_shape)
            )
            if tensor.ggml_type == 30:
                data = data.view(mx.bfloat16)
            mx.eval(data)
        finally:
            view.release()
            self.file.release_mapping()
        return data

    def array(self, name):
        import mlx.core as mx

        if name not in self._dense:
            tensor = self.tensors[name]
            if tensor.ggml_type in (0, 1, 24, 25, 26, 27, 30):
                self._dense[name] = self._native(name)
            else:
                view = self.file.packed_view(name)
                try:
                    data = decode_blocks(
                        view, tensor.ggml_type, max_elements=self.max_dense_elements
                    )
                finally:
                    view.release()
                    self.file.release_mapping()
                self._dense[name] = mx.array(data.reshape(tensor.numpy_shape))
        return self._dense[name]

    def linear(self, name):
        tensor = self.tensors[name]
        if len(tensor.shape) != 2:
            raise GGUFError("linear requires rank2 weights")
        if tensor.ggml_type in (0, 1, 30):
            import mlx.core as mx

            weight = self.array(name)
            cache = getattr(self, "_native_linear_fp32", None)
            if cache is None:
                cache = self._native_linear_fp32 = {}
            if name not in cache:
                # Match the original Matmul promotion graph exactly: transpose
                # native storage first, then promote that operand. Casting the
                # base then transposing could select a different GPU layout.
                prepared = getattr(getattr(self, "file", None), "linear_fp32", None)
                operand = prepared(name) if callable(prepared) else None
                cache[name] = weight.T.astype(mx.float32) if operand is None else operand
            fp32_operand = cache[name]
            return lambda x: x @ (fp32_operand if x.dtype == mx.float32 else weight.T)
        weight = self.packed(name)
        if self.dense_prefill_policy == "nax_fp16" and tensor.ggml_type in (8, 10, 16):

            def project(x):
                from .prefill_scope import dense_prefill_active

                flat = x.reshape(-1, x.shape[-1])
                if dense_prefill_active(flat.shape[0]):
                    from .nax_metal import nax_prefill

                    result = nax_prefill(flat, weight)
                    route = "nax_prefill"
                elif self.compute_policy == "cooperative" and tensor.ggml_type in (
                    8,
                    16,
                ):
                    from .cooperative_metal import cooperative_dense

                    result = cooperative_dense(flat, weight, rows=4)
                    route = "cooperative_dense"
                else:
                    result = weight(flat)
                    route = "packed_reference_dense"
                self.scheduled_dense_calls[route] = (
                    self.scheduled_dense_calls.get(route, 0) + 1
                )
                return result.reshape(*x.shape[:-1], tensor.shape[1])

            return project
        if self.compute_policy == "cooperative" and tensor.ggml_type in (8, 16):
            from .cooperative_metal import cooperative_dense

            def project(x):
                result = cooperative_dense(x.reshape(-1, x.shape[-1]), weight, rows=4)
                return result.reshape(*x.shape[:-1], tensor.shape[1])

            return project
        return weight

    def prepare_embedding(self, name="token_embd.weight"):
        """Upload a resident table or bind exact file rows, never both."""
        tensor = self.tensors[name]
        if self.embedding_policy == "file_rows":
            if name not in self._embedding_rows:
                from .embedding_rows import F16EmbeddingRows

                location = getattr(self.file, "embedding_location", None)
                if callable(location):
                    path, tensor, offset, expected = location(name)
                else:
                    path, offset, expected = self.file.path, self.file.data_offset, self._expected_stat
                self._embedding_rows[name] = F16EmbeddingRows(
                    path,
                    tensor,
                    data_offset=offset,
                    expected_stat=expected,
                )
        elif tensor.ggml_type in (0, 1, 30):
            self.array(name)
        else:
            self.packed(name)

    def gate_up(self, x, gate_name, up_name):
        """Pair compatible Q8 projections within the selected compute profile."""
        from .prefill_scope import dense_prefill_active

        gate, up = self.tensors[gate_name], self.tensors[up_name]
        if (
            self.compute_policy == "cooperative"
            and gate.ggml_type == up.ggml_type == 8
            and gate.shape == up.shape
            and len(gate.shape) == 2
            and not (
                self.dense_prefill_policy == "nax_fp16"
                and dense_prefill_active(math.prod(x.shape[:-1]))
            )
        ):
            from .cooperative_metal import cooperative_gate_up

            g, u = cooperative_gate_up(
                x.reshape(-1, x.shape[-1]), self.packed(gate_name), self.packed(up_name)
            )
            route = "cooperative_paired_gate_up"
            self.scheduled_dense_calls[route] = (
                self.scheduled_dense_calls.get(route, 0) + 1
            )
            shape = (*x.shape[:-1], gate.shape[1])
            return g.reshape(shape), u.reshape(shape)
        return self.linear(gate_name)(x), self.linear(up_name)(x)

    def embedding(self, ids, name="token_embd.weight"):
        if self.embedding_policy == "file_rows":
            import mlx.core as mx
            import numpy as np

            self.prepare_embedding(name)
            return mx.array(self._embedding_rows[name].rows(np.array(ids)))
        tensor = self.tensors[name]
        if tensor.ggml_type in (0, 1, 30):
            return self.array(name)[ids]
        return embedding(ids, self.packed(name))

    def grouped(self, x, name, *, groups=8):
        tensor = self.tensors[name]
        shape = tensor.numpy_shape
        if len(shape) == 2:
            n, k = shape
            if n % groups:
                raise GGUFError("grouped projection rows not divisible by groups")
            shape = (groups, n // groups, k)
        if len(shape) != 3 or shape[0] != groups:
            raise GGUFError("grouped projection group count mismatch")
        if tensor.ggml_type in (0, 1, 30):
            import mlx.core as mx

            return mx.einsum("...gi,goi->...go", x, self.array(name).reshape(shape))
        key = (name, groups)
        if key not in self._grouped:
            weight = self.packed(name)
            grouped_tensor = replace(tensor, shape=shape[::-1])
            self._grouped[key] = PackedMatrix(weight.data, grouped_tensor)
        if self.dense_prefill_policy == "nax_fp16" and tensor.ggml_type in (8, 10, 16):
            from .prefill_scope import dense_prefill_active

            if dense_prefill_active(math.prod(x.shape[:-2])):
                from .nax_metal import nax_grouped_prefill

                result = nax_grouped_prefill(x, self._grouped[key])
                self.scheduled_grouped_calls["nax_grouped_prefill"] = (
                    self.scheduled_grouped_calls.get("nax_grouped_prefill", 0) + 1
                )
                return result
        if self.compute_policy == "cooperative" and tensor.ggml_type in (8, 16):
            from .cooperative_metal import cooperative_grouped

            self.scheduled_grouped_calls["cooperative_grouped"] = (
                self.scheduled_grouped_calls.get("cooperative_grouped", 0) + 1
            )
            return cooperative_grouped(x, self._grouped[key])
        self.scheduled_grouped_calls["grouped_linear"] = (
            self.scheduled_grouped_calls.get("grouped_linear", 0) + 1
        )
        return grouped_linear(x, self._grouped[key])

    def moe(self, x, ids, scores, prefix):
        function = packed_moe
        if self.expert_prefill_policy == "segmented_nax_fp16":
            from .segmented_prefill import (
                segmented_moe_prefill,
                segmented_prefill_active,
            )

            if segmented_prefill_active(x.shape[0]):
                function = segmented_moe_prefill
        if function is packed_moe and self.compute_policy == "cooperative":
            from .cooperative_metal import cooperative_moe

            function = cooperative_moe
        name = function.__name__
        self.scheduled_expert_calls[name] = self.scheduled_expert_calls.get(name, 0) + 1
        return function(
            x,
            ids,
            scores,
            self.packed(prefix + ".ffn_gate_exps.weight"),
            self.packed(prefix + ".ffn_up_exps.weight"),
            self.packed(prefix + ".ffn_down_exps.weight"),
        )

    def native_fp32_parameters(self):
        """Only registered native rank2 linears; raw weights stay authoritative."""
        return dict(getattr(self, "_native_linear_fp32", {}))

    def preload_native_linear_fp32(self):
        """Materialize registered casts before loaded-memory/admission sampling."""
        import mlx.core as mx

        mx.eval(self.native_fp32_parameters())

    def native_linear_fp32_status(self):
        cached = self.native_fp32_parameters()
        additional = sum(
            value.nbytes
            for name, value in cached.items()
            if self.tensors[name].ggml_type != 0
        )
        return {
            "registered_linears": len(cached),
            "additional_bytes": additional,
            "raw_weights_retained": True,
            "quantized_expansion": False,
            "embedding_conversion": False,
        }

    def close(self):
        try:
            for reader in self._embedding_rows.values():
                reader.close()
        finally:
            self._embedding_rows.clear()
            self._packed.clear()
            self._dense.clear()
            self._grouped.clear()
            getattr(self, "_native_linear_fp32", {}).clear()
            self.file.close()

    def verify_storage(self):
        verify = getattr(self.file, "verify", None)
        if callable(verify):
            verify()
