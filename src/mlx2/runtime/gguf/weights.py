"""GGUF weight factory with explicit packed/native allocation boundaries."""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import pairwise

# Hot-path modules are imported once here, not inside the per-step methods
# (each in-function import costs an importlib lookup on every projection).
# None of them imports mlx at module scope.
from .blocks import decode_blocks
from .kblock_metal import (
    kblock_admits,
    kblock_dense,
    kblock_grouped,
    kblock_moe_shared,
    kblock_multi,
)
from .nax_metal import LEGACY_FORMATS, nax_admits
from .packed_metal import PackedMatrix, embedding, grouped_linear, packed_moe
from .prefill_scope import dense_prefill_active, prompt_rows_active
from .reader import GGUFError, GGUFFile
from .segmented_prefill import segmented_prefill_active

_NAX_DENSE_POLICIES = ("nax_fp16", "nax_fp16_v2", "nax_fp16_v2d")
_KBLOCK_ROW_POLICIES = ("kblock_rows", "kblock_exact", "kblock_rows_ahead2")
# decoded_f32_v1: fp32 element values on MLX's fp32 GEMM (class 2: same values,
# another sum order); decoded_f16_v1: half element values and half activations
# on the NAX fp16 GEMM, fp16 output widened to fp32 (class 3, the FP16-operand
# law of the nax_fp16 prompt tiles plus an fp16 output rounding).
_PROMPT_DECODED_POLICIES = ("decoded_f32_v1", "decoded_f16_v1")
_SEGMENTED_EXPERT_POLICIES = ("segmented_nax_fp16", "segmented_nax_fp16_v2")


def _nax_candidate(ggml_type, n, k):
    """NAX prefill eligibility at load-time geometry, never runtime fallback.

    The original Q8_0/Q2_K/IQ2_XXS candidates stay fail-closed in the kernel
    on bad geometry; K-block tensors whose rows do not fit the tile (GLM's
    128-wide Q4_0 KDA low-rank inputs) keep the packed reference kernel.
    """
    return ggml_type in LEGACY_FORMATS or nax_admits(ggml_type, n, k)


class GGUFWeights:
    """Lazy cached packed/native uploads; use only inside GPU ownership.

    Native F16/F32/BF16 tensors retain their dtype. Quantized tensors cannot be
    silently expanded: array() has a bounded reference limit, linear() uses a
    packed kernel. No arbitrary whole-model floating conversion occurs.

    Host dispatch memo: ``linear`` closures and ``multi_linear`` /
    ``moe_with_shared`` admission plans are pure functions of the tensor
    names and the policy attributes, so they are built once per
    (name(s), policy) key instead of every decode step. The key carries every
    policy attribute the decision reads, so reassigning a policy picks a fresh
    entry; ``close`` drops them. The arrays and kernels are the ones the
    unmemoized path builds. Subclasses that evict or replace ``_packed``
    entries (DeepSeekV41Weights pages them) keep the unmemoized path unless
    they set ``_dispatch_memo = True`` themselves.
    """

    _dispatch_memo = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "_dispatch_memo" not in cls.__dict__:
            cls._dispatch_memo = False

    def _memo(self):
        if not self._dispatch_memo:
            return None
        memo = self.__dict__.get("_dispatch_cache")
        if memo is None:
            memo = self.__dict__["_dispatch_cache"] = {}
        return memo

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
        row_policy="reference",
        prompt_decoded_policy="reference",
        native_read_view=None,
        source_sha256=None,
    ):
        if embedding_policy not in ("resident", "file_rows") or compute_policy not in (
            "reference",
            "cooperative",
        ):
            raise GGUFError("unsupported packed GGUF storage/compute policy")
        if expert_prefill_policy not in ("reference", "segmented_nax_fp16", "segmented_nax_fp16_v2"):
            raise GGUFError("unsupported packed expert prefill policy")
        if dense_prefill_policy not in ("reference", "nax_fp16", "nax_fp16_v2", "nax_fp16_v2d"):
            raise GGUFError("unsupported packed dense prefill policy")
        if row_policy not in ("reference",) + _KBLOCK_ROW_POLICIES:
            raise GGUFError("unsupported packed row policy")
        if prompt_decoded_policy not in ("reference",) + _PROMPT_DECODED_POLICIES:
            raise GGUFError("unsupported packed prompt decoded-weight policy")
        from pathlib import Path
        if native_read_view is not None:
            from .native_read_view import NativeReadViewFile
            self.file = NativeReadViewFile(path, native_read_view, source_sha256=source_sha256)
        elif Path(path).is_dir():
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
        self.row_policy = row_policy
        # decoded_f32_v1: prompt-width K-block projections that have no fast
        # tile (grouped weights; K < 256 rows) decode every element once per
        # chunk to float32 (kblock_decode_f32: the reference's gguf_value per
        # element) and multiply on MLX's dense GEMM. Same element values as
        # the packed reference, a different FP32 sum order; prompt rows only.
        self.prompt_decoded_policy = prompt_decoded_policy
        # Exact K-block rows keep packed_metal's per-row law (k order,
        # gguf_k_value, simd_sum) bit for bit; kblock_rows reorders sums
        # (_kblock_exact, read from row_policy per call).
        # kblock_rows_ahead2 is kblock_rows' chunk law with the next work
        # item's block bytes loaded before the current item's FMAs
        # (kblock_metal AHEAD=2); same sums bit for bit, only the load issue
        # order moves. OFF, and not proposed as a default: on
        # GLM-5.3-Flash-REAP50-Q3_K_M.gguf@5a865918 it is bitwise (every real
        # tensor, 139 Metal cases, tokens equal in replay) but slower in every
        # form tried. Cold routed MoE 455 -> 327 GB/s, Q3_K attn_q 399 -> 345
        # (wp2-kbench-t4-a1, ea3267eca); interleaved replay 25.08 -> 27.06 ms
        # per step at 2K and 33.98 -> 38.15 at 32K (wp2-ilv2-*, 67b78cd28).
        # Apple GPUs hide this load latency through occupancy; the extra live
        # raw words cost more occupancy than the overlap returns (a whole chunk
        # step ahead: 443 -> 105 GB/s, 4ede902f7).
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
        memo = self._memo()
        if memo is None:
            return self._build_linear(name)
        # getattr: providers built with __new__ (tests, probes) may omit policies.
        key = ("linear", name, getattr(self, "dense_prefill_policy", None),
               getattr(self, "compute_policy", None), getattr(self, "row_policy", None),
               getattr(self, "prompt_decoded_policy", None))
        project = memo.get(key)
        if project is None:
            project = memo[key] = self._build_linear(name)
        return project

    def _build_linear(self, name):
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
        if self.dense_prefill_policy in _NAX_DENSE_POLICIES and _nax_candidate(
            tensor.ggml_type, *weight.shape
        ):

            def project(x):
                flat = x.reshape(-1, x.shape[-1])
                if (dense_prefill_active(flat.shape[0]) and weight.ggml_type in (11, 12, 13, 14)
                        and self.dense_prefill_policy == "nax_fp16_v2d"):
                    # Decode-once v2: every weight value decoded once to a device
                    # fp16 scratch, then the v2 op reads it in place (64x128 tile).
                    # Bit-identical to kblock_nax_prefill_v2 (s1-scratch).
                    from .kblock_prefill_metal import kblock_nax_prefill_v2d

                    result = kblock_nax_prefill_v2d(flat, weight, tn=128)
                    route = "kblock_nax_prefill_v2d"
                elif (dense_prefill_active(flat.shape[0]) and weight.ggml_type in (11, 12, 13, 14)
                        and self.dense_prefill_policy == "nax_fp16_v2"):
                    # Guide-shaped 64x64 tile, multiply_accumulate: fp64-equidistant
                    # to nax_fp16 (ratio 1.0000, p9-v2), not bit-identical.
                    from .kblock_prefill_metal import kblock_nax_prefill_v2

                    result = kblock_nax_prefill_v2(flat, weight)
                    route = "kblock_nax_prefill_v2"
                elif dense_prefill_active(flat.shape[0]) and weight.ggml_type in (11, 12, 13, 14):
                    # Same FP16-operand law as nax_prefill, bit for bit
                    # (chunk-decoded weight tiles, two 16-row tiles per decode).
                    from .kblock_prefill_metal import kblock_nax_prefill

                    result = kblock_nax_prefill(flat, weight, mt=2)
                    route = "kblock_nax_prefill"
                elif dense_prefill_active(flat.shape[0]):
                    from .nax_metal import nax_prefill

                    result = nax_prefill(flat, weight)
                    route = "nax_prefill"
                else:
                    result, route = self._dense_rows(flat, weight)
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
        if self._prompt_decoded_linear(weight):
            # No NAX tile admits this weight (K % 256 != 0: the KDA ssm_f_b /
            # ssm_g_b Q4_0 rows at K = 128); prompt rows decode it once and
            # run MLX's GEMM instead of the one-token row kernel.
            def project(x):
                flat = x.reshape(-1, x.shape[-1])
                if prompt_rows_active(flat.shape[0]):
                    result, route = self._decoded_matmul(flat, weight)
                else:
                    result, route = self._dense_rows(flat, weight)
                self.scheduled_dense_calls[route] = (
                    self.scheduled_dense_calls.get(route, 0) + 1
                )
                return result.reshape(*x.shape[:-1], tensor.shape[1])

            return project
        if self._kblock_rows(weight):

            def project(x):
                result, route = self._dense_rows(x.reshape(-1, x.shape[-1]), weight)
                self.scheduled_dense_calls[route] = (
                    self.scheduled_dense_calls.get(route, 0) + 1
                )
                return result.reshape(*x.shape[:-1], tensor.shape[1])

            return project
        return weight

    @property
    def _kblock_exact(self):
        """1 under kblock_exact, read per call so a runtime row_policy switch
        (bench arms, replay variants) never keeps a stale law."""
        return int(getattr(self, "row_policy", None) == "kblock_exact")

    @property
    def _kblock_ahead(self):
        """K-block ``ahead=`` of every row call site, read per call from the
        policy (memo keys carry ``row_policy``, so a switch rebuilds them)."""
        return 2 if getattr(self, "row_policy", None) == "kblock_rows_ahead2" else 1

    # Research only (glm53_decode_kernel_bench ROWS sweep): output rows per
    # simdgroup of the one-token K-block row launches. Every row keeps its own
    # chunk sums, so any value gives the same bits; serving keeps 4. Not part
    # of a memo or compile-region key: set it before building closures.
    kblock_row_block = 4

    # K-block chunk decoder (the kernels' VEC): 1 = kblock_dot32_v2; 4 =
    # glm53_kblock_element=fma_v1 (Q3_K/Q6_K element values by one exact
    # fused multiply-add; same values and sums, see kblock_dot32_v5).
    # Applies to one-row chunk-law launches; exact-law and token-shared
    # launches keep their own decoders. The adapter sets it per instance.
    _kblock_vec = 1

    def _kblock_rows(self, weight):
        return self.row_policy in _KBLOCK_ROW_POLICIES and kblock_admits(
            weight.ggml_type, weight.shape[-1]
        )

    def _prompt_decoded(self, weight):
        """Whether prompt rows of this K-block weight take the decoded GEMM."""
        from .kblock_prefill_metal import DECODED_F32_FORMATS

        return (
            getattr(self, "prompt_decoded_policy", None) in _PROMPT_DECODED_POLICIES
            and weight.ggml_type in DECODED_F32_FORMATS
            and weight.shape[-1] % (64 if self.prompt_decoded_policy == "decoded_f16_v1" else 32) == 0
        )

    def _prompt_decoded_linear(self, weight):
        """Rank-2 admission: only projections no NAX dense tile admits. Under
        glm53_dense_prefill=reference the tile-admitted projections keep the
        packed law instead of silently moving to the decoded GEMM; under a
        NAX dense policy those took the tile first, so nothing changes."""
        return self._prompt_decoded(weight) and not _nax_candidate(weight.ggml_type, *weight.shape)

    def decoded(self, weight):
        """The weight's element values in the decoded policy's dtype, its own shape."""
        from .kblock_prefill_metal import kblock_decode_f32, kblock_decode_fp16

        if self.prompt_decoded_policy == "decoded_f16_v1":
            return kblock_decode_fp16(weight)
        return kblock_decode_f32(weight)

    def _decoded_matmul(self, flat, weight):
        """``flat @ W.T`` under the decoded policy; fp32 out either way."""
        import mlx.core as mx

        decoded = self.decoded(weight)
        if decoded.dtype == mx.float16:
            return (flat.astype(mx.float16) @ decoded.T).astype(mx.float32), "decoded_f16_dense"
        return flat @ decoded.T, "decoded_f32_dense"

    def _dense_rows(self, flat, weight):
        """Row (non-prefill-tile) execution of one rank2 packed projection."""
        if self.compute_policy == "cooperative" and weight.ggml_type in (8, 16):
            from .cooperative_metal import cooperative_dense

            return cooperative_dense(flat, weight, rows=4), "cooperative_dense"
        if self._kblock_rows(weight):
            tok = None
            if (getattr(self, "row_token_share", False) and flat.shape[0] > 1 and not self._kblock_exact
                    and weight.ggml_type in (11, 12, 13, 14)):
                # Multi-row decode/verify: decode each weight chunk once for up
                # to 4 tokens (2 at K>=8192); every row equals its 1-row call.
                tok = 2 if weight.shape[1] >= 8192 else 4
            if tok:
                return (kblock_dense(flat, weight, rows=4, tok=tok, ahead=self._kblock_ahead),
                        "kblock_rows_dense_tok")
            return (kblock_dense(flat, weight, rows=self.kblock_row_block, exact=self._kblock_exact, vec=self._kblock_vec,
                                 ahead=self._kblock_ahead),
                    "kblock_rows_dense")
        return weight(flat), "packed_reference_dense"

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
        gate, up = self.tensors[gate_name], self.tensors[up_name]
        if (
            self.compute_policy == "cooperative"
            and gate.ggml_type == up.ggml_type == 8
            and gate.shape == up.shape
            and len(gate.shape) == 2
            and not (
                self.dense_prefill_policy in ("nax_fp16", "nax_fp16_v2", "nax_fp16_v2d")
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

    def grouped_decoded(self, name, *, groups=8, dtype=None):
        """A grouped K-block weight's element values as (groups, out, in),
        decoded once per call in ``dtype`` (float16 or float32)."""
        import mlx.core as mx

        from .kblock_prefill_metal import kblock_decode_f32, kblock_decode_fp16

        tensor = self.tensors[name]
        shape = tensor.numpy_shape
        if len(shape) == 2:
            shape = (groups, shape[0] // groups, shape[1])
        if tensor.ggml_type in (0, 1, 30):
            value = self.array(name).reshape(shape)
            return value if dtype is None else value.astype(dtype)
        key = (name, groups)
        if key not in self._grouped:
            self._grouped[key] = PackedMatrix(self.packed(name).data, replace(tensor, shape=shape[::-1]))
        return (kblock_decode_fp16 if dtype == mx.float16 else kblock_decode_f32)(self._grouped[key])

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
        if self._prompt_decoded(self._grouped[key]) and prompt_rows_active(math.prod(x.shape[:-2])):
            # Decode the (groups, out, in) weight once for this chunk and run
            # one batched GEMM over heads; nax_grouped_prefill's first-
            # generation tile re-decodes every element per 16 rows (~4 TF/s).
            import mlx.core as mx

            decoded = self.decoded(self._grouped[key])
            half = decoded.dtype == mx.float16
            lead = x.shape[:-2]
            rows = mx.moveaxis(x, -2, 0).reshape(groups, -1, x.shape[-1])
            result = mx.matmul(rows.astype(mx.float16) if half else rows, mx.swapaxes(decoded, 1, 2))
            result = mx.moveaxis(result.reshape(groups, *lead, shape[1]), 0, -2)
            if half:
                result = result.astype(mx.float32)
            route = "decoded_f16_grouped" if half else "decoded_f32_grouped"
            self.scheduled_grouped_calls[route] = self.scheduled_grouped_calls.get(route, 0) + 1
            return result
        if (self.dense_prefill_policy in _NAX_DENSE_POLICIES
                and _nax_candidate(tensor.ggml_type, *shape[1:])
                and dense_prefill_active(math.prod(x.shape[:-2]))):
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
        if self._kblock_rows(self._grouped[key]):
            self.scheduled_grouped_calls["kblock_rows_grouped"] = (
                self.scheduled_grouped_calls.get("kblock_rows_grouped", 0) + 1
            )
            return kblock_grouped(x, self._grouped[key], rows=self.kblock_row_block, exact=self._kblock_exact, vec=self._kblock_vec,
                                  ahead=self._kblock_ahead)
        self.scheduled_grouped_calls["grouped_linear"] = (
            self.scheduled_grouped_calls.get("grouped_linear", 0) + 1
        )
        return grouped_linear(x, self._grouped[key])

    def moe(self, x, ids, scores, prefix):
        function = packed_moe
        if self.expert_prefill_policy in ("segmented_nax_fp16", "segmented_nax_fp16_v2"):
            from .segmented_prefill import (
                segmented_moe_prefill,
                segmented_prefill_active,
            )

            if segmented_prefill_active(x.shape[0]):
                function = segmented_moe_prefill
                kblock_experts = all(
                    self.packed(prefix + f".ffn_{p}_exps.weight").ggml_type in (11, 12, 13, 14)
                    for p in ("gate", "up", "down"))
                if kblock_experts and self.expert_prefill_policy == "segmented_nax_fp16_v2":
                    # Guide-shaped tiles, pre-gathered activations: fp64-equidistant
                    # to segmented_nax_fp16 (p14-moe-v3), not bit-identical.
                    from .kblock_prefill_metal import kblock_segmented_moe_prefill_v3

                    function = kblock_segmented_moe_prefill_v3
                elif self.packed(prefix + ".ffn_gate_exps.weight").ggml_type in (11, 12, 13, 14):
                    # K-block experts: 32-row tiles (MT=2) halve weight-tile
                    # decodes; per-output arithmetic is unchanged.
                    from functools import partial

                    function = partial(segmented_moe_prefill, mt=2, half_x=True)
                    function.__name__ = "segmented_moe_prefill"
        if function is packed_moe and self.compute_policy == "cooperative":
            from .cooperative_metal import cooperative_moe

            function = cooperative_moe
        if function is packed_moe and self._kblock_rows(
            self.packed(prefix + ".ffn_gate_exps.weight")
        ) and self._kblock_rows(self.packed(prefix + ".ffn_down_exps.weight")):
            from functools import partial

            from .kblock_metal import kblock_moe

            function = partial(kblock_moe, rows=self.kblock_row_block, exact=self._kblock_exact, vec=self._kblock_vec,
                               ahead=self._kblock_ahead)
            function.__name__ = "kblock_moe"
            if (getattr(self, "expert_union", False) and x.shape[0] > 1 and not self._kblock_exact
                    and all(self.packed(prefix + f".ffn_{p}_exps.weight").ggml_type in (11, 12, 13, 14)
                            for p in ("gate", "up", "down"))):
                # Multi-row decode/verify: each routed expert decoded once per
                # run of sorted slots; every slot equals kblock_moe's.
                from .kblock_metal import kblock_moe_union

                function = partial(kblock_moe_union, ahead=self._kblock_ahead)
                function.__name__ = "kblock_moe_union"
        name = function.__name__
        self.scheduled_expert_calls[name] = self.scheduled_expert_calls.get(name, 0) + 1
        window = getattr(self, "expert_prefill_window", None)
        if window and name.startswith(("kblock_segmented", "segmented")) and x.shape[0] > window:
            # Research: bound the segmented scratch by running row windows of
            # one prompt chunk; each row's output depends on that row only.
            import mlx.core as mx

            parts = [prefix + f".ffn_{p}_exps.weight" for p in ("gate", "up", "down")]
            return mx.concatenate([
                function(x[s:s + window], ids[s:s + window], scores[s:s + window],
                         *(self.packed(n) for n in parts))
                for s in range(0, x.shape[0], window)], axis=0)
        return function(
            x,
            ids,
            scores,
            self.packed(prefix + ".ffn_gate_exps.weight"),
            self.packed(prefix + ".ffn_up_exps.weight"),
            self.packed(prefix + ".ffn_down_exps.weight"),
        )

    def multi_linear(self, inputs, names, *, joined=1):
        """One K-block launch for several projections, or None if not admitted.

        Row-path only: a NAX prefill-tile width keeps the separate projections.
        Each output equals its kblock_dense call exactly. The first ``joined``
        outputs come back as one slice, already concatenated in name order.
        """
        flats = [x.reshape(-1, x.shape[-1]) for x in inputs]
        plan = self._multi_plan(names, joined)
        if plan is None:
            return None
        packed, spans = plan
        if dense_prefill_active(flats[0].shape[0]):
            return None
        if prompt_rows_active(flats[0].shape[0]) and any(self._prompt_decoded_linear(w) for w in packed):
            # A joined row launch would consume the KDA low-rank rows before
            # the decoded-f32 prompt law could; the separate projections take it.
            return None
        try:
            y = kblock_multi(flats, packed, rows=self.kblock_row_block, exact=self._kblock_exact, vec=self._kblock_vec,
                             ahead=self._kblock_ahead)
        except GGUFError:
            return None
        self.scheduled_dense_calls["kblock_rows_multi"] = (
            self.scheduled_dense_calls.get("kblock_rows_multi", 0) + 1
        )
        lead = inputs[0].shape[:-1]
        return [y[:, a:b].reshape(*lead, b - a) for a, b in spans]

    def _multi_plan(self, names, joined):
        """(packed weights, output column spans) or None when not admitted."""
        memo = self._memo()
        key = ("multi", tuple(names), joined, getattr(self, "row_policy", None))
        if memo is not None and key in memo:
            return memo[key]
        packed = [self.packed(n) for n in names]
        plan = None
        if all(self._kblock_rows(w) for w in packed):
            bounds = [0]
            for w in packed:
                bounds.append(bounds[-1] + w.shape[0])
            plan = packed, tuple(pairwise([0] + bounds[joined:]))
        if memo is not None:
            memo[key] = plan
        return plan

    def moe_with_shared(self, x, ids, scores, prefix, shared_prefix):
        """Routed experts plus their shared expert in one set of K-block launches.

        Returns None when the row policy or the formats do not admit the fold,
        so the caller keeps its separate routed + shared computation.
        """
        if (self.expert_prefill_policy in _SEGMENTED_EXPERT_POLICIES
                and segmented_prefill_active(x.shape[0])):
            return None
        memo = self._memo()
        key = ("moe_shared", prefix, shared_prefix, getattr(self, "row_policy", None))
        if memo is not None and key in memo:
            packed = memo[key]
        else:
            names = [prefix + f".ffn_{p}_exps.weight" for p in ("gate", "up", "down")]
            shared = [shared_prefix.format(p) for p in ("gate", "up", "down")]
            packed = [self.packed(n) for n in names + shared]
            if not all(self._kblock_rows(w) for w in packed):
                packed = None
            if memo is not None:
                memo[key] = packed
        if packed is None:
            return None
        try:
            result = kblock_moe_shared(x, ids, scores, *packed, exact=self._kblock_exact, vec=self._kblock_vec,
                                       rows=self.kblock_row_block, ahead=self._kblock_ahead)
        except GGUFError:
            return None
        self.scheduled_expert_calls["kblock_moe_shared"] = (
            self.scheduled_expert_calls.get("kblock_moe_shared", 0) + 1
        )
        return result

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
            self.__dict__.pop("_dispatch_cache", None)
            self.file.close()

    def verify_storage(self):
        verify = getattr(self.file, "verify", None)
        if callable(verify):
            verify()
