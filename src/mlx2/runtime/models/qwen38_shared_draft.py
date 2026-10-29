"""Private shared-history native-MTP branches; provenance/mlx-tree-spec-20261027.json.

Committed KV arrays are borrowed read-only. Only the bounded branch suffix is
appended. This object cannot be published through APCv2 or used as target KV.
The split attention kernel merges online-softmax statistics without copying
or repeating the committed prefix. Arithmetic is a separate opt-in candidate.
"""

from __future__ import annotations

import os
from collections import Counter
from functools import lru_cache

import mlx.core as mx

ENV = "MLX2_MTP_SHARED_HISTORY"
STATS = Counter()


def enabled():
    value = os.environ.get(ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{ENV} must be 0 or 1")
    return value == "1"


def status():
    return {"implemented": True, "selected": enabled(), "qualified": False,
            "law": "shared_history_split_fp32_v1", **STATS}


_SOURCE = r"""
const uint lane = thread_index_in_simdgroup;
const int part = threadgroup_position_in_grid.x;
const int h = threadgroup_position_in_grid.y;
const int b = threadgroup_position_in_grid.z;
const int hk = h / (HQ / HK);
const int total = dims[0] + dims[1];
const int begin = part * CHUNK, end = min(begin + CHUNK, total);
float qv[DP], ov[DP];
for (int j = 0; j < DP; ++j) {
  qv[j] = float(Q[(b * HQ + h) * D + lane * DP + j]) * scale[0];
  ov[j] = 0.0f;
}
float maximum = -INFINITY, sum = 0.0f;
for (int row = begin; row < end; ++row) {
  const bool history = row < dims[0];
  const int64_t at = history ? (int64_t(hk) * dims[0] + row) * D
      : ((int64_t(b) * HK + hk) * dims[1] + row - dims[0]) * D;
  float score = 0.0f;
  for (int j = 0; j < DP; ++j)
    score += qv[j] * float(history ? K[at + lane * DP + j] : KT[at + lane * DP + j]);
  score = simd_sum(score);
  const float next = max(maximum, score);
  const float old_weight = metal::precise::exp(maximum - next);
  const float weight = metal::precise::exp(score - next);
  sum = sum * old_weight + weight;
  for (int j = 0; j < DP; ++j)
    ov[j] = ov[j] * old_weight + weight * float(history ? V[at + lane * DP + j] : VT[at + lane * DP + j]);
  maximum = next;
}
const int out = (b * HQ + h) * PARTS + part;
if (lane == 0) { PM[out] = maximum; PL[out] = sum; }
for (int j = 0; j < DP; ++j) PO[out * D + lane * DP + j] = ov[j];
"""


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(name="mlx2_shared_draft_attention_v1",
        input_names=["Q", "K", "V", "KT", "VT", "dims", "scale"],
        output_names=["PO", "PM", "PL"], source=_SOURCE)


class SharedDraftKV:
    """Ephemeral branch cache with one B1 history and at most 32 private rows."""

    def __init__(self, committed, branches):
        from .cache import KVCache
        if type(committed) is not KVCache or committed.keys is None or committed.offset < 1:
            raise ValueError("shared draft history requires nonempty ordinary KV")
        if not 1 <= branches <= 16 or committed.keys.shape[0] != 1:
            raise ValueError("shared draft history requires B1 and 1..16 branches")
        self.base = int(committed.offset)
        # Materialize the valid B1 view once, never once per branch or step.
        self.keys = mx.contiguous(committed.keys[..., :self.base, :])
        self.values = mx.contiguous(committed.values[..., :self.base, :])
        self.branches, self.offset = branches, self.base
        self.kt = self.vt = None
        STATS["histories_borrowed"] += 1
        STATS["avoided_branch_prefix_bytes"] += (branches - 1) * (self.keys.nbytes + self.values.nbytes)

    def update_and_fetch(self, *_):
        raise RuntimeError("shared draft KV requires its private attention operation")

    def append_and_attend(self, q, k, v, scale):
        b, hq, length, d = q.shape
        hk = k.shape[1]
        if (b != self.branches or length != 1 or d not in (64, 128, 256)
                or hq % hk or k.shape != v.shape or k.shape != (b, hk, 1, d)
                or self.keys.shape != (1, hk, self.base, d)
                or q.dtype != k.dtype or k.dtype != v.dtype or q.dtype != self.keys.dtype
                or q.dtype not in (mx.bfloat16, mx.float16, mx.float32)
                or self.offset - self.base >= 32):
            raise ValueError("shared draft attention geometry/dtype exceeds its contract")
        self.kt = k if self.kt is None else mx.concatenate((self.kt, k), axis=2)
        self.vt = v if self.vt is None else mx.concatenate((self.vt, v), axis=2)
        self.offset += 1
        tail = self.offset - self.base
        if mx.default_device() == mx.gpu and mx.metal.is_available():
            parts = (self.offset + 255) // 256
            po, pm, pl = _kernel()(inputs=[mx.contiguous(q), self.keys, self.values,
                mx.contiguous(self.kt), mx.contiguous(self.vt), mx.array([self.base, tail], mx.int32),
                mx.array([scale], mx.float32)],
                template=[("HQ", hq), ("HK", hk), ("D", d), ("DP", d // 32),
                          ("PARTS", parts), ("CHUNK", 256)],
                grid=(32 * parts, hq, b), threadgroup=(32, 1, 1),
                output_shapes=[(b, hq, parts, d), (b, hq, parts), (b, hq, parts)],
                output_dtypes=[mx.float32] * 3)
            weights = mx.exp(pm - mx.max(pm, axis=-1, keepdims=True))
            out = mx.sum(po * weights[..., None], axis=2) / mx.sum(pl * weights, axis=-1)[..., None]
            STATS["metal_calls"] += 1
            return out[:, :, None].astype(q.dtype)
        # Small CPU oracle for cache ownership and composition tests.
        qr = q.reshape(b, hk, hq // hk, d).astype(mx.float32)
        sh = (qr @ self.keys.astype(mx.float32).swapaxes(-1, -2)) * scale
        st = (qr @ self.kt.astype(mx.float32).swapaxes(-1, -2)) * scale
        p = mx.softmax(mx.concatenate((sh, st), axis=-1), axis=-1)
        out = p[..., :self.base] @ self.values.astype(mx.float32) + p[..., self.base:] @ self.vt.astype(mx.float32)
        STATS["cpu_calls"] += 1
        return out.reshape(b, hq, 1, d).astype(q.dtype)
