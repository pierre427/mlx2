"""Tiny synthetic dense Qwen3 backend gate, dry-run by default.

--execute-gpu is reserved for a separately coordinated GPU lease, both lock
guards, and an extension rebuilt from the exact source revision. This is not
model-serving qualification or a performance measurement.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace


def dry_run() -> dict:
    return {
        "schema": "mlx2.varlen-qwen3-backend-probe.v1",
        "mode": "dry-run", "gpu_executed": False,
        "planned_cell": "two synthetic dense Qwen3 layers; ragged fp16 lanes of 3 and 1 rows",
        "pool_pages": 8, "kv_heads": 2, "head_dim": 128,
        "arena_bytes_two_planes": 1_048_576,
        "peak_plan_scratch_upper_bound_bytes": 8_320,
        "requires": ["fresh GPU FIFO ownership", "both GPU lock guards",
                     "no foreign service owner", "matching native extension build"],
        "not_proven": ["GPU execution", "native callback", "model serving",
                       "qualification", "performance"],
    }


def gpu_cell() -> dict:
    import mlx.core as mx

    from mlx2.adapters.qwen3_paged_candidate import PackedLane, Qwen3PackedCandidate
    from mlx2.runtime.paged_kv_pool import PagedKVPool
    from mlx2.runtime.paged_kv_token import PagedKVTokenOwner, TokenKVProfile
    from mlx2.runtime.paged_kv_write import NativeWriteBackend, PagedKVWriteOwner
    from mlx2.runtime.qwen3_paged_native_backend import NativeQwen3PagedBackend

    if not mx.metal.is_available():
        raise RuntimeError("synthetic Qwen3 backend probe requires MLX GPU")
    mx.set_default_device(mx.gpu)

    class SyntheticQwen3:
        args = SimpleNamespace(model_type="qwen3", num_experts=0, rope_scaling=None,
                               head_dim=128, num_attention_heads=4,
                               num_key_value_heads=2)
        layers = (object(), object())

        def paged_embed(self, tokens):
            return mx.array(tokens, dtype=mx.float16) / 16

        def paged_project(self, layer, hidden, counts, offsets):
            rows = hidden.shape[0]
            base = hidden[:, None, None]
            q = mx.broadcast_to(base / 8, (rows, 4, 128))
            k = mx.broadcast_to((base + layer) / 4, (rows, 2, 128))
            v = mx.broadcast_to((base + layer + 1) / 2, (rows, 2, 128))
            return q, k, v

        def paged_finish_layer(self, layer, hidden, attended):
            return hidden + mx.mean(attended, axis=(1, 2))

        def paged_logits(self, hidden):
            return hidden

    profile = TokenKVProfile(2, 128, "float16")
    pool = PagedKVPool(8)
    native = NativeWriteBackend(pool.capacity * profile.page_bytes,
                                mx.default_stream(mx.gpu), permit_candidate=True)
    writer = PagedKVWriteOwner(pool, native, page_bytes=profile.page_bytes,
                               permit_candidate=True)
    backend = NativeQwen3PagedBackend(writer, permit_candidate=True)
    lanes = tuple(PackedLane(tokens, tuple(
        PagedKVTokenOwner(writer, profile, permit_candidate=True) for _ in range(2)))
        for tokens in ((2, 3, 5), (7,)))
    output, receipt = Qwen3PackedCandidate(SyntheticQwen3(), backend).forward(
        lanes, permit_candidate=True)
    mx.eval(output)
    mx.synchronize(native.stream)
    offsets = [[owner.offset for owner in lane.layers] for lane in lanes]
    if offsets != [[3, 3], [1, 1]] or writer.ledger.pending_count:
        raise AssertionError("Qwen3 backend did not publish and close every native use")
    for lane in lanes:
        for owner in lane.layers:
            owner.close()
    if pool.free_count != pool.capacity:
        raise AssertionError("Qwen3 backend retained pages after closure")
    return {
        "schema": "mlx2.varlen-qwen3-backend-probe.v1",
        "mode": "synthetic-gpu", "gpu_executed": True,
        "layers": 2, "rows": [3, 1], "offsets": offsets,
        "output_shape": list(output.shape),
        "all_native_uses_terminal": True,
        "qualification": False, "serving_route_selected": False,
        "candidate_receipt": receipt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-gpu", action="store_true")
    args = parser.parse_args()
    print(json.dumps(gpu_cell() if args.execute_gpu else dry_run(), indent=2))


if __name__ == "__main__":
    main()
