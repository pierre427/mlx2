# SPDX-License-Identifier: Apache-2.0
"""Bounded direct-row expert prefill using the existing V4.1 NAX equations.

Only activation outputs are concatenated. Expert weights retain their cached
native buffers. Source/provenance: deepseek-v41-adapter.json.
"""

from functools import lru_cache

import numpy as np

from . import v41_prefill as base
from .reader import GGUFError

GROUP_EXPERTS = 12


def routing_plan(ids, experts, group_experts=GROUP_EXPERTS):
    """Stable expert-major tiles, built from the streamer's existing readback."""
    values = np.asarray(ids)
    if (values.dtype != np.uint32 or values.ndim != 2 or not 1 <= values.shape[0] <= 1024
            or not 1 <= values.shape[1] <= 6 or type(experts) is not int
            or not 1 <= experts <= 384 or int(values.max()) >= experts or group_experts not in (6, 12)):
        raise GGUFError("row-group prefill requires bounded uint32 routes and <=384 experts")
    flat = values.reshape(-1)
    order = np.argsort(flat, kind="stable").astype(np.uint32)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(flat.size, dtype=np.uint32)
    offsets = np.concatenate(([0], np.bincount(flat, minlength=experts).cumsum()))
    groups = []
    for begin in range(0, experts, group_experts):
        end = min(experts, begin+group_experts)
        first, last = int(offsets[begin]), int(offsets[end])
        if first == last:
            continue
        tiles = [(row-first, expert-begin, min(16, int(offsets[expert+1])-row), 0)
                 for expert in range(begin, end)
                 for row in range(int(offsets[expert]), int(offsets[expert+1]), 16)]
        groups.append((begin, end, order[first:last], np.asarray(tiles, np.uint32)))
    return inverse, groups


def kernel_spec(kind, experts):
    if kind not in ("gate", "down") or type(experts) is not int or not 1 <= experts <= GROUP_EXPERTS:
        raise GGUFError("invalid row-group kernel binding")

    def choose(name):
        return "("+"".join(f"expert=={i}?{name}{i}:" for i in range(experts-1))+f"{name}{experts-1})"

    source = base.GATE_SOURCE if kind == "gate" else base.DOWN_SOURCE

    def once(source, before, after):
        # Every edit must land exactly once on the shared template, or the
        # kernel references unbound pointers (as after the MT/TILE rewrite).
        if source.count(before) != 1:
            raise GGUFError("V4.1 row-group prefill: shared template anchor changed")
        return source.replace(before, after, 1)

    # CPU-uploaded small descriptors may bind in Metal's constant address
    # space. Scalar access works for both constant and device descriptors.
    source = once(source, "const uint4 desc=*((device const uint4*)tiles+tile);",
                  "const uint4 desc=uint4(tiles[4*tile],tiles[4*tile+1],tiles[4*tile+2],tiles[4*tile+3]);")
    # Choose each expert pointer once per tile, outside both dequantization
    # loops. The original MMA, BF16 stores and reduction order stay intact.
    declarations = "device const uchar *row_w="+choose("w")+";\n"
    if kind == "gate":
        declarations += "device const uchar *row_up="+choose("up")+";\n"
    source = once(source, "float sums[MT][16];", declarations+"float sums[MT][16];")
    source = once(source, "w+(ulong(expert)*N+row)*ROW_BYTES", "row_w+ulong(row)*ROW_BYTES")
    if kind == "gate":
        source = once(source, "up+(ulong(expert)*N+row)*ROW_BYTES", "row_up+ulong(row)*ROW_BYTES")
    if kind == "gate":
        names = ["x", "order", "tiles", "tile_count", "scores"]
        names += [f"w{i}" for i in range(experts)]+[f"up{i}" for i in range(experts)]
    else:
        names = ["x", "tiles", "tile_count"]+[f"w{i}" for i in range(experts)]
    return names, source


def launch_template(kind, k, n, row_bytes, topk=None):
    """Template constants of one row-group launch: the shared V4.1 tile
    geometry (MT one 16-row tile per expert tile) plus the projection's own."""
    template = [("K", k), ("N", n), ("ROW_BYTES", row_bytes), ("FORMAT", 19), ("MT", base.MT)]
    return ([("TOPK", topk)] if kind == "gate" else [])+template


@lru_cache(maxsize=24)
def kernel(kind, experts):
    import mlx.core as mx

    names, source = kernel_spec(kind, experts)
    return mx.fast.metal_kernel(name=f"mlx2_v41_row_nax_{kind}_{experts}_v1",
                               input_names=names, output_names=["y"], source=source,
                               header=base.HEAD, ensure_row_contiguous=True)


def _moe(x, ids, scores, gate, up, down, *, counters=None, max_scratch_bytes=512 << 20, group_experts=GROUP_EXPERTS):
    import mlx.core as mx

    experts = len(gate)
    if not experts or len(up) != experts or len(down) != experts:
        raise GGUFError("row-group expert planes disagree")
    _, hidden, width = gate[0].shape
    if (any(w.ggml_type != 19 or w.shape != (1, hidden, width) for w in (*gate, *up))
            or any(w.ggml_type != 19 or w.shape != (1, width, hidden) for w in down)
            or x.ndim != 2 or x.shape[-1] != width or hidden % 256 or width % 256
            or scores.shape != ids.shape or ids.shape[0] != x.shape[0]):
        raise GGUFError("invalid row-group expert geometry")
    inverse, groups = routing_plan(ids, experts, group_experts)
    tokens, topk = ids.shape
    # Include both group outputs and their concatenation, unlike a weights
    # stack. Total scratch stays bounded even for1024*6 assignments.
    scratch = ids.size*(hidden+2*width)*4 + ids.size*24 + sum(t.nbytes for _, _, _, t in groups)
    if scratch > max_scratch_bytes:
        raise GGUFError("row-group activation scratch exceeds explicit cap")
    projected = []
    for begin, end, order, tiles in groups:
        n, count = order.size, len(tiles)
        tile_array, tile_count = mx.array(tiles), mx.array([count], mx.uint32)
        mid = kernel("gate", end-begin)(
            inputs=[x.astype(mx.float32), mx.array(order), tile_array, tile_count,
                    scores.reshape(-1).astype(mx.float32)]
                   +[w.data for w in gate[begin:end]]+[w.data for w in up[begin:end]],
            template=launch_template("gate", width, hidden, gate[0].row_bytes, topk),
            grid=(((hidden+127)//128)*128, count, 1), threadgroup=(128, 1, 1),
            output_shapes=[(n, hidden)], output_dtypes=[mx.float32])[0]
        value = kernel("down", end-begin)(
            inputs=[mid, tile_array, tile_count]+[w.data for w in down[begin:end]],
            template=launch_template("down", hidden, width, down[0].row_bytes),
            grid=(((width+127)//128)*128, count, 1), threadgroup=(128, 1, 1),
            output_shapes=[(n, width)], output_dtypes=[mx.float32])[0]
        projected.append(value)
    rows = projected[0] if len(projected) == 1 else mx.concatenate(projected)
    if counters is not None:
        counters["calls"] = counters.get("calls", 0)+1
        counters["groups"] = counters.get("groups", 0)+len(groups)
        counters["activation_concat_bytes"] = counters.get("activation_concat_bytes", 0)+(rows.nbytes if len(groups)>1 else 0)
    return base._kernel("combine")(
        inputs=[rows, mx.array(inverse), scores.reshape(-1).astype(mx.float32), mx.array([0], mx.uint32)],
        template=[("TOKENS", tokens), ("TOPK", topk), ("N", width)],
        grid=(((width+255)//256)*256, tokens, 1), threadgroup=(256, 1, 1),
        output_shapes=[(tokens, width)], output_dtypes=[mx.float32])[0]


_CHECKED = False


def self_check():
    global _CHECKED
    if _CHECKED:
        return
    import mlx.core as mx
    from dataclasses import replace
    from .packed_metal import PackedMatrix
    from .reader import GGUFTensor

    if mx.default_device() != mx.gpu or "Apple M5" not in mx.device_info().get("device_name", ""):
        raise GGUFError("row-group NAX prefill requires an owned Apple M5 GPU window")
    rng = np.random.default_rng(41024)
    for count in (2, 13):
        matrices, planes = [], []
        for kind in ("gate", "up", "down"):
            raw = rng.integers(0, 256, (count*256, 50), np.uint8)
            raw[:, :2] = np.frombuffer(np.float16(.0625).tobytes(), np.uint8)
            tensor = GGUFTensor(kind, (256, 256, count), 19, 0, raw.nbytes)
            matrix = PackedMatrix(mx.array(raw.reshape(-1)), tensor)
            matrices.append(matrix)
            row = replace(tensor, shape=(256, 256, 1), nbytes=raw.nbytes//count)
            planes.append(tuple(PackedMatrix(matrix.data[i*row.nbytes:(i+1)*row.nbytes], row) for i in range(count)))
        ids = (np.arange(33*min(6,count), dtype=np.uint32).reshape(33,-1)*7) % count
        x, scores = mx.full((33,256), .0625), mx.full(ids.shape, .5)
        expected = base.moe(x, mx.array(ids), scores, *matrices)
        actual = _moe(x, ids, scores, *planes)
        mx.eval(expected, actual)
        if not np.array(expected).tobytes() == np.array(actual).tobytes():
            raise GGUFError("row-group NAX first-use binding/ordering gate failed")
    _CHECKED = True


def moe(x, ids, scores, gate, up, down, **kwargs):
    self_check()
    return _moe(x, ids, scores, gate, up, down, **kwargs)
