# SPDX-License-Identifier: Apache-2.0
"""Pack independent ancestor queries into one dispatch without copying their KV.

Each row retains singleton split/reduction arithmetic. Only address selection
changes: the map points into the committed prefix plus this tree's new rows.
Runtime lengths avoid per-position compilation. Internal source: attention.py
at c744e90d5; see provenance/inkling-optimization-20261103.json.
"""

import mlx.core as mx

from .attention import _MERGE_KERNEL, _SOURCE, supported


def _replace(source, before, after):
    if source.count(before) != 1:
        raise RuntimeError("Inkling mapped attention source anchor changed")
    return source.replace(before, after)


SOURCE = _replace(
    _SOURCE,
    "const int N = k_shape[2], L = q_shape[2], HQ = q_shape[1];",
    "const int N = lengths[qt], L = q_shape[2], HQ = q_shape[1];",
)
SOURCE = _replace(
    SOURCE,
    "const int chunk = (N + blocks - 1) / blocks;",
    "const int active_blocks=min(blocks,max(1,(N+127)/128));\n"
    "    const int chunk = (N + active_blocks - 1) / active_blocks;",
)
SOURCE = _replace(
    SOURCE,
    "+ t*k_strides[2] + d*k_strides[3];",
    "+ size_t(mapping[qt*mapping_strides[0]+t])*k_strides[2] + d*k_strides[3];",
)
SOURCE = _replace(
    SOURCE,
    "const int row=qt*QT+j/G, pos=int(meta[0])+row, dist=pos-t;",
    "const int row=qt*QT+j/G, pos=N-1, dist=pos-t;",
)
SOURCE = _replace(
    SOURCE,
    "+(tile+u)*v_strides[2]+lane*EPL*v_strides[3];",
    "+size_t(mapping[qt*mapping_strides[0]+tile+u])*v_strides[2]+lane*EPL*v_strides[3];",
)
KERNEL = mx.fast.metal_kernel(
    name="inkling_mapped_relative_attention",
    input_names=[
        "q",
        "k",
        "v",
        "rel",
        "proj",
        "tau",
        "pad",
        "meta",
        "mapping",
        "lengths",
    ],
    output_names=["po", "pl", "pm"],
    source=SOURCE,
    ensure_row_contiguous=False,
)


def attend(q, k, v, rel, proj, *, scale, paths=None, prefix=None, window=0, tau=None):
    if not supported(q, k, v, rel, proj) or q.shape[0] != 1:
        raise ValueError("mapped Inkling attention requires one independent request")
    _, heads, rows, _ = q.shape
    if paths is None:
        if rows != 1:
            raise ValueError("ordinary mapped attention requires singleton decode")
        mapping = mx.arange(k.shape[2], dtype=mx.int32)[None]
        lengths = [k.shape[2]]
    else:
        width = len(paths)
        tail = (
            mx.array(
                [list(path) + [0] * (width - len(path)) for path in paths], mx.int32
            )
            + prefix
        )
        mapping = mx.concatenate(
            [
                mx.broadcast_to(
                    mx.arange(prefix, dtype=mx.int32)[None], (rows, prefix)
                ),
                tail,
            ],
            axis=1,
        )
        lengths = [prefix + len(path) for path in paths]
        if window:
            # Gather only the map, not the K/V vectors, into chronological windows.
            starts = mx.array([max(0, n - window) for n in lengths], mx.int32)
            extent = min(window, max(lengths))
            columns = mx.minimum(
                starts[:, None] + mx.arange(extent)[None], mapping.shape[1] - 1
            )
            mapping = mx.take_along_axis(mapping, columns, axis=1)
            lengths = [min(window, n) for n in lengths]
    blocks = min(16, max(1, (max(lengths) + 127) // 128))
    parts = blocks * 4
    tau = mx.ones((1, rows), q.dtype) if tau is None else tau
    po, pl, pm = KERNEL(
        inputs=[
            q,
            k,
            v,
            rel,
            proj,
            tau,
            mx.zeros((1,), mx.int32),
            mx.array([0, scale, blocks], mx.float32),
            mapping,
            mx.array(lengths, mx.int32),
        ],
        template=[
            ("T", q.dtype),
            ("G", heads // k.shape[1]),
            ("QT", 1),
            ("R", rel.shape[3]),
            ("EXT", proj.shape[1]),
            ("WIN", window),
        ],
        grid=(128, k.shape[1], blocks * rows),
        threadgroup=(128, 1, 1),
        output_shapes=[
            (1, heads, rows, parts, 128),
            (1, heads, rows, parts),
            (1, heads, rows, parts),
        ],
        output_dtypes=[mx.float32] * 3,
    )
    return _MERGE_KERNEL(
        inputs=[po, pl, pm],
        template=[("T", q.dtype)],
        grid=(32 * heads * rows, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]
