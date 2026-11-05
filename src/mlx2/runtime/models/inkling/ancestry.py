# SPDX-License-Identifier: Apache-2.0
"""Ancestry-indexed short convolutions, one launch for a complete tree.

Same ordered four-term convolution and BF16 boundaries as language.py.
See provenance/inkling-optimization-20261103.json.
"""

import mlx.core as mx

_SOURCE = r"""
    const uint c=thread_position_in_grid.x, row=thread_position_in_grid.y;
    const uint C=x_shape[2], L=x_shape[1];
    if(c>=C || row>=L) return;
    float acc=0;
    for(int j=0;j<4;++j) {
        const int r=ancestry[row*4+j];
        float value=r<0 ? state[(r+3)*C+c] : float(x[r*C+c]);
        // Committed histories contain rounded BF16 activations, promoted to FP32.
        acc+=float(w[c*4+j])*value;
        if(j>0) tails[(row*3+j-1)*C+c]=value;
    }
    const float rounded=float(T(acc));
    out[row*C+c]=T(rounded+float(x[row*C+c]));
"""
_KERNEL = mx.fast.metal_kernel(
    name="inkling_tree_shortconv",
    input_names=["x", "state", "w", "ancestry"],
    output_names=["out", "tails"],
    source=_SOURCE,
)


def convolution(module, raw, state, paths):
    channels, rows = raw.shape[-1], raw.shape[1]
    if state is None:
        state = mx.zeros((1, 3, channels), mx.float32)
    ancestry = mx.array([([-3, -2, -1] + list(path))[-4:] for path in paths], mx.int32)
    return _KERNEL(
        inputs=[
            mx.contiguous(raw),
            mx.contiguous(state),
            mx.contiguous(module.conv.weight.reshape(-1)),
            ancestry,
        ],
        template=[("T", raw.dtype)],
        grid=((channels + 31) // 32 * 32, rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[raw.shape, (rows, 3, channels)],
        output_dtypes=[raw.dtype, mx.float32],
    )
