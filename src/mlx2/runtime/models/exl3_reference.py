"""Independent EXL3 host decoder, Sushi ad562c4 MIT format/math.

See provenance/sushi-integration-20261012.json and exl3_kernels/LICENSE.
"""

import numpy as np


def codebook(cb, window=16):
    cw = np.arange(65536, dtype=np.uint32) & np.uint32((1 << window) - 1)
    if cb == "mcg":
        r = ((cw * np.uint32(0xCBAC1FED)) & np.uint32(0x8FFF8FFF)) ^ np.uint32(
            0x3B603B60
        )
        lo = (r & 65535).astype(np.uint16).view(np.float16).astype(np.float32)
        hi = (r >> 16).astype(np.uint16).view(np.float16).astype(np.float32)
        return (lo + hi).astype(np.float16)
    if cb != "mul1":
        raise ValueError(f"EXL3 codebook {cb!r} is not implemented")
    mixed = cw * np.uint32(0x83DCD12D)
    total = sum((mixed >> shift) & 255 for shift in (0, 8, 16, 24))
    h = (0x6400 + total).astype(np.uint16).view(np.float16).astype(np.float64)
    inverse = np.array([0x1EEE], np.uint16).view(np.float16).astype(np.float64)[0]
    bias = np.array([0xC931], np.uint16).view(np.float16).astype(np.float64)[0]
    return (h * inverse + bias).astype(np.float16)


def decode_inner(trellis, cb="mcg", window=16):
    """Independent host unpack and tensor-core permutation, not shader reuse."""
    t = np.ascontiguousarray(trellis, dtype=np.uint16)
    n = t.shape[-1]
    words = t.view(np.uint32).reshape(*t.shape[:-1], n // 2)
    out = np.empty((*t.shape[:-1], 256), dtype=np.float16)
    lut = codebook(cb, window)
    total = 16 * n
    for thread in range(128):
        e0, e1 = ((2 * thread + 1) * n) >> 4, ((2 * thread + 2) * n) >> 4
        j0, j1 = (e0 + total - 16) // 32, (e1 + total - 1) // 32
        shift = (j1 + 1) * 32 - (e1 + total)
        merged = (words[..., j0 % (n // 2)].astype(np.uint64) << np.uint64(32)) | words[
            ..., j1 % (n // 2)
        ].astype(np.uint64)
        funnel = (merged >> np.uint64(shift)) & np.uint64(0xFFFFFFFF)
        for parity in (0, 1):
            slot = 2 * thread + parity
            cw = ((funnel >> np.uint64(e1 - e0)) if parity == 0 else funnel) & 65535
            lane, s = divmod(slot, 8)
            row = (lane & 3) * 2 + (s & 1) + (8 if s in (2, 3, 6, 7) else 0)
            col = (lane >> 2) + (8 if s >= 4 else 0)
            out[..., row * 16 + col] = lut[cw.astype(np.uint16)]
    grid = out.reshape(*t.shape[:-1], 16, 16)
    # Leading expert axis is optional in converter fixtures.
    leading = list(range(grid.ndim - 4))
    axes = leading + [grid.ndim - 4, grid.ndim - 2, grid.ndim - 3, grid.ndim - 1]
    return grid.transpose(axes).reshape(
        *t.shape[:-3], t.shape[-3] * 16, t.shape[-2] * 16
    )


def hadamard128(a):
    """Orthonormal H128 over the last axis in 128-wide blocks, float64."""
    a = np.array(a, dtype=np.float64)
    shape = a.shape
    a = a.reshape(-1, 128)
    for bit in (1, 2, 4, 8, 16, 32, 64):
        block = a.reshape(-1, 128 // (2 * bit), 2, bit)
        lo, hi = block[:, :, 0, :].copy(), block[:, :, 1, :].copy()
        block[:, :, 0, :], block[:, :, 1, :] = lo + hi, lo - hi
    return (a / np.sqrt(128)).reshape(shape)


def effective_weight(trellis, suh, svh, cb="mcg", window=16):
    """Dense [I, O] float64 W with ``x @ W == svh * H(H(suh * x) @ inner)``."""
    suh = np.asarray(suh, dtype=np.float64)
    svh = np.asarray(svh, dtype=np.float64)
    inner = decode_inner(np.asarray(trellis).view(np.uint16), cb, window)
    rows = hadamard128(np.diag(suh)) @ inner.astype(np.float64)
    return hadamard128(rows) * svh
