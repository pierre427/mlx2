# SPDX-License-Identifier: MIT
"""Pure CPU actual MXFP4 operand inspection; no MLX import or GPU work.

Provenance: provenance/deepseek-v41-native-mx-operands.json. The FP64 oracle
checks packing/geometry, not the stock F32 reduction or model quality.
"""
import hashlib


def digest(value):
    return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def half_roundtrip(np, value):
    if value.dtype != np.float32 or value.ndim != 2 or not value.size:
        raise ValueError('Expected nonempty F32 matrix')
    with np.errstate(over='ignore', invalid='ignore'):
        restored = value.astype(np.float16).astype(np.float32)
    mask = value.view(np.uint32) != restored.view(np.uint32)
    where = np.argwhere(mask)
    result = dict(exact=not len(where), changed_words=int(mask.sum()),
                  input_finite=bool(np.isfinite(value).all()),
                  half_finite=bool(np.isfinite(restored).all()),
                  minimum=float(value.min()), maximum=float(value.max()),
                  maximum_absolute=float(np.abs(value).max()),
                  input_sha256=digest(value), restored_sha256=digest(restored))
    if len(where):
        at = tuple(int(i) for i in where[0])
        result['first_difference'] = dict(coordinate=list(at), input=float(value[at]),
            restored=float(restored[at]), input_bits=f'{int(value.view(np.uint32)[at]):08x}',
            restored_bits=f'{int(restored.view(np.uint32)[at]):08x}')
    return result


def decode_scales(np, scales):
    """Installed fp8_e8m0::operator float law, including its 00/ff cases."""
    if scales.dtype != np.uint8:
        raise ValueError('Expected original E8M0 bytes')
    bits = scales.astype(np.uint32) << np.uint32(23)
    bits = np.where(scales == 0, np.uint32(0x00400000), bits).astype(np.uint32)
    return bits.view(np.float32)


def decode_rows(np, q, scales, output_rows):
    if (q.dtype != np.uint32 or q.ndim != 2 or scales.dtype != np.uint8
            or scales.shape != (q.shape[0], q.shape[1] // 4)
            or q.shape[1] % 4):
        raise ValueError('MXFP4 q/scales geometry mismatch')
    rows = tuple(output_rows)
    if (not 1 <= len(rows) <= 16 or any(type(i) is not int or not 0 <= i < q.shape[0] for i in rows)):
        raise ValueError('Bounded output row selection required')
    codes = ((q[list(rows), :, None] >> np.arange(0, 32, 4, dtype=np.uint32)) & 15).astype(np.uint8)
    codes = codes.reshape(len(rows), -1)
    table = np.asarray([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)
    values = np.copysign(table[codes & 7], np.where(codes & 8, -1., 1.)).astype(np.float32)
    return values * np.repeat(decode_scales(np, scales[list(rows)]), 32, axis=1)


def oracle(np, xF32, qU32, scaleU8, *, output_rows=(0, 1, 31, 32), input_rows=None):
    """F64 dot of stored original operands; bounded <=99x16 outputs.

    Original activation values are used: this never rounds to half. Compare a
    half-input candidate against the same truth and report differences honestly.
    """
    if (xF32.dtype != np.float32 or xF32.ndim != 2 or not 1 <= xF32.shape[0] <= 99
            or xF32.shape[1] != qU32.shape[1] * 8 or not np.isfinite(xF32).all()):
        raise ValueError('Bounded finite original activation geometry required')
    rows = tuple(range(xF32.shape[0])) if input_rows is None else tuple(input_rows)
    if not rows or any(type(i) is not int or not 0 <= i < xF32.shape[0] for i in rows):
        raise ValueError('Invalid input row selection')
    weights = decode_rows(np, qU32, scaleU8, output_rows)
    if not np.isfinite(weights).all():
        raise ValueError('Nonfinite decoded weights cannot form finite oracle')
    # Explicit NumPy F64 reduction, not BLAS or a reproduction of Metal's tree.
    return np.stack([np.sum(xF32[list(rows)].astype(np.float64) * w.astype(np.float64), axis=1)
                     for w in weights], axis=1)


def inspect_case(np, case, arrays, host_views, projection='down'):
    if projection not in ('down', 'gate', 'up') or case.get('kind', 'original') != 'original':
        raise ValueError('Only original MXFP4 projections')
    n, k = (5120, 2304) if projection == 'down' else (2304, 5120)
    x, q, scales = arrays['activation'], host_views[projection+'.qweight'], host_views[projection+'.scales']
    if (x.shape != (case['rows'], k) or x.dtype != np.float32
            or q.shape != (n, k//8) or q.dtype != np.uint32
            or scales.shape != (n, k//32) or scales.dtype != np.uint8):
        raise ValueError('Original Source projection geometry mismatch')
    records = {name: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=digest(v))
               for name, v in [('qweight', q), ('scales', scales)]}
    expected = case.get('coefficients')
    if expected is not None and records != expected:
        raise ValueError('Captured original coefficient hash mismatch')
    if expected is None and (records['qweight']['sha256'] != case.get('weight_sha256')
                             or records['scales']['sha256'] != case.get('scales_sha256')):
        raise ValueError('Captured original GU coefficient hash mismatch')
    values, counts = np.unique(scales, return_counts=True)
    return dict(layer=case['layer'], expert=case['expert'], projection=projection, rows=case['rows'],
        shape=[case['rows'], k, n], input_half_roundtrip=half_roundtrip(np, x),
        coefficients=records, scale_counts={f'{int(v):02x}':int(c) for v,c in zip(values,counts)},
        scale_zero_bytes=int((scales == 0).sum()), scale_ff_bytes=int((scales == 255).sum()),
        packed_row_bytes=k//2, scale_row_bytes=k//32, scale_row_padding_bytes=0,
        packed_byte_strides=list(q.view(np.uint8).strides), scale_byte_strides=list(scales.strides),
        nibble_order='low nibble first, eight successive K elements per little-endian uint32',
        scale_semantics='installed F32: 00=2^-127, 01..fe=2^(byte-127), ff=+infinity',
        precision_scope='Operand representation/geometry only; no stock reduction or quality claim')
