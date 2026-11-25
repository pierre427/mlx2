"""Source inspection: large-buffer Metal offsets are 64-bit at the first multiply.

Neywa-Labs/mlx-vlm PR #2831: a sparse-attention kernel computed its key
offset ``((b*H + h)*L + t)*D`` in 32-bit ``int`` and read out of bounds (NaN
on the last heads) once the product passed 2^31.  The 2026-11-25 audit found
two kernels of that shape in mlx2.  These checks read the kernel source
strings only; nothing is compiled or run on the GPU.
"""
import re

from mlx2.runtime.models import fp_verify_tensor as fp
from mlx2.runtime.models import hy_v4_flash_fusions as hy


def _code(source):
    return re.sub(r"//[^\n]*", "", source)


def test_fp_verify_kv_and_mask_offsets_promote_the_lane_index():
    source = _code(fp.SOURCE)
    # B*HK*N*128 passes 2^31 at 16 lanes x 8 KV heads x 128K tokens.
    assert "k[((ulong(b)*HK+hk)*N+t)*128+kk+fn+j]" in source
    assert "v[((ulong(b)*HK+hk)*N+t)*128+np*32+tn*16+fn+j]" in source
    # B*HQ*L*N passes 2^31 at 16 lanes x 32 heads x 16 rows x 256K.
    assert "mask[((ulong(b)*HQ+h)*L+row)*N+t]" in source
    for unpromoted in ("((b*HK+hk)*N+t)", "((b*HQ+h)*L+row)*N"):
        assert unpromoted not in source


def test_hy4_hc_seam_row_offsets_are_size_t():
    # K = hc_mult*hidden = 4*3072 = 12288: r*K wraps uint at 349,526 rows,
    # reachable with an uncapped --prefill-step on the 1M-context model.
    partial = _code(hy._PARTIAL_SRC)
    finalize = _code(hy._FINALIZE_SRC)
    merge = _code(hy._MERGE_SRC)
    assert "residual[size_t(r) * K + k]" in partial
    assert "streams[size_t(r) * K + j * D + d]" in finalize
    assert "normed[size_t(r) * D + d]" in finalize
    assert "branch[size_t(r) * D + d]" in merge
    assert "merged[size_t(r) * K + k]" in merge
    for source in (partial, finalize, merge):
        for buffer in ("residual", "streams", "normed", "branch", "merged"):
            assert not re.search(rf"\b{buffer}\[r \*", source), buffer


def test_promotions_keep_the_merge_into_the_partial_source():
    # The merge body is spliced into the partial kernel at /*MERGE*/; the
    # promoted merge must reach the compiled source unchanged.
    spliced = hy._PARTIAL_SRC.replace("/*MERGE*/", hy._MERGE_SRC)
    assert "merged[size_t(r) * K + k]" in spliced
    assert "residual[size_t(r) * K + k]" in spliced
