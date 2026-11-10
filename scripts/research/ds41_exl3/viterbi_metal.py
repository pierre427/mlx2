"""Metal (MLX custom kernel) tail-biting trellis Viterbi, same contract as viterbi.c.

Original mlx2 code. Two kernels, both bit-identical to viterbi.c:

v1 (KERNEL_VERSION = "v1", and the fallback for window 12 or bit rates whose
per-slot shift leaves {k, k+1} with 2 <= k <= 4): one threadgroup (1024
threads) per 16x16 tile; reduced costs m_i[g] = min over predecessors (2^(w-k_i)
floats per step) in device memory, double buffered; D_i(S) = m_i[S >> k_i] +
e_i(S) recomputed on the fly with the MCG codebook value computed inline;
one backpointer byte per (step, g); thread 0 backtracks after a tie-breaking
(lowest state) argmin reduction.

v2 (default): the same search with costs kept in registers and exchanged
through threadgroup memory, two k=2 steps fused per exchange, packed
backpointers, and an async submit API so host LDLQ work overlaps the GPU.
See the v2 section below.
"""

from __future__ import annotations

import threading

import numpy as np

_KERNEL = None

_SOURCE_V1 = r"""
    const uint tile = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint NT = threads_per_threadgroup.x;
    const int w = params[0];
    const int probe = params[1];
    const int maxg = params[2];
    const uint nstate = 1u << w;
    const uint smask = nstate - 1u;
    device const float* t = targets + tile * 256;
    device float* m0 = mbuf + (size_t)tile * 2 * maxg;
    device float* m1 = m0 + maxg;
    device uchar* bp = bpbuf + (size_t)tile * 256 * maxg;
    threadgroup float red_v[1024];
    threadgroup uint red_s[1024];
    threadgroup int shared_o;
    int k0 = (int)shifts[0];
    uint omask = (1u << (w - k0)) - 1u;
    int o = -1;
    for (int pass = 0; pass < 2; pass++) {
        int len = pass == 0 ? 2 * probe : 256;
        // order(i): pass 0 = 256-probe .. 255, 0 .. probe-1; pass 1 = 0..255
        #define ORDER(i) (pass == 0 ? ((i) < probe ? 256 - probe + (i) : (i) - probe) : (i))
        int s_prev = ORDER(0);
        int kp = (int)shifts[s_prev];
        uint gp = nstate >> kp;
        device float* mp = m0;
        device float* mn = m1;
        for (uint g = tid; g < gp; g += NT) mp[g] = (o < 0 || (int)g == o) ? 0.0f : 3.0e38f;
        threadgroup_barrier(mem_flags::mem_device);
        for (int i = 1; i < len; i++) {
            int s = ORDER(i);
            int k = (int)shifts[s];
            uint gn = nstate >> k;
            float tp = t[s_prev];
            device uchar* b = bp + (size_t)i * maxg;
            for (uint g = tid; g < gn; g += NT) {
                float best = 0.0f;
                uint arg = 0;
                for (uint j = 0; j < (1u << k); j++) {
                    uint S = ((j << (w - k)) | g) & smask;
                    float d = cbval(S, w) - tp;
                    float v = mp[S >> kp] + d * d;
                    if (j == 0 || v < best) { best = v; arg = j; }
                }
                mn[g] = best;
                b[g] = (uchar)arg;
            }
            threadgroup_barrier(mem_flags::mem_device);
            device float* tmp = mp; mp = mn; mn = tmp;
            s_prev = s; kp = k;
        }
        // Final: D(S) = mp[S >> kp] + e(S, t[s_prev]); argmin (lowest S on ties).
        float tl = t[s_prev];
        float bv = 3.4e38f;
        uint bs = 0xFFFFFFFFu;
        for (uint S = tid; S < nstate; S += NT) {
            if (pass == 1 && (int)(S & omask) != o) continue;
            float d = cbval(S, w) - tl;
            float v = mp[S >> kp] + d * d;
            if (v < bv || (v == bv && S < bs)) { bv = v; bs = S; }
        }
        red_v[tid] = bv; red_s[tid] = bs;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = NT / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float v2 = red_v[tid + stride]; uint s2 = red_s[tid + stride];
                if (v2 < red_v[tid] || (v2 == red_v[tid] && s2 < red_s[tid])) { red_v[tid] = v2; red_s[tid] = s2; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            uint S = red_s[0];
            int at255 = -1;
            if (pass == 1) states[tile * 256 + 255] = S;
            if (pass == 0 && probe == 1) at255 = (int)S;  // never: probe >= 2
            for (int i = len - 1; i >= 1; i--) {
                int k = (int)shifts[ORDER(i)];
                uint g = S >> k;
                uint j = bp[(size_t)i * maxg + g];
                S = ((j << (w - k)) | g) & smask;
                if (pass == 1) states[tile * 256 + i - 1] = S;
                if (pass == 0 && i - 1 == probe - 1) at255 = (int)S;
            }
            shared_o = (int)((uint)at255 & omask);
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        o = shared_o;
        #undef ORDER
    }
"""

_HEADER_V1 = r"""
inline float cbval(uint S, int w) {
    uint cw = S & ((1u << w) - 1u);
    uint r = ((cw * 0xCBAC1FEDu) & 0x8FFF8FFFu) ^ 0x3B603B60u;
    half lo = as_type<half>((ushort)(r & 0xFFFFu));
    half hi = as_type<half>((ushort)(r >> 16));
    return (float)(half)((float)lo + (float)hi);
}
"""


def _kernel_v1():
    global _KERNEL
    if _KERNEL is None:
        import mlx.core as mx

        _KERNEL = mx.fast.metal_kernel(
            name="exl3_trellis_viterbi_mcg",
            input_names=["targets", "shifts", "params"],
            output_names=["states", "mbuf", "bpbuf"],
            source=_SOURCE_V1,
            header=_HEADER_V1,
            ensure_row_contiguous=True,
        )
    return _KERNEL


def _viterbi_metal_v1(targets, words, window=16, probe=32, batch=512):
    """v1: targets [N, 256] float32 slot order -> states [N, 256] uint16 (MCG only)."""
    import mlx.core as mx

    from exl3_encode import shifts as _shifts

    targets = np.ascontiguousarray(targets, np.float32)
    n = targets.shape[0]
    sh = _shifts(words)
    maxg = (1 << window) >> int(sh.min())
    params = mx.array([window, probe, maxg], dtype=mx.int32)
    shm = mx.array(sh.astype(np.uint32))
    out = np.empty((n, 256), np.uint16)
    k = _kernel_v1()
    for a in range(0, n, batch):
        b = min(n, a + batch)
        m = b - a
        st, _, _ = k(
            inputs=[mx.array(targets[a:b]), shm, params],
            grid=(1024 * m, 1, 1), threadgroup=(1024, 1, 1),
            output_shapes=[(m, 256), (m, 2 * maxg), (m, 256 * maxg)],
            output_dtypes=[mx.uint32, mx.float32, mx.uint8],
        )
        mx.eval(st)
        out[a:b] = np.array(st).astype(np.uint16)
    return out


# --- v2: register-resident costs, threadgroup exchange ----------------------
#
# Same search, same arithmetic, same tie-breaking as v1 / viterbi.c; only the
# data movement differs. Step i (window w = 16, k_a = k_{i-1}, k_b = k_i)
# decomposes into 2^(16-k_a-k_b) independent butterflies: unit x reads the
# 2^k_b costs m_{i-1}[(j << B) | x] (B = 16-k_a-k_b) and writes the 2^k_a costs
# m_i[(x << k_a) | l], each the min over j of m_{i-1} + e_{i-1}(S) with
# S = (j << (16-k_b)) | (x << k_a) | l. Each of the 1024 threads owns 64
# (S, step) evaluations (whole units, or a quarter/half of a k_a+k_b > 6
# unit), keeps its inputs and outputs in registers, and swaps outputs for the
# next step's inputs through one 32 KB threadgroup buffer (two rounds when
# m_i has 2^14 entries). Backpointers are packed k_b bits per output into one
# uint per thread per step (4 KB per tile-step instead of 16 KB). The MCG
# product S * 0xCBAC1FED is formed by additions (S's bit fields are disjoint).
# Steps are specialised on (k_a, k_b, k_c) in {KLO, KHI}^3, so tiles with a
# fractional bit rate (k alternating KLO/KHI) take the same path.

_HEADER_V2 = r"""
constant constexpr uint NT = 1024;
constant constexpr int WW = 16;
constant constexpr uint XB = 8192;
constant constexpr uint MCG = 0xCBAC1FEDu;
constant constexpr float BIGV = 3.0e38f;

inline float cbv(uint prod) {
    uint r = (prod & 0x8FFF8FFFu) ^ 0x3B603B60u;
    half lo = as_type<half>((ushort)(r & 0xFFFFu));
    half hi = as_type<half>((ushort)(r >> 16));
    return (float)(half)((float)lo + (float)hi);
}

template <int KA, int KB> inline uint out_index(uint tid, int r) {
    constexpr int U = 1 << (KA + KB);
    constexpr int NA = 1 << KA;
    if (U <= 64) {
        uint x = tid + NT * (uint)(r / NA);
        return (x << KA) | (uint)(r % NA);
    } else {
        constexpr uint TPU = U >= 64 ? U / 64 : 1;
        constexpr int C = 64 >> KB;
        uint x = tid / TPU;
        return (x << KA) | ((tid % TPU) * C + (uint)r);
    }
}

template <int KA, int KB> inline uint in_index(uint tid, int s) {
    constexpr int U = 1 << (KA + KB);
    constexpr int NB = 1 << KB;
    constexpr int B = WW - KA - KB;
    if (U <= 64) {
        uint x = tid + NT * (uint)(s / NB);
        return ((uint)(s % NB) << B) | x;
    } else {
        constexpr uint TPU = U >= 64 ? U / 64 : 1;
        return ((uint)s << B) | (tid / TPU);
    }
}

template <int KA, int KB> inline void v2_init(thread float* vin, uint tid, int o) {
    constexpr int U = 1 << (KA + KB);
    constexpr int NIN = U <= 64 ? (64 >> KA) : (1 << KB);
    for (int s = 0; s < NIN; s++) {
        uint idx = in_index<KA, KB>(tid, s);
        vin[s] = (o < 0 || (int)idx == o) ? 0.0f : BIGV;
    }
}

template <int KA, int KB> inline uint v2_compute(thread const float* vin, thread float* vout, uint tid, float tp) {
    constexpr int U = 1 << (KA + KB);
    constexpr int NA = 1 << KA;
    constexpr int NB = 1 << KB;
    constexpr int NOUT = 64 >> KB;
    constexpr uint JM = MCG << (WW - KB);
    uint packed = 0;
    for (int r = 0; r < NOUT; r++) {
        uint g = out_index<KA, KB>(tid, r);
        int base = U <= 64 ? (r / NA) * NB : 0;
        uint p0 = g * MCG;
        float best = 0.0f;
        uint arg = 0;
        for (int j = 0; j < NB; j++) {
            float d = cbv(p0 + (uint)j * JM) - tp;
            float v = vin[base + j] + d * d;
            if (j == 0 || v < best) { best = v; arg = (uint)j; }
        }
        vout[r] = best;
        packed |= arg << (r * KB);
    }
    return packed;
}

template <int KA, int KB, int KC> inline void v2_exchange(thread const float* vout, thread float* vin,
                                                         threadgroup float* xb, uint tid) {
    constexpr int NOUT = 64 >> KB;
    constexpr uint N = 1u << (WW - KB);
    constexpr uint R = N > XB ? N / XB : 1;
    constexpr int U2 = 1 << (KB + KC);
    constexpr int NIN = U2 <= 64 ? (64 >> KB) : (1 << KC);
    for (uint rr = 0; rr < R; rr++) {
        uint lo = rr * XB;
        for (int r = 0; r < NOUT; r++) {
            uint g = out_index<KA, KB>(tid, r) - lo;
            if (g < XB) xb[g] = vout[r];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int s = 0; s < NIN; s++) {
            uint idx = in_index<KB, KC>(tid, s) - lo;
            if (idx < XB) vin[s] = xb[idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// k = 2 everywhere: after v2_compute<2,2> (unit x = tid + 1024 q, outputs
// m_i[(q << 12) | (tid << 2) | l] in vout[4q + l]) the next step's unit
// (tid << 2) | l needs exactly vout[4 j + l] over j, so it runs from registers:
// outputs m_{i+1}[(tid << 4) | (l << 2) | l'] in vout2[4 l + l']. One exchange
// then serves two steps. (An XOR swizzle of the stride-16 stores measured
// 8% slower on M5 Max and is not used.)
inline uint v2_second22(thread const float* vmid, thread float* vout2, uint tid, float tp) {
    constexpr uint JM = MCG << 14;
    uint packed = 0;
    for (int l = 0; l < 4; l++) {
        for (int l2 = 0; l2 < 4; l2++) {
            uint g = (tid << 4) | (uint)(l * 4 + l2);
            uint p0 = g * MCG;
            float best = 0.0f;
            uint arg = 0;
            for (int j = 0; j < 4; j++) {
                float d = cbv(p0 + (uint)j * JM) - tp;
                float v = vmid[4 * j + l] + d * d;
                if (j == 0 || v < best) { best = v; arg = (uint)j; }
            }
            vout2[l * 4 + l2] = best;
            packed |= arg << ((l * 4 + l2) * 2);
        }
    }
    return packed;
}

inline void v2_exchange_second22(thread const float* vout2, thread float* vin, threadgroup float* xb, uint tid) {
    for (uint rr = 0; rr < 2; rr++) {
        uint lo = rr * XB;
        uint gb = (tid << 4) - lo;
        if (gb < XB) {
            for (int r = 0; r < 16; r++) xb[gb + (uint)r] = vout2[r];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int s = 0; s < 16; s++) {
            uint idx = in_index<2, 2>(tid, s) - lo;
            if (idx < XB) vin[s] = xb[idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template <int KA, int KB> inline void v2_final(thread const float* vout, uint tid, float tl, int o, uint omask,
                                               thread float& bv, thread uint& bs) {
    constexpr int NOUT = 64 >> KB;
    constexpr int NB = 1 << KB;
    for (int r = 0; r < NOUT; r++) {
        uint g = out_index<KA, KB>(tid, r);
        for (int b = 0; b < NB; b++) {
            uint S = (g << KB) | (uint)b;
            if (o >= 0 && (int)(S & omask) != o) continue;
            float d = cbv(S * MCG) - tl;
            float v = vout[r] + d * d;
            if (v < bv || (v == bv && S < bs)) { bv = v; bs = S; }
        }
    }
}
"""

_SOURCE_V2 = r"""
    const uint tile = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const int probe = params[0];
    device const float* t = targets + (size_t)tile * 256;
    device uint* bp = bpbuf + (size_t)tile * 256 * NT;
    threadgroup float xb[XB];
    float vin[16];
    float vout[16];
    const int k0 = (int)shifts[0];
    const uint omask = (1u << (WW - k0)) - 1u;
    const uint smask = (1u << WW) - 1u;
    int o = -1;
    for (int pass = 0; pass < 2; pass++) {
        const int len = pass == 0 ? 2 * probe : 256;
        #define ORD(i) (pass == 0 ? ((i) < probe ? 256 - probe + (i) : (i) - probe) : (i))
        #define SH(i) ((int)shifts[ORD(i)])
        #define HI(k) ((k) == KHI ? 1 : 0)
        {
            int ka = SH(0), kb = SH(1);
            switch (HI(ka) * 2 + HI(kb)) {
                case 0: v2_init<KLO, KLO>(vin, tid, o); break;
                case 1: v2_init<KLO, KHI>(vin, tid, o); break;
                case 2: v2_init<KHI, KLO>(vin, tid, o); break;
                default: v2_init<KHI, KHI>(vin, tid, o); break;
            }
        }
        for (int i = 1; i < len; i++) {
            const int ka = SH(i - 1), kb = SH(i);
            const float tp = t[ORD(i - 1)];
            uint packed;
            if (KLO == 2 && KHI == 2 && i + 2 < len) {  // len is even: the last step stays single
                float vmid[16];
                bp[(size_t)i * NT + tid] = v2_compute<2, 2>(vin, vmid, tid, tp);
                bp[(size_t)(i + 1) * NT + tid] = v2_second22(vmid, vout, tid, t[ORD(i)]);
                i++;
                if (i + 1 < len) v2_exchange_second22(vout, vin, xb, tid);
                continue;
            }
            switch (HI(ka) * 2 + HI(kb)) {
                case 0: packed = v2_compute<KLO, KLO>(vin, vout, tid, tp); break;
                case 1: packed = v2_compute<KLO, KHI>(vin, vout, tid, tp); break;
                case 2: packed = v2_compute<KHI, KLO>(vin, vout, tid, tp); break;
                default: packed = v2_compute<KHI, KHI>(vin, vout, tid, tp); break;
            }
            bp[(size_t)i * NT + tid] = packed;
            if (i + 1 < len) {
                const int kc = SH(i + 1);
                switch (HI(ka) * 4 + HI(kb) * 2 + HI(kc)) {
                    case 0: v2_exchange<KLO, KLO, KLO>(vout, vin, xb, tid); break;
                    case 1: v2_exchange<KLO, KLO, KHI>(vout, vin, xb, tid); break;
                    case 2: v2_exchange<KLO, KHI, KLO>(vout, vin, xb, tid); break;
                    case 3: v2_exchange<KLO, KHI, KHI>(vout, vin, xb, tid); break;
                    case 4: v2_exchange<KHI, KLO, KLO>(vout, vin, xb, tid); break;
                    case 5: v2_exchange<KHI, KLO, KHI>(vout, vin, xb, tid); break;
                    case 6: v2_exchange<KHI, KHI, KLO>(vout, vin, xb, tid); break;
                    default: v2_exchange<KHI, KHI, KHI>(vout, vin, xb, tid); break;
                }
            }
        }
        // Final D(S) = m[S >> k_last] + e(S, t_last); argmin, lowest S on ties.
        float bv = 3.4e38f;
        uint bs = 0xFFFFFFFFu;
        {
            const int ka = SH(len - 2), kb = SH(len - 1);
            const float tl = t[ORD(len - 1)];
            const int fo = pass == 1 ? o : -1;
            switch (HI(ka) * 2 + HI(kb)) {
                case 0: v2_final<KLO, KLO>(vout, tid, tl, fo, omask, bv, bs); break;
                case 1: v2_final<KLO, KHI>(vout, tid, tl, fo, omask, bv, bs); break;
                case 2: v2_final<KHI, KLO>(vout, tid, tl, fo, omask, bv, bs); break;
                default: v2_final<KHI, KHI>(vout, tid, tl, fo, omask, bv, bs); break;
            }
        }
        for (ushort off = 16; off > 0; off >>= 1) {
            float v2 = simd_shuffle_xor(bv, off);
            uint s2 = simd_shuffle_xor(bs, off);
            if (v2 < bv || (v2 == bv && s2 < bs)) { bv = v2; bs = s2; }
        }
        if ((tid & 31u) == 0) { xb[tid >> 5] = bv; xb[32 + (tid >> 5)] = as_type<float>(bs); }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        // Thread 0 backtracks serially. (Staging 8 steps of backpointers through
        // threadgroup memory measured 6% slower: the extra barriers cost more
        // than the device-latency chain, which other threadgroups hide.)
        if (tid == 0) {
            for (uint q = 1; q < NT / 32; q++) {
                float v2 = xb[q];
                uint s2 = as_type<uint>(xb[32 + q]);
                if (v2 < bv || (v2 == bv && s2 < bs)) { bv = v2; bs = s2; }
            }
            uint S = bs;
            int at = -1;
            if (pass == 1) states[tile * 256 + 255] = (ushort)S;
            for (int i = len - 1; i >= 1; i--) {
                const int k = SH(i), ka = SH(i - 1);
                const uint g = S >> k;
                const uint x = g >> ka;
                const uint l = g & ((1u << ka) - 1u);
                uint tt, r;
                if (KLO == 2 && KHI == 2 && (i & 1) == 0) {
                    tt = g >> 4; r = g & 15u;  // second step of a fused k = 2 pair
                } else if (ka + k <= 6) {
                    tt = x % NT; r = ((x / NT) << ka) | l;
                } else {
                    const uint tpu = 1u << (ka + k - 6);
                    const uint c = 64u >> k;
                    tt = x * tpu + l / c; r = l % c;
                }
                const uint j = (bp[(size_t)i * NT + tt] >> (r * (uint)k)) & ((1u << k) - 1u);
                S = ((j << (WW - k)) | g) & smask;
                if (pass == 1) states[tile * 256 + i - 1] = (ushort)S;
                if (pass == 0 && i - 1 == probe - 1) at = (int)S;
            }
            xb[64] = as_type<float>((uint)at & omask);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o = (int)as_type<uint>(xb[64]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #undef ORD
        #undef SH
        #undef HI
    }
"""

KERNEL_VERSION = "v2"
_KERNEL_V2 = None
# Serialises MLX calls when a caller also runs MLX on other threads (the
# streaming DS4.1 driver builds fp8 calibration inputs on its prep thread).
MX_LOCK = threading.RLock()


def _kernel_v2():
    global _KERNEL_V2
    if _KERNEL_V2 is None:
        import mlx.core as mx

        _KERNEL_V2 = mx.fast.metal_kernel(
            name="exl3_trellis_viterbi_mcg_v2",
            input_names=["targets", "shifts", "params"],
            output_names=["states", "bpbuf"],
            source=_SOURCE_V2,
            header=_HEADER_V2,
            ensure_row_contiguous=True,
        )
    return _KERNEL_V2


def v2_supported(words, window=16):
    from exl3_encode import shifts as _shifts

    sh = _shifts(words)
    lo, hi = int(sh.min()), int(sh.max())
    return window == 16 and 2 <= lo and hi <= 4 and hi - lo <= 1


class _Pending:
    def __init__(self, parts, n):
        self.parts, self.n = parts, n

    def result(self):
        import mlx.core as mx

        out = np.empty((self.n, 256), np.uint16)
        with MX_LOCK:
            for a, st in self.parts:
                mx.eval(st)
                out[a:a + st.shape[0]] = np.asarray(st)
        return out


def viterbi_metal_submit(targets, words, window=16, probe=32, batch=1024):
    """Queue the v2 search on the GPU and return a handle whose .result() waits."""
    import mlx.core as mx

    from exl3_encode import shifts as _shifts

    if not v2_supported(words, window):
        raise ValueError("v2 needs window 16 and per-slot shifts in {k, k+1}, 2 <= k, k+1 <= 4")
    if not 2 <= probe <= 128:
        raise ValueError("probe must be in [2, 128]")
    targets = np.ascontiguousarray(targets, np.float32)
    n = targets.shape[0]
    sh = _shifts(words)
    klo, khi = int(sh.min()), int(sh.max())
    with MX_LOCK:
        params = mx.array([probe], dtype=mx.int32)
        shm = mx.array(sh.astype(np.uint32))
        return _Pending(_submit_parts(_kernel_v2(), targets, n, batch, shm, params, klo, khi), n)


def _submit_parts(k, targets, n, batch, shm, params, klo, khi):
    import mlx.core as mx

    parts = []
    for a in range(0, n, batch):
        b = min(n, a + batch)
        m = b - a
        st, _ = k(
            inputs=[mx.array(targets[a:b]), shm, params],
            template=[("KLO", klo), ("KHI", khi)],
            grid=(1024 * m, 1, 1), threadgroup=(1024, 1, 1),
            output_shapes=[(m, 256), (m, 256 * 1024)],
            output_dtypes=[mx.uint16, mx.uint32],
            stream=mx.gpu,
        )
        mx.async_eval(st)
        parts.append((a, st))
    return parts


def viterbi_metal(targets, words, window=16, probe=32, batch=None):
    """targets [N, 256] float32 slot order -> states [N, 256] uint16 (MCG only).

    Uses v2 when the bit rate allows it (KERNEL_VERSION = "v2"), else v1.
    """
    if KERNEL_VERSION == "v2" and v2_supported(words, window):
        return viterbi_metal_submit(targets, words, window, probe, batch or 1024).result()
    return _viterbi_metal_v1(targets, words, window, probe, batch or 512)
