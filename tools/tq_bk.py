"""Byte-code TQ decode kernel (layout v2): K codes uint8, transposed in tiles of 32 keys [H, cap/32, D/4, 32] (uint32 = 4 codes of one key), V codes uint8 [H, cap, D].
Lane = key for the K dot (no cross-lane reduce), lane = 8 dims for the V accumulate. Same codes + codebook as TQ6 => same numbers up to summation order."""
import math, numpy as np, mlx.core as mx
from kyojin_mlx import tq

SRC = """
    constexpr int G = D / 32;
    constexpr int D4 = D / 4;
    constexpr int NCH = REP / RC;
    uint split = threadgroup_position_in_grid.x;
    uint hk = threadgroup_position_in_grid.y / NCH;
    int r0 = (int)(threadgroup_position_in_grid.y % NCH) * RC;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint tid = sg * 32 + lane;
    int L = info[0], S = info[1], T0 = info[2], NR = info[3], cap = info[4], NS = info[5];
    int NT = cap / 32;
    threadgroup float cbk_s[1 << KB];
    threadgroup float cbv_s[1 << VB];
    threadgroup float sh[((ABL & 4) ? 4 : 8) * D];
    threadgroup float sml[16];
    if (tid < (1u << KB)) cbk_s[tid] = cbk[tid];
    if (tid < (1u << VB)) cbv_s[tid] = cbv[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float qsh[RC * D];
    for (int i = (int)tid; i < RC * D; i += 256) qsh[i] = (float)q[((int)hk * REP + r0) * D + i] * sgn[i % D];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int len = 1; len < D; len <<= 1) {
        for (int idx = (int)tid; idx < RC * D / 2; idx += 256) {
            int rr = idx / (D / 2), k = idx % (D / 2);
            int a = rr * D + (k / len) * 2 * len + (k % len);
            float x0 = qsh[a], x1 = qsh[a + len];
            qsh[a] = x0 + x1; qsh[a + len] = x0 - x1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float qscl = scl[0];
    for (int i = (int)tid; i < RC * D; i += 256) qsh[i] *= qscl;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc[RC][G];
    for (int r = 0; r < RC; r++) for (int g = 0; g < G; g++) acc[r][g] = 0.0f;
    float m_l[RC], l_l[RC];
    for (int r = 0; r < RC; r++) { m_l[r] = -INFINITY; l_l[r] = 0.0f; }
    bool ov = (int)split == NSPLIT;                                   // the extra split = sink + fp16 ring
    int Pa = S, Pb = T0;
    int j0, j1, ja;
    if (ov) { j0 = 0; j1 = S + (L - T0); ja = 0; }
    else {
        int per = ((Pb - Pa + NSPLIT - 1) / NSPLIT + 31) / 32 * 32;
        int base = Pa & ~31;
        j0 = base + (int)split * per; j1 = min(Pb, j0 + per); ja = Pa;
    }
    for (int tb = j0 + (int)sg * 32; tb < j1; tb += 256) {
        int j = tb + (int)lane;
        bool ok = j >= ja && j < j1;
        int jc = min(max(j, ja), j1 - 1);
        float s[RC];
        for (int r = 0; r < RC; r++) s[r] = 0.0f;
        float kscl = 1.0f, vscl = 1.0f;
        if (!ov) {
            if (ABL & 2) {} else if (KM == 0) {
            const device uint* kp = (const device uint*)kc + ((size_t)((int)hk * NT + (tb >> 5)) * D4) * 32 + lane;
            for (int d4 = 0; d4 < D4; d4 += 4) {
                uint u[4];
                for (int i = 0; i < 4; i++) u[i] = kp[(d4 + i) * 32];
                for (int i = 0; i < 4; i++) {
                    uchar4 c = as_type<uchar4>(u[i]);
                    float4 kv = float4(cbk_s[c.x], cbk_s[c.y], cbk_s[c.z], cbk_s[c.w]);
                    for (int r = 0; r < RC; r++) s[r] += dot(*(threadgroup float4*)(qsh + r * D + (d4 + i) * 4), kv);
                }
            }
            } else {
            constexpr int WL = KB - 4, VPL = 32 / WL;
            const device uint* kp = (const device uint*)kc + ((size_t)((int)hk * NT + (tb >> 5)) * G * KB) * 32 + lane;
            for (int g = 0; g < G; g++) {
                uint wh[4], wl[WL];
                for (int i = 0; i < 4; i++) wh[i] = kp[(g * KB + i) * 32];
                for (int i = 0; i < WL; i++) wl[i] = kp[(g * KB + 4 + i) * 32];
                for (int q4 = 0; q4 < 8; q4++) {
                    float4 kv;
                    for (int e = 0; e < 4; e++) {
                        int i = q4 * 4 + e;
                        uint hi = (wh[i >> 3] >> ((i & 7) * 4)) & 15u;
                        uint lo = (wl[i / VPL] >> ((i % VPL) * WL)) & ((1u << WL) - 1u);
                        if (ABL & 8) kv[e] = cbk_s[(wh[i & 3] >> (i & 7)) & 63u]; else
                        kv[e] = cbk_s[(hi << WL) | lo];
                    }
                    for (int r = 0; r < RC; r++) s[r] += dot(*(threadgroup float4*)(qsh + r * D + g * 32 + q4 * 4), kv);
                }
            }
            }
            int bb = (int)hk * cap + jc; kscl = (float)ks[bb]; vscl = (float)vs[bb];
        } else {
            const device T* kp;
            if (jc < S) kp = sk + ((int)hk * NS + jc) * D; else kp = rk + ((int)hk * NR + (jc - S + T0 - (L - NR))) * D;
            for (int d4 = 0; d4 < D4; d4++) {
                float4 kv = float4(*(const device vec<T, 4>*)(kp + d4 * 4));
                for (int r = 0; r < RC; r++) s[r] += dot(*(threadgroup float4*)(qsh + r * D + d4 * 4), kv);
            }
        }
        float p[RC];
        bool any = false;
        for (int r = 0; r < RC; r++) {
            float sc = ok ? s[r] * kscl : -INFINITY;
            float bm = simd_max(sc);
            float mn = max(m_l[r], bm);
            float al = (mn == -INFINITY) ? 1.0f : exp(m_l[r] - mn);
            p[r] = (sc == -INFINITY) ? 0.0f : exp(sc - mn);
            l_l[r] = l_l[r] * al + p[r];
            m_l[r] = mn;
            for (int g = 0; g < G; g++) acc[r][g] *= al;
            p[r] *= vscl;
        }
        if (!(ABL & 1))
        _Pragma("unroll UNR")
        for (int kk = 0; kk < 32; kk++) {
            int jk = min(max(tb + kk, ja), j1 - 1);
            float vf[G];
            if (!ov) {
                if (VM == 0) {
                    const device uchar* vp = (const device uchar*)vc + ((size_t)((int)hk * cap + jk)) * D + lane * G;
                    for (int g = 0; g < G; g++) vf[g] = cbv_s[vp[g]];
                } else {
                    uint c[G]; tq_codes<VB, G>((const device uint*)vc + ((size_t)((int)hk * cap + jk)) * (G * VB), lane, c);
                    for (int g = 0; g < G; g++) vf[g] = cbv_s[c[g]];
                }
            } else {
                const device T* vp;
                if (jk < S) vp = sv + ((int)hk * NS + jk) * D; else vp = rv + ((int)hk * NR + (jk - S + T0 - (L - NR))) * D;
                for (int g = 0; g < G; g++) vf[g] = (float)vp[lane * G + g];
            }
            for (int r = 0; r < RC; r++) {
                float pk = simd_shuffle(p[r], (ushort)kk);
                for (int g = 0; g < G; g++) acc[r][g] += pk * vf[g];
            }
        }
    }
    for (int r = 0; r < RC; r++) {
        float lsum = simd_sum(l_l[r]);
        if (lane == 0) { sml[sg] = m_l[r]; sml[8 + sg] = lsum; }
        for (int g = 0; g < G; g++) sh[((ABL & 4) ? (sg & 3) : sg) * D + lane * G + g] = acc[r][g];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float mt = -INFINITY;
        for (int s = 0; s < 8; s++) mt = max(mt, sml[s]);
        float lt = 0.0f, o = 0.0f;
        for (int s = 0; s < 8; s++) {
            float w = (sml[s] == -INFINITY) ? 0.0f : exp(sml[s] - mt);
            lt += w * sml[8 + s];
            if (tid < (uint)D) o += w * sh[((ABL & 4) ? (s & 3) : s) * D + tid];
        }
        int row = ((int)hk * (NSPLIT + 1) + (int)split) * REP + r0 + r;
        if (tid < (uint)D) po[row * D + tid] = o;
        if (tid == 0) { pml[row * 2] = mt; pml[row * 2 + 1] = lt; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""
_k = {}
def kernel():
    if "dec" not in _k:
        _k["dec"] = mx.fast.metal_kernel("tq_decode_bk", ["q", "kc", "ks", "vc", "vs", "sk", "sv", "rk", "rv", "cbk", "cbv", "info", "sgn", "scl"], ["po", "pml"], SRC, header=tq._HEADER)
    return _k["dec"]

def relayout_np(words, bits, D):
    """plane words uint32 [H, N, D/32*bits] -> (K tiles uint32 [H, N/32, D/4, 32], V bytes uint8 [H, N, D])"""
    H, N, _ = words.shape
    codes = tq.unpack_ref(words.reshape(H * N, D // 32, bits), bits).reshape(H, N, D).astype(np.uint8)
    return codes

def k_tiles(codes):
    H, N, D = codes.shape
    c = codes.reshape(H, N // 32, 32, D // 4, 4).transpose(0, 1, 3, 2, 4)
    return np.ascontiguousarray(c).view(np.uint32).reshape(H, N // 32, D // 4, 32)

def decode(q, kc, ks, vc, vs, sk, sv, rk, rv, info, D, KB, VB, REP, Hk, NSPLIT, scale, dtype, RC=None, KM=0, VM=0):
    RC = RC or min(REP, 8 if D <= 128 else 4)
    Hq = Hk * REP
    po, pml = kernel()(inputs=[q.reshape(Hq, D), kc, ks, vc, vs, sk, sv, rk, rv, tq._cb_mx(D, KB)[0], tq._cb_mx(D, VB)[0], info, tq._sgn_mx(D), tq._scl_mx(scale, D, False)],
        template=[("D", D), ("KB", KB), ("VB", VB), ("REP", REP), ("RC", RC), ("NSPLIT", NSPLIT), ("KM", KM), ("VM", VM), ("ABL", int(__import__("os").environ.get("ABL", 0))), ("UNR", int(__import__("os").environ.get("UNR", 4))), ("T", dtype)],
        grid=(256 * (NSPLIT + 1), Hk * (REP // RC), 1), threadgroup=(256, 1, 1),
        output_shapes=[(Hk, NSPLIT + 1, REP, D), (Hk, NSPLIT + 1, REP, 2)], output_dtypes=[mx.float32, mx.float32])
    if __import__('os').environ.get('SKIPMRG'): return po[:, 0].reshape(1, Hq, 1, D).astype(dtype)
    out = tq._kernel("mrg")(inputs=[po, pml, tq._sgn_mx(D), tq._scl_mx(scale, D, True)], template=[("D", D), ("REP", REP), ("NSPLIT", NSPLIT + 1), ("T", dtype)],
        grid=(256 * Hq, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(Hq, D)], output_dtypes=[dtype])[0]
    return out.reshape(1, Hq, 1, D)

def k_tiles_planes(words, D, bits):
    """plane words [H, N, D/32*bits] -> [H, N/32, D/32*bits, 32]"""
    H, N, NW = words.shape
    return np.ascontiguousarray(words.reshape(H, N // 32, 32, NW).transpose(0, 1, 3, 2))
