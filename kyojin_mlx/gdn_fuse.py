# SPDX-License-Identifier: Apache-2.0
"""One fused gated-delta decode step for R = 1..8 rows (B = 1, Hk * 128 q / k, Hv * 128 v, conv kernel 4): the front end of gdn_pf.pre (depthwise conv, silu,
q / k L2 norm, g, beta, conv-state shift), the recurrence of mlx_lm's packed step kernel (state in registers, fp32) and the gated output norm of
gdn_pf.gated_norm, in ONE launch per layer: one 512-thread threadgroup per value head (16 simdgroups, 8 value rows per simdgroup, 4 lanes per row). Every
rounding and every reduction order is the one of the three kernels it replaces (conv -> T, silu -> T, rms -> T, scale -> T; recurrence sums in the
packed-kernel tree; norm sums as 4 sequential + simd_sum), so the output is bit-equal to the chain. The final cast saturates for float16.
KYOJIN_GDN_FUSE=0 turns it off (default on); with `capture` the kernel also writes q, k, v, g, beta (what qspec.rollback_gdn replays)."""
import os

import mlx.core as mx

ENABLED = os.environ.get("KYOJIN_GDN_FUSE", "1") != "0"
MAX_ROWS = 8
NSG = int(os.environ.get("KYOJIN_GDN_FUSE_NSG", "16"))      # simdgroups per threadgroup: 16 (8 rows each, 4 lanes per row) or 32 (4 rows, 8 lanes)

_SRC = r"""
    constexpr int LB = 32;
    uint hv = threadgroup_position_in_grid.y;
    uint sg = thread_position_in_threadgroup.y;
    uint l = thread_index_in_simdgroup;
    uint hk = hv / (HV / HK);
    bool own_qk = (hv % (HV / HK)) == 0;

    threadgroup T sh_q[8 * 128];
    threadgroup T sh_k[8 * 128];
    threadgroup T sh_v[8 * 128];
    threadgroup T sh_y[8 * 128];
    threadgroup float sh_g[8];
    threadgroup T sh_b[8];

    // phase A: conv + silu (+ L2 norm for q / k) of this head's 3 x 128 channels for every row; one simdgroup per (row, part)
    for (uint job = sg; job < 3 * S; job += NSG) {
        uint t = job / 3;
        uint part = job % 3;
        uint cb = part == 0 ? hk * 128 : (part == 1 ? HK * 128 + hk * 128 : 2 * HK * 128 + hv * 128);
        uint c0 = cb + 4 * l;
        float y[4];
        for (int i = 0; i < 4; i++) {
            uint c = c0 + i;
            float x[4];
            for (int j = 0; j < 4; j++) { uint r = t + j; x[j] = (r < 3) ? (float)cs[r * C + c] : (float)qkv[(r - 3) * qkv_strides[1] + c]; }
            float acc = (float)w[c * 4] * x[0];
            acc += (float)w[c * 4 + 1] * x[1]; acc += (float)w[c * 4 + 2] * x[2]; acc += (float)w[c * 4 + 3] * x[3];
            float cv = (float)(T)acc;
            float se = 1.0f / (1.0f + metal::precise::exp(metal::abs(cv)));
            float sgm = (float)(T)((cv < 0.0f) ? se : 1.0f - se);
            y[i] = (float)(T)(cv * sgm);
            if (t == 0 && (part == 2 || own_qk)) {
                for (uint r = 0; r < 3; r++) cn[r * C + c] = (r + S < 3) ? cs[(r + S) * C + c] : qkv[(r + S - 3) * qkv_strides[1] + c];
            }
        }
        if (part < 2) {
            float ss = 0.0f;
            for (int i = 0; i < 4; i++) ss += y[i] * y[i];
            ss = simd_sum(ss);
            float r = metal::precise::rsqrt(ss / 128.0f + eps[0]);
            float sc = (float)(T)(part == 0 ? 0.0078125f : 0.08838834764831845f);
            for (int i = 0; i < 4; i++) y[i] = (float)(T)(sc * (float)(T)(y[i] * r));
        }
        threadgroup T* dst = part == 0 ? sh_q : (part == 1 ? sh_k : sh_v);
        for (int i = 0; i < 4; i++) dst[t * 128 + 4 * l + i] = (T)y[i];
    }
    if (sg == NSG - 1 && l == 0) {
        for (uint t = 0; t < S; t++) {
            float bb = (float)bv[t * bv_strides[1] + hv];
            float be = 1.0f / (1.0f + metal::precise::exp(metal::abs(bb)));
            sh_b[t] = (T)((bb < 0.0f) ? be : 1.0f - be);
            float xs = (float)(T)((float)av[t * av_strides[1] + hv] + (float)dtb[hv]);
            float sp = (float)(T)(max(xs, 0.0f) + metal::precise::log(1.0f + metal::precise::exp(-metal::abs(xs))));
            sh_g[t] = metal::precise::exp(-metal::precise::exp(Al[hv]) * sp);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // phase B: the packed recurrence, 8 value rows per simdgroup, state in registers
    {
        constexpr int RPS = 128 / NSG;              // value rows per simdgroup
        constexpr int LPR = 32 / RPS;               // lanes per row
        constexpr int EPL = 128 / LPR;              // state elements per lane
        constexpr int NP = EPL / 4;                 // 4-element partials per lane
        uint row_in = l / LPR;
        uint lir = l & (LPR - 1);
        uint dv = sg * RPS + row_in;
        uint n = hv;
        const device float* i_state = state_in + (n * 128 + dv) * 128 + lir * EPL;
        device float* o_state = state_out + (n * 128 + dv) * 128 + lir * EPL;
        float state[EPL];
        for (int i = 0; i < EPL; ++i) state[i] = i_state[i];
        for (uint t = 0; t < S; ++t) {
            float gt = sh_g[t];
            const threadgroup T* k_ = sh_k + t * 128 + lir * EPL;
            const threadgroup T* q_ = sh_q + t * 128 + lir * EPL;
            float part[NP];
            for (int pb = 0; pb < NP; ++pb) {
                float acc = 0.0f;
                for (int i = 0; i < 4; ++i) { int e = pb * 4 + i; state[e] = state[e] * gt; acc = fma(state[e], (float)k_[e], acc); }
                part[pb] = acc;
            }
            float kv_mem = TREE;
            kv_mem += simd_shuffle_xor(kv_mem, 1);
            kv_mem += simd_shuffle_xor(kv_mem, 2);
            if (LPR == 8) kv_mem += simd_shuffle_xor(kv_mem, 4);
            float delta = ((float)sh_v[t * 128 + dv] - kv_mem) * (float)sh_b[t];
            for (int pb = 0; pb < NP; ++pb) {
                float acc = 0.0f;
                for (int i = 0; i < 4; ++i) { int e = pb * 4 + i; state[e] = fma((float)k_[e], delta, state[e]); acc = fma(state[e], (float)q_[e], acc); }
                part[pb] = acc;
            }
            float o = TREE;
            o += simd_shuffle_xor(o, 1);
            o += simd_shuffle_xor(o, 2);
            if (LPR == 8) o += simd_shuffle_xor(o, 4);
            if (lir == 0) sh_y[t * 128 + dv] = (T)o;
        }
        for (int i = 0; i < EPL; ++i) o_state[i] = state[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // phase C: gated RMS norm, one simdgroup per row
    for (uint t = sg; t < S; t += NSG) {
        const threadgroup T* hp = sh_y + t * 128 + 4 * l;
        const device T* zp = z + t * z_strides[1] + hv * 128 + 4 * l;
        float xs[4];
        float acc = 0.0f;
        for (int i = 0; i < 4; i++) { xs[i] = (float)hp[i]; acc += xs[i] * xs[i]; }
        acc = simd_sum(acc);
        float nrm = metal::precise::rsqrt(acc / 128.0f + nw[0]);
        for (int i = 0; i < 4; i++) {
            float o = (float)(T)((float)w2[4 * l + i] * (xs[i] * nrm));
            float gt = (float)zp[i];
            float e = 1.0f / (1.0f + metal::precise::exp(metal::abs(gt)));
            float s = (float)(T)(gt * ((gt < 0.0f) ? e : 1.0f - e));
            float res = s * o;
            if (FP16) res = metal::clamp(res, -65504.0f, 65504.0f);
            OUTSTORE
        }
    }
    CAPTURE
"""
_CAP = """
    for (uint idx = sg * 32 + l; idx < S * 128; idx += NSG * 32) {
        uint t = idx / 128, j = idx % 128;
        if (own_qk) { q[t * HK * 128 + hk * 128 + j] = sh_q[idx]; k[t * HK * 128 + hk * 128 + j] = sh_k[idx]; }
        v[t * HV * 128 + hv * 128 + j] = sh_v[idx];
    }
    if (sg == 0 && l < S) { g[l * HV + hv] = sh_g[l]; beta[l * HV + hv] = sh_b[l]; }
"""
_TREE = "(NP == 8 ? ((part[0] + part[1]) + (part[2] + part[3])) + ((part[4] + part[5]) + (part[6] + part[7])) : ((part[0] + part[1]) + (part[2] + part[3])))"
_k = {}
_eps = {}


def _kernel(perm, capture):
    key = (perm, capture)
    if key not in _k:
        store = ("out[t * (HV * 128) + inv[hv * 128 + 4 * l + i]] = (T)res;" if perm else "out[t * (HV * 128) + hv * 128 + 4 * l + i] = (T)res;")
        src = _SRC.replace("OUTSTORE", store).replace("CAPTURE", _CAP if capture else "").replace("TREE", _TREE)
        names = ["qkv", "cs", "w", "av", "bv", "Al", "dtb", "eps", "state_in", "z", "w2", "nw"] + (["inv"] if perm else [])
        outs = ["out", "cn", "state_out"] + (["q", "k", "v", "g", "beta"] if capture else [])
        _k[key] = mx.fast.metal_kernel(name="kyojin_gdn_fuse" + ("_p" if perm else "") + ("_c" if capture else ""), input_names=names, output_names=outs,
                                       source=src, header="#pragma METAL fp contract(off)\n", ensure_row_contiguous=False)
    return _k[key]


def _e(x):
    r = _eps.get(x)
    if r is None:
        r = _eps[x] = mx.array([x], dtype=mx.float32)
    return r


def fused_step(qkv, z, a, b, cs, state, conv_w, A_log, dt_bias, norm_w, norm_eps, hk, hv, inv=None, capture=False):
    """qkv [1,S,C], z [1,S,hv,128], a / b [1,S,hv], cs [1,3,C], state [1,hv,128,128] f32 -> (out [1,S,hv*128], conv state [1,3,C], state[, q, k, v, g, beta]).
    out is written permuted through inv (uint32 [hv * 128]) when given (the V-tiled out_proj input)."""
    S, C = qkv.shape[1], qkv.shape[-1]
    T = qkv.dtype
    outs = [(1, S, hv * 128), (1, 3, C), state.shape]
    dts = [T, T, mx.float32]
    if capture:
        outs += [(1, S, hk, 128), (1, S, hk, 128), (1, S, hv, 128), (1, S, hv), (1, S, hv)]
        dts += [T, T, T, mx.float32, T]
    ins = [qkv, cs, conv_w, a, b, A_log, dt_bias, _e(1e-6 / 128.0), state, z, norm_w, _e(norm_eps)]
    if inv is not None:
        ins.append(inv)
    return _kernel(inv is not None, capture)(inputs=ins, template=[("T", T), ("C", C), ("S", S), ("HK", hk), ("HV", hv), ("FP16", 1 if T == mx.float16 else 0), ("NSG", NSG)],
                                             grid=(32, NSG * hv, 1), threadgroup=(32, NSG, 1), output_shapes=outs, output_dtypes=dts)
