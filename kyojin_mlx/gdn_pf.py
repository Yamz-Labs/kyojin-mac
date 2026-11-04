# SPDX-License-Identifier: Apache-2.0
"""Fused GDN prefill block (B = 1, S >= MIN_T rows, conv kernel 4, head dim 128): conv1d + silu + split + L2-normalize of q / k + beta = sigmoid(b) +
g = exp(-exp(A_log) * softplus(a + dt_bias)) in ONE launch over [rows, channel groups] (stock: concat of the conv state, depthwise conv, silu, split, 2 rms_norm, 2 scalar
multiplies, g / beta prep = ~12 passes over [S, 8192]); the recurrence runs on mx.fast.gated_delta_update (chunked, same fp32 state layout) and the gated output norm is
one launch for bf16 / fp16 or fp32 norm weights. On a neural-accelerator GPU the recurrence core is gdn_nax (MLX chunked NAX kernel with fp32 g / beta: mx.fast.gated_delta_update casts them to bf16, which costs 8x the
logit KL of a chunk-size change on real text); elsewhere the step-wise mlx_lm kernel. Roundings of the stock graph are mirrored (conv -> T, silu -> T, rms_norm -> T, scalar multiply -> T).
Everything else (decode, masks, other shapes) goes to the stock call. KYOJIN_GDN_PF=0 restores it; KYOJIN_GDN_PF=<n> sets the minimum rows (default 64)."""
import os

import mlx.core as mx
import mlx.nn as nn

from . import decfuse as DF

MIN_T = int(os.environ.get("KYOJIN_GDN_PF", "64"))
CORE = os.environ.get("KYOJIN_GDN_PF_CORE", "auto")        # auto = nax on a neural-accelerator GPU, else step; step = the mlx_lm step kernel; nax = gdn_nax (chunked, fp32 g / beta); fast / fast32 = mx.fast.gated_delta_update (experiments)
DEC = os.environ.get("KYOJIN_GDN_DEC", "0") == "1"           # the same fused front end + norm for decode and verify rows (S <= 8; the step kernel is the core below MIN_T rows, so decode and verify chunks stay row-consistent)
GDN_PROJ_ROWS = int(os.environ.get("KYOJIN_DECFUSE_GDN_PROJ_ROWS", "5"))
NORM = os.environ.get("KYOJIN_GDN_PF_NORM", "1") != "0"
_PRE = """
    uint gi = threadgroup_position_in_grid.x;
    uint t = threadgroup_position_in_grid.y;
    uint l = thread_index_in_simdgroup;
    if (gi < NH) {
        uint c0 = gi * 128 + 4 * l;
        float y[4];
        for (int i = 0; i < 4; i++) {
            uint c = c0 + i;
            float x[4];
            for (int j = 0; j < 4; j++) { uint r = t + j; x[j] = (r < 3) ? (float)cs[r * C + c] : (float)qkv[(r - 3) * qkv_strides[1] + c]; }
            float acc = (float)w[c * 4] * x[0];
            acc += (float)w[c * 4 + 1] * x[1]; acc += (float)w[c * 4 + 2] * x[2]; acc += (float)w[c * 4 + 3] * x[3];
            float cv = (float)(T)acc;
            float se = 1.0f / (1.0f + metal::precise::exp(metal::abs(cv)));
            float sg = (float)(T)((cv < 0.0f) ? se : 1.0f - se);
            y[i] = (float)(T)(cv * sg);
            if (t + 3 >= S) cn[(t + 3 - S) * C + c] = qkv[t * qkv_strides[1] + c];
            if (t == 0 && S < 3) for (uint k2 = 0; k2 < 3 - S; k2++) cn[k2 * C + c] = cs[(k2 + S) * C + c];       // decode / short chunks: the older conv-state rows shift down
        }
        if (gi < 2 * HK) {
            float ss = 0.0f;
            for (int i = 0; i < 4; i++) ss += y[i] * y[i];
            ss = simd_sum(ss);
            float r = metal::precise::rsqrt(ss / 128.0f + eps[0]);
            float sc = (float)(T)(gi < HK ? 0.0078125f : 0.08838834764831845f);
            for (int i = 0; i < 4; i++) y[i] = (float)(T)(sc * (float)(T)(y[i] * r));
        }
        for (int i = 0; i < 4; i++) {
            if (gi < HK) q[t * HK * 128 + c0 + i] = (T)y[i];
            else if (gi < 2 * HK) k[t * HK * 128 + c0 - HK * 128 + i] = (T)y[i];
            else v[t * HV * 128 + c0 - 2 * HK * 128 + i] = (T)y[i];
        }
    } else {
        for (uint h = l; h < HV; h += 32) {
            float bb = (float)bv[t * bv_strides[1] + h];
            float be = 1.0f / (1.0f + metal::precise::exp(metal::abs(bb)));
            beta[t * HV + h] = (T)((bb < 0.0f) ? be : 1.0f - be);
            float xs = (float)(T)((float)av[t * av_strides[1] + h] + (float)dtb[h]);
            float sp = (float)(T)(max(xs, 0.0f) + metal::precise::log(1.0f + metal::precise::exp(-metal::abs(xs))));
            g[t * HV + h] = metal::precise::exp(-metal::precise::exp(Al[h]) * sp);
        }
    }
"""
_NORM = """
    uint row = thread_position_in_grid.y;
    uint l = thread_index_in_simdgroup;
    const device T* hp = h + row * 128 + 4 * l;
    const device T* zp = z + (row / HV) * z_strides[1] + (row % HV) * 128 + 4 * l;          // z may be a column slice of a grouped projection output (one sequence)
    float xs[4];
    float acc = 0.0f;
    for (int i = 0; i < 4; i++) { xs[i] = (float)hp[i]; acc += xs[i] * xs[i]; }
    acc = simd_sum(acc);
    float nrm = metal::precise::rsqrt(acc / 128.0f + eps[0]);
    for (int i = 0; i < 4; i++) {
        float o = (float)(T)((float)w[4 * l + i] * (xs[i] * nrm));
        float gt = (float)zp[i];
        float e = 1.0f / (1.0f + metal::precise::exp(metal::abs(gt)));
        float s = (float)(T)(gt * ((gt < 0.0f) ? e : 1.0f - e));
        out[row * 128 + 4 * l + i] = (T)(s * o);
    }
"""
# Same kernel with the permuted input of a V-tiled out_proj (ForeignLinear._in_inv): element c of a row is written at column inv[c], so the Gather that
# would reorder the activation before the projection is gone.
_NORM_P = _NORM.replace("out[row * 128 + 4 * l + i] = (T)(s * o);",
                        "out[(row / HV) * (HV * 128) + inv[(row % HV) * 128 + 4 * l + i]] = (T)(s * o);")
_k = {}
_eps = {}


def pre(qkv, cs, w, a, b, A_log, dt_bias, hk, hv):
    """qkv [1,S,C], cs [1,3,C], w [C,4,1], a / b [1,S,hv] -> q [1,S,hk,128], k, v [1,S,hv,128], conv state [1,3,C], g [1,S,hv] f32, beta [1,S,hv]."""
    if "pre" not in _k:
        _k["pre"] = mx.fast.metal_kernel(name="kyojin_gdn_pf_pre", input_names=["qkv", "cs", "w", "av", "bv", "Al", "dtb", "eps"],
                                         output_names=["q", "k", "v", "cn", "g", "beta"], source=_PRE, header="#pragma METAL fp contract(off)\n",
                                         ensure_row_contiguous=False)
        _eps["pre"] = mx.array([1e-6 / 128.0], dtype=mx.float32)
    S, C = qkv.shape[1], qkv.shape[-1]
    T = qkv.dtype
    nh = 2 * hk + hv
    return _k["pre"](inputs=[qkv, cs, w, a, b, A_log, dt_bias, _eps["pre"]], template=[("T", T), ("C", C), ("S", S), ("HK", hk), ("HV", hv), ("NH", nh)],
                     grid=((nh + 1) * 32, S, 1), threadgroup=(32, 1, 1),
                     output_shapes=[(1, S, hk, 128), (1, S, hk, 128), (1, S, hv, 128), (1, 3, C), (1, S, hv), (1, S, hv)],
                     output_dtypes=[T, T, T, T, mx.float32, T])


def gated_norm(h, gate, weight, eps, inv=None):
    """Gated RMS norm of the recurrence output; inv (uint32 [HV * 128], one sequence) writes the row permuted: out[inv[c]] = value of column c."""
    if inv is not None and gate.ndim == 4 and gate.shape[0] == 1 and gate.shape[-1] == 128 and inv.size == gate.shape[-2] * 128:
        if "normp" not in _k:
            _k["normp"] = mx.fast.metal_kernel(name="kyojin_gdn_pf_norm_p", input_names=["h", "z", "w", "eps", "inv"], output_names=["out"], source=_NORM_P,
                                               header="#pragma METAL fp contract(off)\n", ensure_row_contiguous=False)
        e = _eps.get(eps)
        if e is None:
            e = _eps[eps] = mx.array([eps], dtype=mx.float32)
        return _k["normp"](inputs=[h, gate, weight, e, inv], template=[("T", h.dtype), ("HV", gate.shape[-2])], grid=(32, h.size // 128, 1), threadgroup=(32, 1, 1),
                           output_shapes=[h.shape], output_dtypes=[h.dtype])[0]
    if "norm" not in _k:
        _k["norm"] = mx.fast.metal_kernel(name="kyojin_gdn_pf_norm", input_names=["h", "z", "w", "eps"], output_names=["out"], source=_NORM,
                                          header="#pragma METAL fp contract(off)\n", ensure_row_contiguous=False)
    e = _eps.get(eps)
    if e is None:
        e = _eps[eps] = mx.array([eps], dtype=mx.float32)
    if gate.ndim != 4 or gate.shape[0] != 1:                 # the kernel reads the gate through its sequence stride: one sequence [1, S, HV, 128] (a column slice is read in place)
        gate = mx.contiguous(gate.reshape(1, -1, *gate.shape[-2:]))
    return _k["norm"](inputs=[h, gate, weight, e], template=[("T", h.dtype), ("HV", gate.shape[-2])], grid=(32, h.size // 128, 1), threadgroup=(32, 1, 1),
                      output_shapes=[h.shape], output_dtypes=[h.dtype])[0]


BA = os.environ.get("KYOJIN_GDN_BA_FUSE", "1") != "0"                  # in_proj_b | in_proj_a (dense) in one GEMV


def _dense_fl(m):
    """True for a bias-free dense ForeignLinear without an input permutation."""
    from . import foreign_linear as FL
    return type(m) is FL.ForeignLinear and m.dev.fmt == "dense" and m.dev.E is None and m._in_perm is None and m._bias is None


def prep_ba(self) -> None:
    """Build the joint b | a weight now (eager: a compiled region must not evaluate arrays while it traces). The two linears keep views into it (one
    resident copy). Plain bias-free nn.Linear or dense ForeignLinear of one dtype and width; anything else runs apart."""
    if getattr(self, "_ba_w", None) is not None:
        return
    b, a = self.in_proj_b, self.in_proj_a
    kind = None
    if BA and type(b) is nn.Linear and type(a) is nn.Linear and "bias" not in b and "bias" not in a:
        kind, wb, wa = "lin", b.weight, a.weight
    elif BA and _dense_fl(b) and _dense_fl(a) and b.out_dtype == a.out_dtype:
        kind, wb, wa = "fl", b.dev.w, a.dev.w
    if kind is None or wb.dtype != wa.dtype or wb.shape[1] != wa.shape[1]:
        object.__setattr__(self, "_ba_w", False)
        return
    w = mx.concatenate([wb, wa], axis=0)
    mx.eval(w)
    nb = wb.shape[0]
    if kind == "lin":
        b.weight, a.weight = w[:nb], w[nb:]
        mx.eval(b.weight, a.weight)
    else:
        b.dev.w, a.dev.w = w[:nb], w[nb:]
        mx.eval(b.dev.w, a.dev.w)
    object.__setattr__(self, "_ba_w", (w, nb, kind, b.out_dtype if kind == "fl" else None))


def proj_ba(self, x):
    """(in_proj_b(x), in_proj_a(x)) as slices of one GEMV over the concatenated rows."""
    prep_ba(self)
    w = self._ba_w
    if w is False:
        return self.in_proj_b(x), self.in_proj_a(x)
    w, nb, kind, odt = w
    if kind == "fl":
        x2 = x.reshape(-1, x.shape[-1])
        from . import foreign_linear as FL
        y = FL._matmul(x2 if x2.dtype == mx.float16 else x2.astype(mx.float16), w)
        if y.dtype != odt:
            y = y.astype(odt)
        y = y.reshape(*x.shape[:-1], -1)
    else:
        y = x @ w.T
    return y[..., :nb], y[..., nb:]


SCATTER = os.environ.get("KYOJIN_GDN_NORM_SCATTER", "1") != "0"       # the gated norm writes the V-tiled out_proj input permuted (no Gather launch)


def norm_out(self, y, z, dtype):
    """out_proj(gated_norm(y, z)): a V-tiled out_proj (ForeignLinear with an input permutation) takes the permuted row straight from the norm kernel."""
    B, S = y.shape[0], y.shape[1]
    op = self.out_proj
    inv = getattr(op, "_in_inv", None)
    if inv is not None and B == 1 and SCATTER:
        out = gated_norm(y.astype(dtype), z, self.norm.weight, self.norm.eps, inv=inv)
        return op(out.reshape(B, S, -1), permuted=True)
    return op(gated_norm(y.astype(dtype), z, self.norm.weight, self.norm.eps).reshape(B, S, -1))


def fused_mixer(self, qkv, z, a, b, cs, state, capture=False):
    """KYOJIN_GDN_FUSE: conv state, front end, step recurrence and gated norm in one launch (gdn_fuse), then out_proj. -> (out, conv state, state, capture tuple)."""
    from . import gdn_fuse
    inv = getattr(self.out_proj, "_in_inv", None)
    perm = inv is not None and SCATTER and qkv.shape[0] == 1
    r = gdn_fuse.fused_step(qkv, z, a, b, cs, state, self.conv1d.weight, self.A_log, self.dt_bias, self.norm.weight, self.norm.eps, self.num_k_heads, self.num_v_heads,
                            inv=inv if perm else None, capture=capture)
    out = self.out_proj(r[0], permuted=True) if perm else self.out_proj(r[0])
    return out, r[1], r[2], r[3:]


def _fuse_on(S):
    from . import gdn_fuse
    return gdn_fuse.ENABLED and 1 <= S <= gdn_fuse.MAX_ROWS


ROWS_MAX = int(os.environ.get("KYOJIN_DECFUSE_GDN_ROWS", "128"))     # most rows of a decode / verify round that take forward_rows (foreign packs; the capture call of qspec)


def rows_ok(self, inputs, mask, cache):
    """True when forward_rows serves this call: the gated-delta decode class of decfuse for 1..ROWS_MAX rows, one sequence, no padding mask."""
    B, S, _ = inputs.shape
    return (DF.ENABLED and DF.GDN and 1 <= S <= ROWS_MAX and B == 1 and mask is None and cache is not None and cache.lengths is None and self.sharding_group is None
            and not self.training and mx.default_device() == mx.gpu and inputs.dtype in (mx.float16, mx.bfloat16) and self.head_k_dim == 128 and self.head_v_dim == 128
            and self.conv_kernel_size == 4 and self.dt_bias.dtype == inputs.dtype and self.norm.weight.size == 128 and (cache[1] is None or cache[1].dtype == mx.float32))


def forward_rows(self, inputs, cache):
    """Gated-delta mixer for 1..ROWS_MAX rows in 3 launches around the recurrence (front end: conv, silu, q / k L2 norm, g, beta; gated output norm), the step
    kernel as core so decode and verify rows agree. Returns (out, rec): rec = (conv state before, qkv, q, k, v, g, beta, state before) for a rollback to a
    shorter prefix (qspec.rollback_gdn). Stock: ~25 ops per layer."""
    from mlx_lm.models.gated_delta import gated_delta_kernel
    B, S, _ = inputs.shape
    qkv, z = self.in_proj_qkv(inputs), self.in_proj_z(inputs)
    b, a = proj_ba(self, inputs)
    z = z.reshape(B, S, self.num_v_heads, self.head_v_dim)
    cs = cache[0] if cache[0] is not None else mx.zeros((B, 3, self.conv_dim), dtype=inputs.dtype)
    state0 = cache[1] if cache[1] is not None else mx.zeros((B, self.num_v_heads, 128, 128), dtype=mx.float32)
    if _fuse_on(S):
        from . import qspec
        out, cache[0], cache[1], cap = fused_mixer(self, qkv, z, a, b, cs, state0, capture=qspec.CAP is not None)
        cache.advance(S)
        return out, (cs, qkv) + (tuple(cap[:3]) + (cap[3], cap[4]) if cap else (None,) * 5) + (state0,)
    q, k, v, cn, g, beta = pre(qkv, cs, self.conv1d.weight, a, b, self.A_log, self.dt_bias, self.num_k_heads, self.num_v_heads)
    cache[0] = cn
    y, state = gated_delta_kernel(q, k, v, g, beta, state0, None)
    cache[1] = state
    cache.advance(S)
    return norm_out(self, y, z, inputs.dtype), (cs, qkv, q, k, v, g, beta, state0)


def install(prefill=True, dec=None):
    """Patch GatedDeltaNet.__call__: fused prefill chunks (S >= MIN_T) when `prefill`, fused decode / verify rows (S <= 8) when `dec` (default
    KYOJIN_GDN_DEC). The mlx-pack loader installs the verify-row path only, so verify rows round like the decfuse 1-row regions."""
    global DEC
    from mlx_lm.models import qwen3_5 as Q5
    cls = Q5.GatedDeltaNet
    if dec is not None:
        DEC = DEC or bool(dec)
    pf_ok = prefill and MIN_T > 0
    if not (pf_ok or DEC) or getattr(cls, "_kyojin_pf", False):
        return False
    stock = cls.__call__

    def call(self, inputs, mask=None, cache=None):
        B, S, _ = inputs.shape
        if (((pf_ok and S >= MIN_T and S >= 3) or (DEC and S <= 8)) and B == 1 and mask is None and cache is not None and cache.lengths is None and self.sharding_group is None and not self.training
                and mx.default_device() == mx.gpu and inputs.dtype in (mx.float16, mx.bfloat16) and self.head_k_dim == 128 and self.head_v_dim == 128
                and self.conv_kernel_size == 4 and self.dt_bias.dtype == inputs.dtype and self.norm.weight.size == 128
                and (cache[1] is None or cache[1].dtype == mx.float32)):
            fq = getattr(self, "_kr_fused", None) if S <= GDN_PROJ_ROWS else None       # decfuse: qkv | z | b | a in one GEMV (bit-equal; neutral to 5 rows, slower at 6..8)
            if fq is not None and DF.ENABLED and DF.PROJ:
                qkv, z, b, a = fq.parts(inputs)
            else:
                qkv, z = self.in_proj_qkv(inputs), self.in_proj_z(inputs)
                b, a = proj_ba(self, inputs)
            z = z.reshape(B, S, self.num_v_heads, self.head_v_dim)
            cs = cache[0] if cache[0] is not None else mx.zeros((B, 3, self.conv_dim), dtype=inputs.dtype)
            state = cache[1] if cache[1] is not None else mx.zeros((B, self.num_v_heads, 128, 128), dtype=mx.float32)
            if S <= 8 and not (pf_ok and S >= MIN_T) and _fuse_on(S) and NORM:
                out, cache[0], cache[1], _ = fused_mixer(self, qkv, z, a, b, cs, state)
                cache.advance(S)
                return out
            q, k, v, cn, g, beta = pre(qkv, cs, self.conv1d.weight, a, b, self.A_log, self.dt_bias, self.num_k_heads, self.num_v_heads)
            cache[0] = cn
            core = CORE if (pf_ok and S >= MIN_T) else "step"
            if core == "auto":
                from . import metal_nax
                core = "nax" if metal_nax.nax_available() else "step"
            if core == "fast16":
                y, state = mx.fast.gated_delta_update(q.astype(mx.float16), k.astype(mx.float16), v.astype(mx.float16), g, beta.astype(mx.float16), state)
            elif core == "fast32":
                y, state = mx.fast.gated_delta_update(q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), g, beta.astype(mx.float32), state)
            elif core == "nax":
                from .gdn_nax import gated_delta_nax
                y, state = gated_delta_nax(q, k, v, g, beta, state)
            elif core == "fast":
                y, state = mx.fast.gated_delta_update(q, k, v, g, beta, state)
            else:
                from mlx_lm.models.gated_delta import gated_delta_kernel
                y, state = gated_delta_kernel(q, k, v, g, beta, state, None)
            cache[1] = state
            cache.advance(S)
            if NORM:
                return norm_out(self, y, z, inputs.dtype)
            return self.out_proj(self.norm(y.astype(inputs.dtype), z).reshape(B, S, -1))
        return stock(self, inputs, mask, cache)

    cls.__call__ = call
    cls._kyojin_pf = True
    return True
