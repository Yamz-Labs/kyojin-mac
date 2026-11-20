"""KYOJIN_HC_FUSE: the gated hyper-connection module (mlx-vlm qwen4_exp `Qwen4ExpGatedResidual`) at decode as 3 fused Metal kernels around the stock linears.
Stock chain (compiled by HC_COMPILE it is 14 launches): rms norm (grouped, f32) -> down -> silu(x / hc) -> up -> sigmoid * streams, mean -> inject linear -> 2 * sigmoid(x / hc).
down / up are Q8_0 quantised packs of the GGUF (reduction order of the quantised matmul is the library's), inject is a dense F32 / half 4-row matrix: all three stay the
module's own linears (not re-implemented: that would change their accumulation order), and the element-wise / norm parts around them become
  K1 norm (one threadgroup per (row, stream): f32 sum of squares, rsqrt, (1 + w), round to T)
  K2 act (silu(down / hc) and 2 * sigmoid(inject / hc) in one launch)
  K3 mix (sigmoid(up) * normed rounded to T per stream, f32 sum over streams / hc, round to T)
= 3 kernels + 3 linears = 6 launches per call (stock compiled: 14). Every rounding point of the stock T-typed ops is kept (mean = sum * (1 / N), the 4-stream sum rounds to T at each add,
sigmoid in MLX's 1 / (1 + exp(|x|)) form, the f32 sum of squares in MLX's 640-thread tree): GPU micro bit-equal vs stock at rows 1 and 4, half and f32 inject. Only hidden_size 2560. Default off; supersedes HC_COMPILE for qualifying calls (rows <= MAX_ROWS), else the stock /
compiled call. Read at load; ACTIVE is a runtime toggle (A/B cells); STATS counts fused vs fallback calls."""
import os
import mlx.core as mx

ENABLED = os.environ.get("KYOJIN_HC_FUSE", "0") not in ("0", "")
ACTIVE = ENABLED
MAX_ROWS = int(os.environ.get("KYOJIN_HC_FUSE_ROWS", "16"))
INJ_ENABLED = os.environ.get("KYOJIN_HC_INJ", "0") not in ("0", "")
INJ_ACTIVE = INJ_ENABLED
STATS = {"fused": 0, "fallback": 0, "inj": 0}

_NORM = r"""
    uint row = threadgroup_position_in_grid.y;
    uint g = threadgroup_position_in_grid.x;            // stream
    uint t = thread_position_in_threadgroup.x;
    uint l = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const device T* xp = x + (row * HC + g) * H + 4 * t;
    // MLX's row reduce of one 2560-long f32 row (found by micro, bit-equal): 640 threads x 4 contiguous elements added in order, simd_sum, then simd_sum of the 20 partials
    float v0 = (float)xp[0], v1 = (float)xp[1], v2 = (float)xp[2], v3 = (float)xp[3];
    float ss = (0.0f + v0 * v0) + v1 * v1;
    ss = (ss + v2 * v2) + v3 * v3;
    ss = simd_sum(ss);
    threadgroup float sh[32];
    if (l == 0) sh[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = simd_sum(l < NSG ? sh[l] : 0.0f);
    float r = metal::precise::rsqrt(tot * (1.0f / (float)H) + EPS);
    for (uint i = 0; i < 4; i++) {
        uint c = 4 * t + i;
        float v = (float)xp[i] * r;
        out[(row * HC + g) * H + c] = (T)(v * (1.0f + (float)w[g * H + c]));
    }
"""
_ACT = r"""
    uint i = thread_position_in_grid.x;
    if (i < NA) {
        float p = (float)(T)((float)a[i] / (float)HC);
        float se = 1.0f / (1.0f + metal::exp(metal::abs(p)));
        float s = (p < 0.0f) ? se : 1.0f - se;
        oa[i] = (T)(p * (float)(T)s);
    } else if (i < NA + NB) {
        uint j = i - NA;
        float p = (float)(TB)((float)b[j] / (float)HC);
        float se = 1.0f / (1.0f + metal::exp(metal::abs(p)));
        float s = (p < 0.0f) ? se : 1.0f - se;
        ob[j] = (TB)(2.0f * (float)(TB)s);
    }
"""
_MIX = r"""
    uint row = threadgroup_position_in_grid.y;
    uint j = thread_position_in_grid.x;
    if (j < H) {
        float acc = 0.0f;
        for (uint s = 0; s < HC; s++) {
            float u = (float)up[row * HC * H + s * H + j];
            float se = 1.0f / (1.0f + metal::exp(metal::abs(u)));
            float sg = (float)(T)((u < 0.0f) ? se : 1.0f - se);
            float pr = (float)(T)(sg * (float)nm[row * HC * H + s * H + j]);
            acc = (float)(T)(acc + pr);
        }
        out[row * H + j] = (T)(acc / (float)HC);
    }
"""
_INJ = r"""
    // block_inject_weight (HC x K dense) + 2 * sigmoid(. / HC): one threadgroup per (output column, row), 256 threads strided over K, f32 accumulate, simd_sum tree
    uint c = threadgroup_position_in_grid.x;
    uint row = threadgroup_position_in_grid.y;
    uint t = thread_position_in_threadgroup.x;
    uint l = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const device T* xp = x + row * K;
    const device TW* wp = w + c * K;
    float acc = 0.0f;
    for (uint k = 4 * t; k < K; k += 1024) {
        acc += (float)xp[k] * (float)wp[k] + (float)xp[k + 1] * (float)wp[k + 1] + (float)xp[k + 2] * (float)wp[k + 2] + (float)xp[k + 3] * (float)wp[k + 3];
    }
    acc = simd_sum(acc);
    threadgroup float sh[8];
    if (l == 0) sh[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        float tot = 0.0f;
        for (uint i = 0; i < 8; i++) tot += sh[i];
        float p = (float)(TW)(tot / (float)HC);
        float se = 1.0f / (1.0f + metal::exp(metal::abs(p)));
        float s = (p < 0.0f) ? se : 1.0f - se;
        out[row * HC + c] = (TW)(2.0f * (float)(TW)s);
    }
"""
_K = {}


def _k(name, src, ins, outs):
    if name not in _K:
        _K[name] = mx.fast.metal_kernel(name=name, input_names=ins, output_names=outs, source=src)
    return _K[name]


def is_hc(m) -> bool:
    return all(hasattr(m, a) for a in ("hc_norm", "input_mix_weight_down", "input_mix_weight_up", "hc_count", "hidden_size")) and not hasattr(m, "layers")


def _fused(m, x):
    HC, H = m.hc_count, m.hidden_size
    shp = x.shape
    rows = x.size // (HC * H)
    T = x.dtype
    xf = x.reshape(rows, HC * H)
    eps = float(m.hc_norm.eps)
    normed = _k(f"hcf_norm_{eps!r}".replace(".", "_").replace("-", "m"), _NORM.replace("EPS", f"{eps!r}f"), ["x", "w"], ["out"])(
        inputs=[xf, m.hc_norm.weight], template=[("T", T), ("H", H), ("HC", HC), ("NSG", H // 128)],
        grid=(HC * (H // 4), rows, 1), threadgroup=(H // 4, 1, 1), output_shapes=[(rows, HC * H)], output_dtypes=[T])[0]
    nb = x.reshape(*shp[:-1], HC * H)
    normed3 = normed.reshape(*shp[:-1], HC * H)
    has_inj = "block_inject_weight" in m
    down = m.input_mix_weight_down(normed3)
    inj = None
    inj_gate = None
    if has_inj:
        W = m.block_inject_weight.weight
        if INJ_ACTIVE and H * HC % 1024 == 0 and W.ndim == 2 and W.shape == (HC, HC * H) and rows <= MAX_ROWS:
            inj_gate = _k("hcf_inj", _INJ, ["x", "w"], ["out"])(
                inputs=[normed, W], template=[("T", T), ("TW", W.dtype), ("K", HC * H), ("HC", HC)],
                grid=(HC * 256, rows, 1), threadgroup=(256, 1, 1), output_shapes=[(rows, HC)], output_dtypes=[W.dtype])[0]
            STATS["inj"] += 1
        else:
            inj = m.block_inject_weight(normed3)
    NA = down.size
    NB = inj.size if inj is not None else 0
    oa, ob = _k("hcf_act", _ACT, ["a", "b"], ["oa", "ob"])(
        inputs=[down.reshape(-1), (inj if inj is not None else down).reshape(-1)],
        template=[("T", T), ("TB", inj.dtype if inj is not None else T), ("HC", HC), ("NA", NA), ("NB", NB)], grid=(((NA + NB + 255) // 256) * 256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(NA,), (max(NB, 1),)], output_dtypes=[T, inj.dtype if inj is not None else T])
    up = m.input_mix_weight_up(oa.reshape(down.shape))
    mixed = _k("hcf_mix", _MIX, ["up", "nm"], ["out"])(
        inputs=[up.reshape(rows, HC * H), normed], template=[("T", T), ("H", H), ("HC", HC)],
        grid=(((H + 255) // 256) * 256, rows, 1), threadgroup=(256, 1, 1), output_shapes=[(rows, H)], output_dtypes=[T])[0].reshape(*shp[:-1], H)
    if not has_inj:
        return mixed
    if inj_gate is not None:
        return mixed, x, inj_gate.reshape(*shp[:-1], HC)
    return mixed, x, ob.reshape(inj.shape)


def _wrap(mod):
    cls = mod.__class__
    prev = cls.__call__

    def call(self, x):
        rows = x.size // (self.hc_count * self.hidden_size)
        if ACTIVE and rows <= MAX_ROWS and x.dtype in (mx.float16, mx.bfloat16) and x.shape[-1] == self.hc_count * self.hidden_size and self.hidden_size == 2560:
            STATS["fused"] += 1
            return _fused(self, x)
        STATS["fallback"] += 1
        return prev(self, x)

    mod.__class__ = type("Hcf" + cls.__name__, (cls,), {"__call__": call, "_hc_fused": True})


def install(model, log=print, force=False) -> int:
    global ACTIVE
    if not (ENABLED or force):
        return 0
    ACTIVE = True
    n = 0
    for _, m in model.named_modules():
        if is_hc(m) and not getattr(m.__class__, "_hc_fused", False):
            _wrap(m)
            n += 1
    if n:
        log(f"[foreign] hc_fuse: {n} hyper-connection modules fused (rows <= {MAX_ROWS})")
    return n
