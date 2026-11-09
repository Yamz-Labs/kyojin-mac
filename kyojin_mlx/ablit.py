# SPDX-License-Identifier: Apache-2.0
"""Runtime refusal-direction projection (decensor hook), no weight edit and no re-quantisation. KYOJIN_ABLIT_SPEC=<edit_spec.json>, default off.

Editing a writer as W' = (I - w r r^T) W is linear in the writer output, so it equals projecting the sublayer output before the residual write:

    y <- y - w_L * R^T (R y)        attention output (after wo_b) with w = attn_w[L],  MoE output (routed + shared) with w = mlp_w[L]

R is [K, hidden] with orthonormal rows (K = 1 for one direction r). The projection sits right before `hc_expand` (the mHC post / comb mix) of the trunk
block L, at every place that block is computed: the stock `DeepseekV41Block.__call__` (replaced here, see `install_block`), the compiled decode / chain regions
(fuse_layer), the prompt block (pf_glue) and the CED midpoint block (ced). Blocks without `_ablit` (the MTP / dspark draft head, layers >= spec n_layers,
layers whose weights are 0) are never touched, so verify keeps the output of the edited main model while drafts may differ. Spec switch off = the helpers
return their input array unchanged (bit-identical).

Spec: JSON {hidden, n_layers, attn_w[L], mlp_w[L], optional r_file} (the layout of the box tool, EXL3_ABLIT_RUNTIME) and next to it <spec>.safetensors with tensor
"r" [hidden] (one direction for every layer), "R" [K, hidden] (one subspace) or "R" [n_layers, K, hidden] (per layer). Order: call `install_block()` before
fuse.install / rows.enable (patch_model does when the env is set), then `attach(model)` once the weights are loaded."""
from __future__ import annotations

from . import archs as _A
import json
import os
from types import SimpleNamespace

import mlx.core as mx

from . import ablate as AZ

ENV = "KYOJIN_ABLIT_SPEC"
_OFF = ("", "off", "0", "false", "no", "none")


def spec_path() -> str | None:
    p = os.environ.get(ENV, "").strip()
    return None if p.lower() in _OFF else p


def load(path: str) -> SimpleNamespace:
    """-> (n_layers, hidden, attn_w, mlp_w, R) with R fp32 [K, hidden] or [n_layers, K, hidden], rows normalised."""
    from safetensors.numpy import load_file
    import numpy as np
    if not os.path.isfile(path):
        raise FileNotFoundError(f"ablit: spec not found: {path}")
    spec = json.load(open(path))
    miss = [k for k in ("hidden", "n_layers", "attn_w", "mlp_w") if k not in spec]
    if miss:
        raise ValueError(f"ablit: spec {path} misses {miss}")
    d = os.path.dirname(os.path.abspath(path))
    rf = spec.get("r_file")
    rf = (rf if os.path.isabs(rf) else os.path.join(d, rf)) if rf else os.path.splitext(path)[0] + ".safetensors"
    t = load_file(rf)
    R = t["R"] if "R" in t else t["r"]
    R = R.astype(np.float64)
    R = R / np.maximum(np.linalg.norm(R, axis=-1, keepdims=True), 1e-12)
    if R.shape[-1] != spec["hidden"] or R.ndim not in (1, 2, 3) or (R.ndim == 3 and R.shape[0] < spec["n_layers"]):
        raise ValueError(f"ablit: direction shape {R.shape} does not fit the spec (hidden {spec['hidden']}, layers {spec['n_layers']})")
    if R.ndim == 1:
        R = R[None]
    if len(spec["attn_w"]) < spec["n_layers"] or len(spec["mlp_w"]) < spec["n_layers"]:
        raise ValueError("ablit: attn_w / mlp_w shorter than n_layers")
    return SimpleNamespace(n_layers=int(spec["n_layers"]), hidden=int(spec["hidden"]), attn_w=[float(x) for x in spec["attn_w"]],
                           mlp_w=[float(x) for x in spec["mlp_w"]], R=R.astype(np.float32), path=path,
                           attn_mu=spec.get("attn_mu"), mlp_mu=spec.get("mlp_mu"))


def signature(path: str | None = None) -> str | None:
    """Content hash of the active spec (json + direction file), None when off: the disk KV namespace key (an armed hook changes every cached layer
    after the first armed block)."""
    import hashlib
    path = path or spec_path()
    if path is None:
        return None
    d = os.path.dirname(os.path.abspath(path))
    rf = json.load(open(path)).get("r_file")
    rf = (rf if os.path.isabs(rf) else os.path.join(d, rf)) if rf else os.path.splitext(path)[0] + ".safetensors"
    h = hashlib.blake2b(digest_size=12)
    for f in (path, rf):
        h.update(open(f, "rb").read())
    return h.hexdigest()


def _project(y: mx.array, w: mx.array, R: mx.array) -> mx.array:
    y32 = y.astype(mx.float32)
    return (y32 - w * ((y32 @ R.T) @ R)).astype(y.dtype)


def _project_affine(y: mx.array, w: mx.array, R: mx.array, mu: mx.array) -> mx.array:
    y32 = y.astype(mx.float32)
    return (y32 - w * (((y32 @ R.T) - mu) @ R)).astype(y.dtype)


_project_c = mx.compile(_project, shapeless=True)
_project_affine_c = mx.compile(_project_affine, shapeless=True)

# Flag ablit_fused (KYOJIN_ABLIT_FUSED, runtime): the whole projection of a row in ONE launch. The stock path is ~7 launches per call
# (fp16 clip = maximum + minimum, the fp32 cast the matmul reads, R y, (R y - mu) R, the fused update), and a decensored DS4.1 step makes
# ~58 calls: ~350 of its ~2080 dispatches. Same math in fp32 and the same association as _project_affine; only the summation order of
# the K dot products differs (one threadgroup per row: per-thread partial sums, simd_sum, then the simdgroup partials in order).
FUSED = os.environ.get("KYOJIN_ABLIT_FUSED", "1") not in ("0", "")
_TG = 256                                                                      # threads per row: 20 columns each at hidden 5120
_SRC = """
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint tg = threads_per_threadgroup.x;
    const ulong base = ulong(row) * H;
    float s[K];
    for (int k = 0; k < K; ++k) { s[k] = 0.0f; }
    for (uint h = tid; h < uint(H); h += tg) {
        float v = static_cast<float>(y[base + h]);
        if (is_same<T, half>::value) { v = clamp(v, -65504.0f, 65504.0f); }   // an inf column would give inf - inf = NaN
        for (int k = 0; k < K; ++k) { s[k] += v * R[ulong(k) * H + h]; }
    }
    threadgroup float part[K * 32];
    for (int k = 0; k < K; ++k) {
        const float t = simd_sum(s[k]);
        if (thread_index_in_simdgroup == 0) { part[k * 32 + simdgroup_index_in_threadgroup] = t; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nsg = (tg + 31) / 32;
    float c[K];
    for (int k = 0; k < K; ++k) {
        float t = 0.0f;
        for (uint i = 0; i < nsg; ++i) { t += part[k * 32 + i]; }
        c[k] = t - mu[k];
    }
    const float wv = w[0];
    for (uint h = tid; h < uint(H); h += tg) {
        float v = static_cast<float>(y[base + h]);
        if (is_same<T, half>::value) { v = clamp(v, -65504.0f, 65504.0f); }
        float p = 0.0f;
        for (int k = 0; k < K; ++k) { p += c[k] * R[ulong(k) * H + h]; }
        out[base + h] = static_cast<T>(v - wv * p);
    }
"""
_kern = None
_consts: dict = {}                                                             # (w,) / ("mu0", K) -> small fp32 arrays, built once


def _const(key, make):
    a = _consts.get(key)
    if a is None:
        a = _consts[key] = make()
    return a


def _project_fused(y: mx.array, w: float, R: mx.array, mu: mx.array | None) -> mx.array:
    global _kern
    if _kern is None:
        _kern = mx.fast.metal_kernel(name="kyojin_ablit_project", input_names=["y", "R", "mu", "w"], output_names=["out"], source=_SRC)
    K, H = R.shape
    if R.dtype != mx.float32:
        R = R.astype(mx.float32)
    if mu is None:
        mu = _const(("mu0", K), lambda: mx.zeros((K,), dtype=mx.float32))
    wa = _const(("w", float(w)), lambda: mx.array([w], dtype=mx.float32))
    rows = y.size // H
    return _kern(inputs=[y, R, mu, wa], template=[("T", y.dtype), ("K", K), ("H", H)], grid=(rows * _TG, 1, 1), threadgroup=(_TG, 1, 1),
                 output_shapes=[y.shape], output_dtypes=[y.dtype])[0]


def project(y: mx.array, w: float, R: mx.array, mu: mx.array | None = None) -> mx.array:
    """y - w * R^T (R y - mu) over the last axis, fp32 inside, y.dtype out. R [K, hidden] fp32. mu [K] (spec fields
    attn_mu / mlp_mu) = mean of R y on harmless text: the writer keeps its harmless level of the direction, which costs
    far less KL than pulling it to zero. Without mu this is the plain ablation."""
    if AZ.ACTIVE and AZ.on("ablit"):                                           # ablate: the decensor projection is the identity (timing probe)
        AZ.hit("ablit")
        return y
    if FUSED and R.ndim == 2 and y.shape[-1] == R.shape[1] and R.shape[0] <= 8 and y.size:
        return _project_fused(y, w, R, mu)
    if y.dtype == mx.float16:                                                  # an inf row (fp16 overflow of the sub-layer) would give inf - inf = NaN
        y = mx.clip(y, -65504, 65504)
    if mu is not None:
        return _project_affine_c(y, mx.array(w, dtype=mx.float32), R, mu)
    return _project_c(y, mx.array(w, dtype=mx.float32), R)


def _trunk(model):
    return getattr(model, _A.active().trunk, model)


def attach(model, path: str | None = None) -> int:
    """Arm the trunk blocks of `model` (language_model.layers) from the spec; returns the number of armed blocks. No spec = 0 (everything stays off)."""
    path = path or spec_path()
    if path is None:
        return 0
    import numpy as np
    s = load(path)
    R_all = mx.array(s.R)
    layers = _trunk(model).layers
    n = 0
    for L, blk in enumerate(layers):
        if L >= s.n_layers:
            break
        wa, wm = s.attn_w[L], s.mlp_w[L]
        wa, wm = (0.0 if abs(wa) <= 1e-9 else wa), (0.0 if abs(wm) <= 1e-9 else wm)
        if wa == 0.0 and wm == 0.0:
            blk.__dict__.pop("_ablit", None)
            continue
        R = R_all[L] if R_all.ndim == 3 else R_all
        mua = mx.array([float(x) for x in np.atleast_1d(s.attn_mu[L])], dtype=mx.float32) if s.attn_mu else None
        mum = mx.array([float(x) for x in np.atleast_1d(s.mlp_mu[L])], dtype=mx.float32) if s.mlp_mu else None
        blk._ablit = SimpleNamespace(L=L, wa=wa, wm=wm, R=R, mua=mua, mum=mum)
        n += 1
    return n


def detach(model) -> None:
    for blk in _trunk(model).layers:
        blk.__dict__.pop("_ablit", None)


_MU_REC: dict | None = None    # mu_job only: writer -> [sum of R y, count]; the hook edits nothing while it is set


def _record(a, kind: str, y: mx.array) -> None:
    import numpy as np
    d = y.astype(mx.float32) @ a.R.T
    mx.eval(d)
    _MU_REC.setdefault(f"{a.L}{kind}", []).append(np.array(d.reshape(-1, d.shape[-1])))


def mu_job(be, messages_path: str, out_path: str, log=print) -> None:
    """Measure mu per writer on harmless text, one forward per conversation, nothing edited. The active spec must arm
    every writer with a non-zero weight. Input: a JSON list of token id lists. Output: {"attn_mu": [...], "mlp_mu": [...]}."""
    import numpy as np
    global _MU_REC
    lm = getattr(be, "lm", None) or be.model.language_model
    conv = json.load(open(messages_path))
    _MU_REC = {}
    try:
        for ids in conv:
            mx.eval(lm(mx.array(ids)[None], cache=lm.make_cache()))
        rec = _MU_REC
    finally:
        _MU_REC = None
    if not rec:
        raise RuntimeError("mu_job: no writer recorded, a compiled path skipped the hooks")
    n = 1 + max(int(k[:-1]) for k in rec)
    out = {"attn_mu": [[0.0]] * n, "mlp_mu": [[0.0]] * n, "stats": {}}
    for k, v in rec.items():
        d = np.concatenate(v, 0)
        # The median, not the mean: a few template tokens have a huge projection and would drag every other token.
        out["attn_mu" if k[-1] == "a" else "mlp_mu"][int(k[:-1])] = np.median(d, axis=0).astype(float).tolist()
        out["stats"][k] = {"mean": d.mean(0).astype(float).tolist(), "sd": d.std(0).astype(float).tolist(),
                           "p05": np.percentile(d, 5, axis=0).astype(float).tolist(), "p95": np.percentile(d, 95, axis=0).astype(float).tolist()}
    json.dump(out, open(out_path, "w"))
    log(f"ablit mu_job: {len(rec)} writers, {len(conv)} conversations -> {out_path}")


def attn_out(blk, y: mx.array) -> mx.array:
    """Attention output of `blk` (after wo_b) -> projected, or `y` itself when the block is not armed."""
    a = blk.__dict__.get("_ablit")
    if a is None or a.wa == 0.0:
        return y
    if _MU_REC is not None:
        _record(a, "a", y)
        return y
    return project(y, a.wa, a.R, a.mua)


def ffn_out(blk, y: mx.array) -> mx.array:
    """MoE output of `blk` (routed + shared) -> projected, or `y` itself."""
    a = blk.__dict__.get("_ablit")
    if a is None or a.wm == 0.0:
        return y
    if _MU_REC is not None:
        _record(a, "m", y)
        return y
    return project(y, a.wm, a.R, a.mum)


_installed = False


def install_block() -> bool:
    """Replace the stock `DeepseekV41Block.__call__` by the same code with the two hooks (identical when the block is not armed). Once per process, before the
    fuse / rows installs wrap it."""
    global _installed
    if _installed:
        return False
    import importlib
    L = _A.lang()

    def __call__(self, h, pre_mix, image_mask, cache):
        residual = h
        attn_pre, attn_post, attn_comb = self.hc_mixes(h, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.hc_pre(h, pre_mix)
        x = self.attn_norm(x)
        x = attn_out(self, self.attn(x, cache))
        x = L.hc_expand(x, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.hc_pre(x, attn_pre)
        x = self.ffn_norm(x)
        x = ffn_out(self, self.ffn(x, image_mask))
        x = L.hc_expand(x, residual, ffn_post, ffn_comb)
        return x, ffn_pre

    _A.cls("block").__call__ = __call__
    _installed = True
    return True
