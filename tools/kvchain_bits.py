# SPDX-License-Identifier: Apache-2.0
"""kvchain / ropefuse / rope_fma: every op of the fused path against the stock op, bit for bit, on real-shaped data. Needs Metal (a Mac).

What is compared (fused = the kernels of dsv4_core under the chosen KYOJIN_ROPE_FMA, stock = the ops the model runs when kvchain / ropefuse are off):

  kv_norm      fused kv-chain kernel cut after the norm        vs  mx.fast.rms_norm(kvr, w, eps)   (= nn.RMSNorm)
  kv_rope      same kernel, norm replaced by a copy, cut after the rope tail     vs  DeepseekV4RoPE(...)(kvr)            (the stock rope op, full head, nope pairs inf)
  kv_fp8       same kernel, norm and rope replaced by copies                      vs  fake_quant_fp8_ue8m0(kvr)           (mlx_vlm's deepseek_v41_fp8_roundtrip)
  kv_chain     the production kernel `dsv4_core.kv_chain` (norm + rope + fp8 + ring scatter)   vs  stock kv_norm, rope, fp8, _snapshot + overlay scatter
  ring_kept    the slots a pass does not write: the base ring bytes (copy), checked inside kv_chain's output
  q_rope       the q rope of the attention kernel's q load (dsv4_core._Q_ROPE, scale 1)       vs  rope(q, pos)
  inv_rope     the inverse rope of the merge epilogue (dsv4_core._MERGE_INV, l = 1)           vs  rope(o, pos, inverse=True)
  attend       dsv4_core.attend with rt / it (ropefuse) vs attend without + stock rope before and inverse rope after

Each op is fed the same bytes on both sides, so a difference is that op's own arithmetic. For every arm (KYOJIN_ROPE_FMA = 0 / 1 / 3 by default) the
table prints the number of differing elements and, for the first differing op in the order above, the first flat index, its multi-index and both values
(fp16 value, hex bits, ulp distance). Exit code 0 when every op is bit-identical in every arm of --require (default: 1 and 3), else 1.

  python tools/kvchain_bits.py [--rows 6] [--layout "5000,5001,5002;130,131;4000"] [--w 128] [--heads 64] [--yarn] [--arms 0,1,3] [--seed 1]

The data are synthetic with the model's shapes, dtypes and ranges (kv rows with a per-row scale and outliers, norm weights near 1, an fp8-rounded ring,
fp16 pool rows, random top-k indices), not the pack: the question is arithmetic, which does not depend on the weights. Peak memory < 300 MB (the script
refuses above 3.5 GiB). Nothing here is timed.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import mlx.core as mx  # noqa: E402

from kyojin_mlx import dsv4_core as C  # noqa: E402

ORDER = ("kv_norm", "kv_rope", "kv_fp8", "kv_chain", "ring_kept", "q_rope", "inv_rope", "attend")
MEM_LIMIT = int(3.5 * 2 ** 30)


# ------------------------------------------------------------------------------------------------------------------------------ comparison
def _bits(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    return a.view({2: np.uint16, 4: np.uint32}[a.dtype.itemsize])


def _ordered(u: np.ndarray, width: int) -> np.ndarray:
    """Monotone integer key of float bit patterns (sign-magnitude -> two's complement order), so a difference of keys is an ulp distance."""
    top = 1 << (width - 1)
    u = u.astype(np.int64)
    return np.where(u & top, -(u & (top - 1)), u)


def compare(name: str, fused: np.ndarray, stock: np.ndarray, index_names: tuple = ()) -> dict:
    """Bit comparison of two arrays of one float dtype. Returns {"op", "n", "bad", "first"} with first = None or the first differing element
    (flat index, multi-index, both values, both bit patterns, ulp distance)."""
    assert fused.shape == stock.shape and fused.dtype == stock.dtype, (name, fused.shape, stock.shape, fused.dtype, stock.dtype)
    fb, sb = _bits(fused), _bits(stock)
    width = 8 * fused.dtype.itemsize
    ne = fb != sb
    both_nan = np.isnan(fused.astype(np.float32)) & np.isnan(stock.astype(np.float32))        # two NaNs with other payloads are not a difference
    ne &= ~both_nan
    bad = int(ne.sum())
    first = None
    if bad:
        flat = int(np.argmax(ne.reshape(-1)))
        idx = tuple(int(i) for i in np.unravel_index(flat, fused.shape))
        a, b = fused.reshape(-1)[flat], stock.reshape(-1)[flat]
        ka, kb = _ordered(fb.reshape(-1)[flat:flat + 1], width)[0], _ordered(sb.reshape(-1)[flat:flat + 1], width)[0]
        first = dict(flat=flat, index=idx, fused=float(a), stock=float(b), fused_bits=int(fb.reshape(-1)[flat]), stock_bits=int(sb.reshape(-1)[flat]),
                     ulp=int(abs(ka - kb)))
    return dict(op=name, n=int(fused.size), bad=bad, first=first)


def describe(r: dict, index_names: tuple = ()) -> str:
    if not r["bad"]:
        return f"{r['op']:10s} identical ({r['n']} elements)"
    f = r["first"]
    nm = ",".join(f"{k}={v}" for k, v in zip(index_names, f["index"])) if index_names else str(f["index"])
    hexw = 4 if f["fused_bits"] < 1 << 16 and f["stock_bits"] < 1 << 16 else 8
    return (f"{r['op']:10s} {r['bad']} / {r['n']} differ ({r['bad'] / r['n']:.2e}); first at flat {f['flat']} [{nm}]: fused {f['fused']!r} (0x{f['fused_bits']:0{hexw}x}) "
            f"stock {f['stock']!r} (0x{f['stock_bits']:0{hexw}x}), {f['ulp']} ulp")


# ------------------------------------------------------------------------------------------------------------------------------ stage kernels
def stage_sources(form: int) -> dict:
    """The kv-chain kernel text cut into isolated ops by exact string surgery on dsv4_core._KV_CHAIN (each anchor must occur exactly once, so a changed
    kernel breaks this loudly instead of comparing the wrong thing). Test-only kernels: the production source is untouched."""
    src = C._KV_CHAIN
    NORM = "    for (uint i = 0; i < 4; ++i) nb[lid * 4 + i] = w[lid * 4 + i] * static_cast<T>(float(xr[lid * 4 + i]) * inv);\n"
    COPY = "    for (uint i = 0; i < 4; ++i) nb[lid * 4 + i] = xr[lid * 4 + i];\n"
    CUT_NORM = "    if (lid >= 64) return;\n"
    OUT_NORM = "    for (uint i = 0; i < 4; ++i) out[o + lid * 4 + i] = nb[lid * 4 + i];\n    return;\n"
    CUT_ROPE = "    float amax = 0.0f;\n"
    OUT_ROPE = "    for (uint i = 0; i < 8; ++i) out[o + lid * 8 + i] = static_cast<T>(v[i]);\n    return;\n" + CUT_ROPE
    TAIL = "    if (lid * 8 >= uint(D - RD)) {"
    for a in (NORM, CUT_NORM, CUT_ROPE, TAIL):
        assert src.count(a) == 1, f"_KV_CHAIN changed: anchor not found once: {a.strip()[:60]}"
    out = {
        "kv_norm": src.replace(CUT_NORM, OUT_NORM),
        "kv_rope": src.replace(NORM, COPY).replace(CUT_ROPE, OUT_ROPE),
        "kv_fp8": src.replace(NORM, COPY).replace(TAIL, "    if (false) {"),
    }
    return {k: C.rope_src(v, form) for k, v in out.items()}


def q_rope_source(form: int) -> str:
    """The q rope of the attention q load as its own kernel: one simdgroup per (row, group of HG heads), scale 1 so qv * scale is exact."""
    body = (
        "    constexpr uint DPL = D / 32;\n"
        "    constexpr uint NHG = H / HG;\n"
        "    const uint tg = threadgroup_position_in_grid.x;\n"
        "    const uint lane = thread_index_in_simdgroup;\n"
        "    const uint u = tg / NHG;\n"
        "    const uint h0 = (tg % NHG) * HG;\n"
        "    const float scale = 1.0f;\n" + C._Q_ROPE +
        "    for (uint hh = 0; hh < HG; ++hh)\n"
        "        for (uint i = 0; i < DPL; ++i) out[(u * H + h0 + hh) * D + lane * DPL + i] = static_cast<T>(qv[hh][i]);\n")
    return C.rope_src(body, form)


def inv_rope_source(form: int) -> str:
    """The inverse rope of the merge epilogue as its own kernel: one 128-thread group per (row, head), acc = the input, l = 1."""
    body = (
        "    constexpr uint DPT = D / 128;\n"
        "    const uint r = threadgroup_position_in_grid.x;\n"
        "    const uint tid = thread_position_in_threadgroup.x;\n"
        "    float acc[DPT];\n"
        "    for (uint i = 0; i < DPT; ++i) acc[i] = float(x[r * D + tid * DPT + i]);\n"
        "    const float l = 1.0f;\n" + C._MERGE_INV)
    return C.rope_src(body, form)


# ------------------------------------------------------------------------------------------------------------------------------ data
def make_data(a, rng):
    from mlx_vlm.models.deepseek_v4.language import DeepseekV4RoPE
    layout = [[int(p) for p in s.split(",")] for s in a.layout.split(";")]
    pos = [p for s in layout for p in s]
    counts = [len(s) for s in layout]
    if a.rows and a.rows != len(pos):
        pos = (pos * (a.rows // len(pos) + 1))[:a.rows]
        counts = [1] * len(pos)
    scaling = {"type": "yarn", "factor": 16, "original_max_position_embeddings": 65536, "beta_fast": 32, "beta_slow": 1} if a.yarn else None
    rope = DeepseekV4RoPE(64, 160000.0 if a.yarn else 10000.0, scaling)
    D, W, H, S, n = 512, a.w, a.heads, len(counts), len(pos)
    scale_row = np.exp(rng.uniform(-1.0, 1.0, (n, 1)))
    kvr = rng.standard_normal((n, D)) * scale_row
    kvr[rng.random((n, D)) < 0.002] *= 30.0                                                 # outliers: the fp8 scale and the norm see them
    data = dict(
        layout=layout, pos=pos, counts=counts, rope=rope, D=D, W=W, H=H, S=S, n=n,
        kvr=mx.array(kvr.astype(np.float16)),
        w=mx.array((1.0 + 0.3 * rng.standard_normal(D)).astype(np.float16)),
        eps=1e-6,
        q=mx.array((rng.standard_normal((n, H, 1, D)) * 0.7).astype(np.float16)),
        o=mx.array((rng.standard_normal((n, H, 1, D)) * 0.3).astype(np.float16)),
    )
    ring = (rng.standard_normal((S, W, D)) * 0.8).astype(np.float16)
    data["ring"] = _fq8(mx.array(ring))
    return data


def _fq8(x):
    from mlx_vlm.models.deepseek_v41.fakequant import fake_quant_fp8_ue8m0
    return fake_quant_fp8_ue8m0(x)


def _np(x) -> np.ndarray:
    mx.eval(x)
    return np.array(x)


# ------------------------------------------------------------------------------------------------------------------------------ one arm
def run_arm(a, d, form: int) -> list[dict]:
    from mlx_vlm.models.deepseek_v41.fakequant import fake_quant_fp8_ue8m0 as fq8
    n, D, W, H = d["n"], d["D"], d["W"], d["H"]
    rope, plan = d["rope"], C.Plan(d["pos"], d["counts"])
    C.ROPE_FMA = form
    pos_arr = plan.pos_arr
    rt, it = plan.attn_rope(rope)
    RD = rt.shape[-1]
    kvr, w, eps = d["kvr"], d["w"], d["eps"]
    res = []

    NAMES = {"kv_norm": ["kvr", "w", "ring", "sess", "wmap", "rt", "eps"], "kv_rope": ["kvr", "w", "ring", "sess", "wmap", "rt", "eps"],
             "kv_fp8": ["kvr", "w", "ring", "sess", "wmap", "rt", "eps"], "q_rope": ["q", "rt"], "inv_rope": ["x", "it"]}

    def kern(name, src, inputs, template, grid, tg, shapes, dtype):
        k = mx.fast.metal_kernel(name=f"kbits_{name}_{form}", input_names=NAMES[name], output_names=["out"], source=src, header=C._CONTRACT_OFF)
        return k(inputs=inputs, template=template, grid=grid, threadgroup=tg, output_shapes=shapes, output_dtypes=[dtype])[0]

    # ---- kv ops: a pass that writes every slot it maps (stage kernels write only mapped slots; compare through wmap)
    wmap = plan.wmap(W)
    wm = np.array(wmap).reshape(n, W)
    rows_b, slots_s = np.nonzero(wm >= 0)
    src_e = wm[rows_b, slots_s]
    srcs = stage_sources(form)
    ring0 = d["ring"]
    tmpl = [("T", mx.float16), ("D", D), ("W", W), ("RD", RD)]
    stock = {
        "kv_norm": mx.fast.rms_norm(kvr, w, eps),
        "kv_rope": rope(kvr[:, None], pos_arr).reshape(n, D),
        "kv_fp8": fq8(kvr),
    }
    for op in ("kv_norm", "kv_rope", "kv_fp8"):
        out = kern(op, srcs[op], [kvr, w, ring0, plan.sess_arr, wmap, rt, C._f32((eps,))], tmpl, (n * W * 128, 1, 1), (128, 1, 1), [(n, W, D)], mx.float16)
        f = _np(out)[rows_b, slots_s]                                                        # [m, D] one row per written (b, slot)
        s = _np(stock[op])[src_e]
        res.append(compare(op, f, s))
    # ---- the production chain vs the stock chain
    fused = C.kv_chain(kvr, w, ring0, plan.sess_arr, wmap, rt, eps)
    kv = fq8(rope(stock["kv_norm"][:, None], pos_arr).reshape(n, D))
    win = C._snapshot(ring0, plan, (W, D), mx.float16, 0.0)
    ov = plan.overlay(W)
    win[ov.rows, ov.slots] = kv if ov.src is None else kv[ov.src]
    fn, sn = _np(fused), _np(win)
    written = wm >= 0
    res.append(compare("kv_chain", fn[written], sn[written]))
    res.append(compare("ring_kept", fn[~written], sn[~written]))
    # ---- q rope / inverse rope
    hg = 4
    out = kern("q_rope", q_rope_source(form), [d["q"].reshape(n, H, D), rt], [("T", mx.float16), ("H", H), ("HG", hg), ("D", D), ("RD", RD)],
               (n * (H // hg) * 32, 1, 1), (32, 1, 1), [(n, H, D)], mx.float16)
    res.append(compare("q_rope", _np(out).reshape(n, H, 1, D), _np(rope(d["q"], pos_arr))))
    out = kern("inv_rope", inv_rope_source(form), [d["o"].reshape(n, H, D), it], [("T", mx.float16), ("H", H), ("D", D), ("RD", RD)],
               (n * H * 128, 1, 1), (128, 1, 1), [(n, H, D)], mx.float16)
    res.append(compare("inv_rope", _np(out).reshape(n, H, 1, D), _np(rope(d["o"], pos_arr, inverse=True))))
    # ---- the attention kernels with rope in them vs without
    K, Cn, maxk = 512, 1024, 512
    rng = np.random.default_rng(7)
    pool = mx.array((rng.standard_normal((d["S"], Cn, D)) * 0.8).astype(np.float16))
    idx = mx.array(rng.integers(-1, Cn, (n, K)).astype(np.int32))
    sink = mx.array(rng.standard_normal(H).astype(np.float32))
    sess = plan.sess_arr
    posq = mx.array(d["pos"], dtype=mx.int32)
    sc = D ** -0.5
    wins = mx.array(sn)
    base = C.attend(rope(d["q"], pos_arr).reshape(n, H, D), wins, pool, idx, posq, sink, sc, maxk, sess)
    ref = rope(base.reshape(n, H, 1, D), pos_arr, inverse=True)
    fus = C.attend(d["q"].reshape(n, H, D), wins, pool, idx, posq, sink, sc, maxk, sess, rt, it)
    res.append(compare("attend", _np(fus).reshape(n, H, 1, D), _np(ref)))
    return res


# ------------------------------------------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="5000,5001,5002;130,131;4000", help="positions of the rows, sessions separated by ';' (rows of a session consecutive)")
    ap.add_argument("--rows", type=int, default=0, help="0 = the layout; else this many one-row sessions cycling the layout positions")
    ap.add_argument("--w", type=int, default=128, help="window ring slots")
    ap.add_argument("--heads", type=int, default=64)
    ap.add_argument("--yarn", action="store_true", help="the yarn-scaled frequencies of the compressing layers (default: the plain ones of the window layers)")
    ap.add_argument("--arms", default="0,1,3", help="KYOJIN_ROPE_FMA values to run (3 = auto, resolved on this device)")
    ap.add_argument("--require", default="1,3", help="arms that must be bit-identical for exit code 0")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if not mx.metal.is_available():
        print("Metal not available: this compares the GPU kernels with the GPU stock ops")
        return 2
    est = 6 * a.heads * 512 * 2 * 64 + 2 * 1024 * 512 * 2 * 8
    if est > MEM_LIMIT:
        print("estimated memory above 3.5 GiB, refusing")
        return 2
    d = make_data(a, np.random.default_rng(a.seed))
    old = C.ROPE_FMA
    rc = 0
    try:
        detected = C.detect_rope_form()
        print(f"detect_rope_form (fp32 mx.fast.rope vs the three numpy forms): {detected}  {C.ROPE_DETECT}")
        for arm in [int(x) for x in a.arms.split(",")]:
            form = C.resolve_rope_fma(arm)
            print(f"\n== KYOJIN_ROPE_FMA={arm}" + (f" (resolved: form {form})" if arm == 3 else ""))
            res = run_arm(a, d, form)
            first = None
            for r in res:
                print("  " + describe(r))
                if r["bad"] and first is None:
                    first = r
            if first:
                print(f"  FIRST DIFFERING OP: {first['op']}")
            else:
                print("  all ops bit-identical")
            if first and str(arm) in a.require.split(","):
                rc = 1
    finally:
        C.ROPE_FMA = old
    return rc


if __name__ == "__main__":
    sys.exit(main())
