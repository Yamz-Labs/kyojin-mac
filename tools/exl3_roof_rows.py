#!/usr/bin/env python3
"""R-row (verify) EXL3 linear vs the read roof, dependent chain up(2560->9216) -> down(9216->2560), distinct weights per call, ABAB in one process.
usage: exl3_roof_rows.py [--rows 5] [--variants lin,nax,nodec,nomm] [--pool 12]"""
import argparse, statistics, time
import numpy as np
import mlx.core as mx
from kyojin_mlx import kernels as KN, metal_src as S, exl3_linear as EL, metal_nax as NX
from exl3_roof import synth

NX.HEADER_NAX += "\ninline half dec_null(uint s) { return as_type<half>(ushort(s)); }\n"
NX.HEADER_NAX += """
inline half dec_d1(uint s) { uint h = s * 0x83DCD12Du; return as_type<half>(ushort(h >> 8)); }
inline half dec_d2(uint s) { uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu); return as_type<half>(ushort(ushort(t) + ushort(t >> 16) + ushort(0x6400))); }
inline half dec_d3(uint s) { uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu); uint u = (t & 0xffffu) + (t >> 16); return fma(as_type<half>(ushort(u + 0x6400u)), as_type<half>(ushort(0x1eee)), as_type<half>(ushort(0xc931))); }
inline half dec_d4(uint s) { uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu); uint u = (t * 0x10001u) >> 16; return fma(as_type<half>(ushort(u + 0x6400u)), as_type<half>(ushort(0x1eee)), as_type<half>(ushort(0xc931))); }
inline half dec_e1(uint s) { uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu); uint u = t + (t >> 16) + 0x6400u; return fma(as_type<half>(ushort(u)), as_type<half>(ushort(0x1eee)), as_type<half>(ushort(0xc931))); }
inline half dec_e2(uint s) { uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h & 0xff00ff00u) >> 8); uint u = t + (t >> 16) + 0x6400u; return fma(as_type<half>(ushort(u)), as_type<half>(ushort(0x1eee)), as_type<half>(ushort(0xc931))); }
inline half dec_e3(uint s) { uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu); ushort u = ushort(t) + ushort(t >> 16) + ushort(0x6400); return fma(as_type<half>(u), as_type<half>(ushort(0x1eee)), as_type<half>(ushort(0xc931))); }
inline half dec_d5(uint s) { uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu); uint u = (t * 0x10001u) >> 16; return half(float(u) * 0.0010004043579101562f - 9.6953125f + 1.0f); }
"""
_orig_coop = NX.coop_source
def coop_two(*args, **kw):
    src = _orig_coop(*args, **kw)
    ra = "mpp::tensor_ops::__tensor_ops_detail::__remove_addrspace_t"
    decl = f"  auto cT0b = op.get_destination_cooperative_tensor<{ra}<decltype(tA0)>, {ra}<decltype(rB)>, float>();\n  for (uint16_t i = 0; i < cT0b.get_capacity(); ++i) if (cT0b.is_valid_element(i)) cT0b[i] = 0.0f;\n"
    assert "const ushort i0 = 0;" in src
    src = src.replace("const ushort i0 = 0;", decl + "const ushort i0 = 0;")
    assert src.count("op.run(sA, rB, cT0);") == 1
    src = src.replace("op.run(sA, rB, cT0);", "if ((kt >> 2) & 1u) op.run(sA, rB, cT0b); else op.run(sA, rB, cT0);")
    merge = "  for (uint16_t i = 0; i < cT0.get_capacity(); ++i) if (cT0.is_valid_element(i)) cT0[i] += cT0b[i];\n"
    assert "  threadgroup half fb[" in src
    src = src.replace("  threadgroup half fb[", merge + "  threadgroup half fb[", 1)
    return src


def coop_dsum(*args, **kw):
    src = _orig_coop(*args, **kw)
    assert src.count("op.run(sA, rB, cT0);") == 1
    return src.replace("op.run(sA, rB, cT0);", "{ float sx = 0.0f; for (uint16_t i = 0; i < rB.get_capacity(); ++i) sx += float(rB[i]); cT0[0] += sx; }")


BODY_A = "const uint G = uint(ix0[0]) >> 2, nb = uint(ix0[1]);\n"


def coop_mix(*args, **kw):
    """Co-residency probe: simdgroups 0-1 of every threadgroup run the decode-only body (dsum), simdgroups 2-3 the matmul-only body (nodec), in ONE launch."""
    a = list(args)
    a[7] = "dsum"
    sd = coop_dsum(*a, **kw)
    a[7] = "nodec"
    sn = _orig_coop(*a, **kw)
    def body(src):
        i = src.index(BODY_A) + len(BODY_A)
        j = src.index("\n  threadgroup half fb[") if "\n  threadgroup half fb[" in src else src.index("\n  for (uint16_t i = 0; i < cT0.get_capacity(); ++i) if (cT0.is_valid_element(i)) { const auto ix")
        return i, j
    i, j = body(sd)
    ni, nj = body(sn)
    return sd[:i] + "if ((sg & 2u) == 0u) {\n" + sd[i:j] + "\n} else {\n" + sn[ni:nj] + "\n}" + sd[j:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=5); ap.add_argument("--pool", type=int, default=12); ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--chain", type=int, default=12); ap.add_argument("--variants", default="lin,nax,nodec,nomm"); ap.add_argument("--cfg", default=""); ap.add_argument("--side", default="chain")
    a = ap.parse_args()
    mx.set_cache_limit(256 << 20)
    R = a.rows
    ups = [synth(2560, 9216, 4, 10 + i) for i in range(a.pool)]; dns = [synth(9216, 2560, 4, 100 + i) for i in range(a.pool)]
    mx.eval(*[l._words for l in ups + dns])
    x0 = mx.array(np.random.default_rng(5).standard_normal((R, 2560)).astype(np.float16)); mx.eval(x0)
    cfg = tuple(int(v) for v in a.cfg.split(",")) if a.cfg else None
    def mk(name):
        if name == "lin": return lambda l, x: l(x)
        if name == "mma":
            def f(l, x):
                EL.MMA_ROWS, EL.ROWS_NAX = 5, False
                try: return l(x)
                finally: EL.MMA_ROWS, EL.ROWS_NAX = 0, True
            return f
        if name == "rg":
            def f(l, x):
                EL.ROW_SPLIT_MAX, EL.ROWS_NAX = 32, False
                try: return l(x)
                finally: EL.ROW_SPLIT_MAX, EL.ROWS_NAX = 0, True
            return f
        if name.startswith("ks:"):       # ks:KS[,bmu,nmb,bns,bks]  k-split NAX partials (fp32) + GEMV finalize
            parts = name[3:].split(","); KS = int(parts[0]); cc = tuple(int(v) for v in parts[1:5]) if len(parts) > 1 else NX.pick_cfg([R])
            def f(l, x):
                if not hasattr(l, "_stk"): l._stk = {}
                if l.cb not in l._stk: l._stk[l.cb] = (NX.SingleStack(l._words, l.K, l._kt, l._nt, l.cb), mx.array([0], dtype=mx.int32))
                st, eids = l._stk[l.cb]
                xh = KN.rot(x, l._suh, mx.float16, perm=True)
                NX.coop_source = _orig_coop
                part = NX.expert_gemm((st,), eids, mx.array([0, R], dtype=mx.uint32), xh, R, R, cc, ksplit=KS).reshape(KS, R, l.out_features)
                if not hasattr(l, "_fp_ks"): l._fp_ks = {}
                if KS not in l._fp_ks: l._fp_ks[KS] = mx.array([KS, l.out_features, R], dtype=mx.uint32)
                grid, tgs = KN.S.had_grid(l.out_features // 128, R)
                return KN.fin_kernel(l.out_dtype, mx.float32)(inputs=[part, l._svh, l._fp_ks[KS]], template=[("OT", l.out_dtype)], grid=grid, threadgroup=tgs,
                                                               output_shapes=[(R, l.out_features)], output_dtypes=[l.out_dtype])[0]
            return f
        if name.startswith("cfg:"):      # cfg:bmu,nmb,bns,bks[,mode]  (no fused finalize when bns != 32)
            parts = name[4:].split("/"); cc = tuple(int(v) for v in parts[0].split(",")); md = parts[1] if len(parts) > 1 else "full"
            def f(l, x):
                if not hasattr(l, "_stk"): l._stk = {}
                if l.cb not in l._stk: l._stk[l.cb] = (NX.SingleStack(l._words, l.K, l._kt, l._nt, l.cb), mx.array([0], dtype=mx.int32))
                st, eids = l._stk[l.cb]
                xh = KN.rot(x, l._suh, mx.float16, perm=True)
                NX.coop_source = _orig_coop
                fin = (l._svh, l.out_dtype) if cc[2] == 32 else None
                y = NX.expert_gemm((st,), eids, mx.array([0, R], dtype=mx.uint32), xh, R, R, cc, mode=md, fin=fin)
                return y
            return f
        mode = {"nax": "full", "nodec": "nodec", "nomm": "nomm", "two": "two", "mix": "mix", "dsum": "dsum", "dsumnull": "dsum", "dsumd1": "dsum", "dsumd2": "dsum", "dsumd3": "dsum", "dsumd4": "dsum", "dsume3": "dsum", "dsume2": "dsum", "dsume1": "dsum", "nullmm": "full", "nopf": "nopf", "pfw": "pfw", "pf": "pf", "nf": "full", "nfnodec": "nodec"}[name]
        def f(l, x):
            cbn = "null" if name in ("dsumnull", "nullmm") else (name[4:] if name.startswith("dsumd") else l.cb)
            if not hasattr(l, "_stk"): l._stk = {}
            if cbn not in l._stk:
                l._stk[cbn] = (NX.SingleStack(l._words, l.K, l._kt, l._nt, cbn), mx.array([0], dtype=mx.int32))
            st, eids = l._stk[cbn]
            c = cfg or NX.pick_cfg([R])
            xh = KN.rot(x, l._suh, mx.float16, perm=True)
            NX.coop_source = coop_two if name.startswith("two") else (coop_mix if name == "mix" else (coop_dsum if name.startswith("dsum") else _orig_coop))
            if name.startswith("nf"):      # no fused finalize: expert_gemm alone with the cfg given (bns 16 / 64 allowed)
                return NX.expert_gemm((st,), eids, mx.array([0, R], dtype=mx.uint32), xh, R, R, c, mode=mode)[:, : l.out_features].reshape(R, l.out_features)
            return NX.expert_gemm((st,), eids, mx.array([0, R], dtype=mx.uint32), xh, R, R, c, mode=mode, fin=(l._svh, l.out_dtype))
        return f
    names = a.variants.split("|") if "|" in a.variants else a.variants.split(",")
    l0 = ups[0]
    ref = {n: mk(n)(l0, x0) for n in names if n in ("nopf", "pf", "nax") or n.startswith("ks:")}
    mx.eval(*ref.values())
    r0 = ref.get("nax", ref.get("nopf"))
    for n, v in ref.items():
        if r0 is not None:
            d = (v.astype(mx.float32) - r0.astype(mx.float32)); print("check", n, "equal:", bool(mx.array_equal(v, r0).item()), "rel", float(mx.linalg.norm(d) / mx.linalg.norm(r0.astype(mx.float32))))
    fs = {n: mk(n) for n in names}
    xd = mx.array(np.random.default_rng(6).standard_normal((R, 9216)).astype(np.float16)); mx.eval(xd)
    def once(f, n):
        if a.side == "up": return [f(ups[i % a.pool], x0) for i in range(n)]
        if a.side == "down": return [f(dns[i % a.pool], xd) for i in range(n)]
        x = x0
        for i in range(n): x = f(dns[i % a.pool], f(ups[i % a.pool], x))
        return x
    for n in names: mx.eval(once(fs[n], 2))
    mx.eval([once(fs[n], 1) for n in names])
    T = {n: [] for n in names}
    for rep in range(a.reps):
        for n in names:
            t0 = time.perf_counter(); mx.eval(once(fs[n], a.chain)); T[n].append((time.perf_counter() - t0) / a.chain * 1e6 / (1 if a.side != 'chain' else 2))
    print(f"R={R} cfg={cfg or NX.pick_cfg([R])}")
    for n in names: print(f"  {n:6s} min {min(T[n]):7.1f} med {statistics.median(T[n]):7.1f} us per linear", flush=True)
if __name__ == "__main__":
    main()
