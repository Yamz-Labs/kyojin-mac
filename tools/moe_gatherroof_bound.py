#!/usr/bin/env python3
"""Bound of the routed gate/up (and down) gather GEMV at rows 1 / 2: real decode vs decode stub (same words read) vs ALU-only (no trellis loads).
Dependent chain of NL launches per variant (each takes the previous partial as `dep`), min of N runs. Run under the GPU lock."""
import sys, time, os, re
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from kyojin_mlx import moe as M, metal_src as S
from test_moe import make_stack

KD, NI, TOPK, NL, NS, NE = 5120, int(os.environ.get("NI", "2304")), 6, 40, 6, 16
ks = [2, 3, 3, 2, 2, 3, 2, 2, 3, 2, 3, 3, 2, 3, 2, 3]
STUB = r"""
inline half dec_stub(uint s) { return as_type<half>(ushort(s)); }
inline half dec_nofma(uint s) { uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu); return as_type<half2>(t * 0x00010001u + 0x64000000u).y; }
inline half dec_mul(uint s) { uint h = s * 0x83DCD12Du; return as_type<half2>(h).y; }
inline half dec_nosum(uint s) { uint h = s * 0x83DCD12Du; uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu); return as_type<half2>(t).y; }
"""
rng = np.random.default_rng(0)
POOL = [i for i, k in enumerate(ks) if "KONLY" not in os.environ or k == int(os.environ["KONLY"])]
if len(POOL) < TOPK: POOL = POOL * 2
what = sys.argv[1] if len(sys.argv) > 1 else "gu"
VARS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["real", "stub", "alu"]


def build(stk, variant, nproj, cb, pf=os.environ.get("PF", "1") == "1"):
    base = variant.replace("_alu", "").replace("t128", "").strip("_") or "real"
    src = S.gather_gemv_source((2, 3), nproj, base if base in ("stub", "nofma", "mul", "nosum", "mul1r2", "mul1r1", "mul1") else cb, pf)
    if variant.endswith("alu"):                                           # no trellis load: words are made from the loop counter
        TI = "uint(tp - tbase)"
        src = re.sub(r"\(\(const device uint4\*\) tp\)\[(\d+) \* g \+ (\d+)\]", lambda m: f"uint4({TI} * 0x9E3779B1u + g * {int(m.group(1))*7+int(m.group(2))}u + 1u, {TI} * 0x85EBCA6Bu + g, {TI} * 0xC2B2AE35u + {m.group(2)}u, {TI} * 0x27D4EB2Fu + g)", src)
        src = re.sub(r"\(\(const device uint2\*\) tp\)\[(\d+) \* g \+ (\d+)\]", lambda m: f"uint2({TI} * 0x9E3779B1u + g + {m.group(2)}u, {TI} * 0x85EBCA6Bu + g)", src)
        src = re.sub(r"tp\[wprev\]", f"({TI} * 0x165667B1u + g)", src)
        src = re.sub(r"tp\[(\d+) \* g \+ (\d+)\]", lambda m: f"({TI} * 0x9E3779B1u + g + {m.group(2)}u)", src)
    if variant.startswith("t128"):
        for a, b in (("const uint blk = threadgroup_position_in_grid.x; ", "const uint blk = threadgroup_position_in_grid.x >> 1; "),
                     ("const uint half_ = warp & 1u, split = warp >> 1;", "const uint half_ = threadgroup_position_in_grid.x & 1u, split = warp;"),
                     ("for (uint idx = tid; idx < 128; idx += 256) {\n        const uint c = idx;", "for (uint idx = tid; idx < 64; idx += 128) {\n        const uint c = half_ * 64 + idx;")):
            assert a in src or a.strip() in src, a
            src = src.replace(a.strip() if a.strip() in src else a, b.strip() if a.strip() in src else b)
    src = src.replace("const uint KT = params[0]", "if (dep[0] == 1234567.0f) return;\n    const uint KT = params[0]", 1)
    names = ["xh", "ids", "tr0"] + (["tr1"] if nproj == 2 else []) + ["off0"] + (["off1"] if nproj == 2 else []) + ["kk0"] + (["kk1"] if nproj == 2 else []) + ["params", "dep"]
    hdr = S.HEADER_MOE + S.HEADER_MOE_ROOF + STUB
    return mx.fast.metal_kernel(name=f"gr_{variant}_{nproj}_{len(src)%9973}", input_names=names, output_names=["part"], source=src, header=hdr)


def main():
    fam = {"gu": (KD, NI, 2), "dn": (NI, KD, 1)}[what]
    kin, nout, nproj = fam
    stacks = []
    for s in range(NS):
        a, _ = make_stack(kin, nout, ks, "mul1", 10 * s + 1)
        b = make_stack(kin, nout, ks, "mul1", 10 * s + 2)[0] if nproj == 2 else None
        stacks.append((a, b))
    mx.eval([x for a, b in stacks for st in (a, b) if st is not None for x in (st.bound, st.offset, st.k)])
    g0, _ = make_stack(KD, NI, ks, "mul1", 1); u0, _ = make_stack(KD, NI, ks, "mul1", 2); d0, _ = make_stack(NI, KD, ks, "mul1", 3)
    m0 = M.MoeGather(g0, u0, d0, TOPK, wpre=True, fq=True)
    cb = m0.cbf
    print("cb", cb, "bytes/pair-proj avg", np.mean([int(x) for x in np.array(stacks[0][0].offset)[1:] - np.array(stacks[0][0].offset)[:-1]]) * 4 / 1e6, "MB")
    for rows in ([float(os.environ["ROWS"])] if "ROWS" in os.environ else (1, 2)):
        npairs = int(rows * TOPK)
        tgk = M.pick_tgk(stacks[0][0].kt, stacks[0][0].blocks, nproj * npairs)
        tgk = int(os.environ.get("TGK", tgk))
        grid = (stacks[0][0].blocks * 256, tgk, nproj * npairs)
        ids_l = [mx.array(np.concatenate([rng.choice(POOL, TOPK, replace=False) for _ in range(int(rows + 0.99))]).astype(np.int32)) for _ in range(NL)]
        xh = mx.array(rng.standard_normal((nproj * npairs, kin)).astype(np.float32))
        params = mx.array([stacks[0][0].kt, stacks[0][0].nt, tgk, npairs], dtype=mx.uint32)
        dep0 = mx.zeros((1,), mx.float32)
        mx.eval(xh, params, dep0, ids_l)
        sb = []
        for a, b in stacks:
            sb.append((a, b))
        # bytes actually read in the chain
        tot = 0
        for ids in ids_l:
            for e in np.array(ids):
                for st in (stacks[0][0],) + ((stacks[0][1],) if nproj == 2 else ()):
                    tot += int(np.array(st.offset)[e + 1] - np.array(st.offset)[e]) * 4
        mb = tot / NL / 1e6
        res = {}
        for variant in VARS:
            kern = build(stacks[0][0], variant, nproj, cb)
            def run():
                dep = dep0
                for l in range(NL):
                    a, b = sb[l % NS]
                    ins = [xh, ids_l[l], a.bound] + ([b.bound] if nproj == 2 else []) + [a.offset] + ([b.offset] if nproj == 2 else []) + [a.k] + ([b.k] if nproj == 2 else []) + [params, dep]
                    dep = kern(inputs=ins, grid=grid, threadgroup=(128 if variant.startswith("t128") else 256, 1, 1), output_shapes=[(nproj * npairs, tgk, nout)], output_dtypes=[mx.float32])[0]
                return dep
            mx.eval(run())
            r = []
            for _ in range(30):
                t0 = time.perf_counter(); mx.eval(run()); r.append(time.perf_counter() - t0)
            res[variant] = sorted(r)[0] * 1e6 / NL
        print(f"{what} rows {rows}: tg {grid[0]//256}x{grid[1]}x{grid[2]} = {grid[0]//256*grid[1]*grid[2]}  bytes/layer {mb:.1f} MB (roof {mb/0.925:.1f} us @925GB/s)  "
              + "  ".join(f"{k} {v:.1f} us" for k, v in res.items()))


main()
