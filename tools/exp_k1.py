"""K1 (scores) tile-shape sweep: NSG simdgroups per threadgroup (32 keys each), KT k-chunk. Bit-compare vs the shipped K1. usage: exp_k1.py [T=512 queries]"""
import sys, time, itertools, numpy as np, mlx.core as mx
from kyojin_mlx import attn_prefill as AP, metal_nax as NX
H, D, NK, TC = 64, 512, 640, 512
rng = np.random.default_rng(0)
def rnd(*s, sc=0.5): return mx.array((rng.standard_normal(s) * sc).astype(np.float32)).astype(mx.float16)
qt = rnd(TC, H, D); tab = rnd(6144, D)
ids = mx.array(np.stack([rng.choice(6144, NK, replace=False) for _ in range(TC)]).astype(np.int32))
ids = mx.where(mx.arange(NK)[None, :] % 37 == 5, -1, ids).astype(mx.int32)
SRC = r"""
    const uint sg = simdgroup_index_in_threadgroup, tid = thread_index_in_threadgroup;
    const uint t = threadgroup_position_in_grid.y;
    const uint nbase = threadgroup_position_in_grid.x * NBK;
    threadgroup half stage[NBK * KT];
    threadgroup int sid[NBK];
    for (uint j = tid; j < NBK; j += NTH) sid[j] = nbase + j < NK ? ids[ulong(t) * NK + nbase + j] : -1;
    constexpr auto desc = matmul2d_descriptor(HH, SGN, KT, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA((device half*)(Q + ulong(t) * HH * DD), dextents<int32_t, 2>(DD, HH));
    auto sA0 = tA.slice(0, 0);
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tB0((threadgroup half*)stage + sg * SGN * KT, dextents<int32_t, 2>(KT, SGN));
    auto cT = op.get_destination_cooperative_tensor<decltype(sA0), decltype(tB0), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;
    for (uint kt = 0; kt < DD; kt += KT) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = tid; c < NBK * KT / 8; c += NTH) {
            const uint r = c / (KT / 8), o = (c % (KT / 8)) * 8u;
            const int id = sid[r];
            *((threadgroup uint4*)(stage + r * KT + o)) = id >= 0 ? *((const device uint4*)(tab + ulong(id) * DD + kt + o)) : uint4(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sA = tA.slice(int(kt), 0);
        op.run(sA, tB0, cT);
    }
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) {
        const auto ix = cT.get_multidimensional_index(i);
        const uint n = nbase + sg * SGN + uint(ix[0]);
        if (n < NK) S[(ulong(t) * HH + uint(ix[1])) * NK + n] = cT[i] * SCALE;
    }
"""
scale = D ** -0.5
def mk(nsg, kt, sgn=32, abl=''):
    nbk = sgn * nsg
    s = SRC.replace("NBK", str(nbk)).replace("SGN", str(sgn)).replace("NTH", str(32 * nsg)).replace("KT", str(kt)).replace("HH", str(H)).replace("DD", str(D)).replace("NK", str(NK)).replace("SCALE", repr(scale) + "f")
    if abl == 'Atg':
        s = s.replace(f'threadgroup int sid[{nbk}];', f'threadgroup int sid[{nbk}]; threadgroup half aS[HH * KT];')
        s = s.replace('auto sA0 = tA.slice(0, 0);', 'tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tAs((threadgroup half*)aS, dextents<int32_t, 2>(KT, HH)); auto sA0 = tAs.slice(0, 0);')
        s = s.replace('        threadgroup_barrier(mem_flags::mem_threadgroup);\n        auto sA = tA.slice(int(kt), 0);\n        op.run(sA, tB0, cT);',
          '        for (uint c = tid; c < HH * KT / 8; c += NTH) { const uint r = c / (KT / 8), o = (c % (KT / 8)) * 8u; *((threadgroup uint4*)(aS + r * KT + o)) = *((const device uint4*)(Q + ulong(t) * HH * DD + r * DD + kt + o)); }\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n        op.run(tAs, tB0, cT);')
    if abl == 'nogather': s = s.replace('id >= 0 ? *((const device uint4*)(tab + ulong(id) * DD + kt + o)) : uint4(0)', 'uint4(id)')
    if abl == 'nomm': s = s.replace('op.run(sA, tB0, cT);', '')
    if abl == 'nostore': s = s.replace('if (n < NK) S[', 'if (n < NK && cT[i] == 12345.f) S[')
    s = s.replace("HH", str(H)).replace("DD", str(D)).replace("KT", str(kt)).replace("NTH", str(32 * nsg))
    k = mx.fast.metal_kernel(name=f"exp_k1_{nsg}_{kt}_{abl}", header=NX.HEADER_NAX, input_names=["Q", "tab", "ids"], output_names=["S"], source=s)
    return lambda: k(inputs=[qt, tab, ids], grid=(-(-NK // nbk) * 32 * nsg, TC, 1), threadgroup=(32 * nsg, 1, 1), output_shapes=[(TC * H, NK)], output_dtypes=[mx.float32])[0]
base = AP._kernels(H, D, NK, scale)[0]
fb = lambda: base(inputs=[qt, tab, ids], grid=(5 * 128, TC, 1), threadgroup=(128, 1, 1), output_shapes=[(TC * H, NK)], output_dtypes=[mx.float32])[0]
ref = fb(); mx.eval(ref)
arms = {"base": fb, 'a_nogather': mk(4, 64, 32, 'nogather'), 'a_nomm': mk(4, 64, 32, 'nomm'), 'a_nostore': mk(4, 64, 32, 'nostore'), 'Atg_1_64': mk(1, 64, 32, 'Atg'), 'Atg_2_64': mk(2, 64, 32, 'Atg'), 'Atg_4_64': mk(4, 64, 32, 'Atg'), 'Atg_4_32': mk(4, 32, 32, 'Atg') }
for sgn, nsg, kt in []:
    if sgn * nsg * kt * 2 + 4 * sgn * nsg > 32768: continue
    arms[f"sgn{sgn}_nsg{nsg}_kt{kt}"] = mk(nsg, kt, sgn)
flops = 2 * H * NK * D * TC
ok = {}
for k, f in arms.items():
    o = f(); mx.eval(o); ok[k] = float(mx.max(mx.abs(o - ref)).item())
best = {k: 1e9 for k in arms}
for r in range(8):
    time.sleep(1.0)
    ks = list(arms); ks = ks[r % len(ks):] + ks[:r % len(ks)]
    for k in ks:
        for _ in range(2):
            t0 = time.perf_counter(); mx.eval(arms[k]()); best[k] = min(best[k], time.perf_counter() - t0)
for k in arms: print(f"{k:12s} {best[k]*1e3:6.2f} ms {flops/best[k]/1e12:5.1f} TF  maxdiff {ok[k]:.2e}", flush=True)
