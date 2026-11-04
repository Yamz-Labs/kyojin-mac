"""Gate for metal_nax_sh: (1) synthetic linears K=1..8 x cb mul1/mcg x odd rows: SH == coop bit for bit; (2) model logits SH on/off bit for bit.
usage: gate.py [MODEL_DIR ...]"""
import sys, itertools, numpy as np, mlx.core as mx
from kyojin_mlx import exl3_linear as EL, kernels as KN, metal_nax as NX, metal_nax_sh as SH
bad = 0
for K, cb, (kin, nout), R in itertools.product((1, 2, 3, 4, 5, 6, 8), ("mul1", "mcg"), ((1024, 512), (2560, 96)), (17, 64, 200, 512)):
    rng = np.random.default_rng(K)
    kt, nt = kin // 16, nout // 16
    tr = mx.array(rng.integers(-32768, 32768, (kt, nt, 16 * K), dtype=np.int16))
    f16 = lambda n: mx.array((rng.choice([-1.0, 1.0], n) * rng.uniform(0.5, 1.5, n)).astype(np.float16))
    try:
        lin = EL.Exl3Linear(tr, f16(kin), f16(nout), cb=cb)
    except Exception as e:
        continue
    st = NX.SingleStack(lin._words, lin.K, lin._kt, lin._nt, lin.cb); eids = mx.array([0], dtype=mx.int32)
    x = (mx.random.normal((R, kin)) * 0.5).astype(mx.float16); xh = KN.rot(x, lin._suh, mx.float16, perm=True)
    a = NX.expert_gemm((st,), eids, mx.array([0, R], dtype=mx.uint32), xh, R, R, (32, 2, 32, 64))
    for nsg, nq in ((4, 2), (8, 2), (4, 1)):
        if kt % (nsg * nq): continue
        b = SH.dense_sh(st, xh, R, (64, 1), True, "", nq, False, nsg)
        d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()
        if d != 0: bad += 1; print("DIFF", K, cb, kin, nout, R, nsg, nq, d)
print("synthetic: bad =", bad, flush=True)
if len(sys.argv) > 1:
    from kyojin_mlx.serve.backends import load_backend
    mx.set_cache_limit(256 * 2 ** 20)
    for path in sys.argv[1:]:
        be, _ = load_backend(path)
        ids = [(i * 7919) % 5000 + 100 for i in range(1024)]
        outs = []
        for sh in (False, True):
            EL.DENSE_SH = sh
            c = be.make_cache(); lg = None
            for i in range(0, 1024, 512):
                lg = be.prefill_chunk(c, ids[i:i + 512])
            mx.eval(lg); outs.append(lg.astype(mx.float32))
        print(path.split("/")[-1], "logits max |SH - coop| =", mx.abs(outs[0] - outs[1]).max().item(), flush=True)
