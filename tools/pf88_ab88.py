"""Interleaved whole-model prefill A/B of module switches on the served call (backend.prefill_chunk), exact-equality check of the last logits.
usage: ab88.py EXL3_DIR LEVER T chunk [rounds=6]   LEVER: remap | decmm"""
import sys, time, statistics, mlx.core as mx
exl3, lever, T, ch = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); R = int(sys.argv[5]) if len(sys.argv) > 5 else 6
mx.set_cache_limit(256 * 2 ** 20)
from transformers import AutoTokenizer
ids = AutoTokenizer.from_pretrained(exl3).encode("The history of the Roman Empire begins with the founding of Rome in 753 BC. " * 1200)[:T]
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx import wcache as WC, metal_src as S, exl3_linear as EL, metal_nax as NX
be, _ = load_backend(exl3)
LINS = [q for q in be.model.modules() if type(q) is EL.Exl3Linear]
def setarm(a):
    if lever == "remap": S.DEC_REMAP = (a == "lever")
    if lever == "decmm":                                   # lever = decode + MLX matmul for every row count, stock = the served policy (fused up to DENSE_FUSED_MAX)
        for q in LINS: q.fused_max = 0 if a == "lever" else 1024
last = {}
def go(a):
    setarm(a); c = be.make_cache(); mx.synchronize(); t0 = time.perf_counter()
    with WC.prefill_scope(be.model, ch, -(-T // ch)):
        for i in range(0, T, ch): lg = be.prefill_chunk(c, ids[i:i + ch])
    dt = time.perf_counter() - t0; last[a] = lg
    return T / dt
for a in ("stock", "lever"): go(a)
res = {"stock": [], "lever": []}
for r in range(R):
    for a in (("stock", "lever") if r % 2 == 0 else ("lever", "stock")): res[a].append(go(a))
exact = bool(mx.array_equal(last["stock"], last["lever"]))
f = lambda v: sorted(v, reverse=True)
print(f"{lever} T={T} chunk={ch}: stock best {f(res['stock'])[0]:.0f} p25 {f(res['stock'])[R//4]:.0f} med {statistics.median(res['stock']):.0f} | lever best {f(res['lever'])[0]:.0f} p25 {f(res['lever'])[R//4]:.0f} med {statistics.median(res['lever']):.0f} | best x{f(res['lever'])[0]/f(res['stock'])[0]:.3f} p25 x{f(res['lever'])[R//4]/f(res['stock'])[R//4]:.3f} | logits exact={exact}", flush=True)
