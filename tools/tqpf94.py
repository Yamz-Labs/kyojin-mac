"""tqpf94.py DEPTHS [T] [LAYERS]: per-op prefill table of one chunk of T rows at depth L (keys after the chunk), TQ KV vs fp16 KV, Qwen3.5 4B / 9B
attention shape (Hq 16, Hk 4, D 256; both packs share it), synthetic state. ms per attention layer (x LAYERS = the chunk's share of the model):
  fp16: cache write (KVCache.update_and_fetch) + sdpa causal
  tq:   write (rotate + quantize + pool write + ring), attend old (kvwin score path, KYOJIN_TQ_PF=0), attend new (_prefill), its parts (q rotation,
        pool dequant, concat, fused sdpa), plus max |new - old| / |old| on the output.
Run through macrun (the old score path holds [Hq, T, L] fp16 scores: 2 GiB at 32K)."""
import sys, time, os
import mlx.core as mx
from mlx_lm.models.cache import KVCache
from kyojin_mlx import tq

DEPTHS = [int(x) for x in sys.argv[1].split(",")]
T = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
NL = int(sys.argv[3]) if len(sys.argv) > 3 else 8
B, Hq, Hk, D = 1, 16, 4, 256
dt = mx.bfloat16
scale = D ** -0.5


def tm(f, n=3):
    mx.eval(f())
    ts = []
    for _ in range(n):
        a = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - a)
    return min(ts) * 1e3


def chunk(i):
    mx.random.seed(i)
    return (mx.random.normal((B, Hk, T, D)) * 0.5).astype(dt), mx.random.normal((B, Hk, T, D)).astype(dt), mx.random.normal((B, Hq, T, D)).astype(dt)


for L in DEPTHS:
    n = L // T
    q = None
    fc, tc = KVCache(), tq.TQKVCache(6, 6, 4, 256)
    wf = wt = 0.0
    for i in range(n):
        k, v, q = chunk(i)
        mx.eval(k, v, q)
        a = time.perf_counter(); r = fc.update_and_fetch(k, v); mx.eval(r); f = time.perf_counter() - a
        a = time.perf_counter(); r = tc.update_and_fetch(k, v); mx.eval(r, *tc.kq, *tc.vq, tc.rk, tc.rv); w = time.perf_counter() - a
        if i >= n - 2:
            wf, wt = max(wf, f * 1e3) if i == n - 2 else min(wf, f * 1e3), max(wt, w * 1e3) if i == n - 2 else min(wt, w * 1e3)
    kf, vf = fc.keys[..., :fc.offset, :], fc.values[..., :fc.offset, :]
    t_f = tm(lambda: mx.fast.scaled_dot_product_attention(q, kf, vf, scale=scale, mask="causal"))
    ref = mx.fast.scaled_dot_product_attention(q, kf, vf, scale=scale, mask="causal")
    rel = lambda o, r: (mx.abs(o.astype(mx.float32) - r.astype(mx.float32)).max() / mx.abs(r.astype(mx.float32)).max()).item()
    res = {}
    for pf in (0, 2, 1):
        tq.PF = pf
        o = tc.attend(q, scale); mx.eval(o)
        res[pf] = (tm(lambda: tc.attend(q, scale), 2 if pf == 0 else 3), o)
    old = res[0][1]
    t_rot = tm(lambda: tq.rotate(q, D).astype(dt) * 1)
    t_unr = tm(lambda: tq.unrotate(q, D).astype(dt) * 1)
    rk, rv, _ = tc._prev; ck, cv = tc._chunk
    S, c0 = 4, L - T - rk.shape[2]
    t_dq = tm(lambda: tc._deq_rows(S, c0))
    dk, dv = tc._deq_rows(S, c0); mx.eval(dk, dv)
    t_cat = tm(lambda: (mx.concatenate([tc.sk.astype(dt), dk.astype(dt), rk, ck], axis=-2), mx.concatenate([tc.sv.astype(dt), dv.astype(dt), rv, cv], axis=-2)))
    kk = mx.concatenate([tc.sk.astype(dt), dk.astype(dt), rk, ck], axis=-2); mx.eval(kk)
    t_sd = tm(lambda: mx.fast.scaled_dot_product_attention(q, kk, kk, scale=scale, mask="causal"))
    print(f"L {L:6d} T {T} ms/layer: fp16 write {wf:.2f} sdpa {t_f:.2f} | tq write {wt:.2f} attend old {res[0][0]:.2f} mask(PF2) {res[2][0]:.2f} causal(PF1) {res[1][0]:.2f} "
          f"(rot q+o {t_rot + t_unr:.2f} deq {t_dq:.2f} cat {t_cat:.2f} sdpa {t_sd:.2f}) | x{NL} layers fp16 {(wf + t_f) * NL:.0f} tq old {(wt + res[0][0]) * NL:.0f} "
          f"PF2 {(wt + res[2][0]) * NL:.0f} PF1 {(wt + res[1][0]) * NL:.0f} ms | rel err vs old: PF2 {rel(res[2][1], old):.1e} PF1 {rel(res[1][1], old):.1e}; "
          f"vs fp16 KV: old {rel(old, ref):.1e} PF2 {rel(res[2][1], ref):.1e} PF1 {rel(res[1][1], ref):.1e}", flush=True)
    del res, o, kk, dk, dv, ref
    del fc, tc, old, kf, vf
    tq._PFMASK.clear()
