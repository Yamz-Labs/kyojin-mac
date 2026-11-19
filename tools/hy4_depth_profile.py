"""Decode-step cost of one Hy4 kpool (DSA) layer by op class, versus context depth S (real shapes, synthetic data).

Why: the curves must stay flat up to 1M (>= 90 % of the short-context decode). Per-call mx.eval timing hides everything under the dispatch floor (~0.3 ms), so every
class is timed as K independent calls evaluated in ONE mx.eval (K=24): the floor is amortised and the number is the GPU + encode cost per call.
Classes: pk_update (pooled key of the new rows), score (relu(iq . pk) * w summed over heads) in fp32 / fp16 / int8 pooled keys, topk (radix, 511 pools), slots (gather_slots),
attend_gather (the 2048 kept keys + tail), layer (the whole layer as one compiled program, gather + pk cache, as served). Quality of the cheaper scorers: overlap of the
top-511 pool set with the fp32 scorer on planted data (clear winners) and on pure noise (worst case).
Usage (Metal, chain cell only): python tools/hy4_depth_profile.py [L=1] [S list, comma] ; prints one RESULTS line per S and a json file when HY4P_OUT is set."""
import os, sys, time, json
sys.dont_write_bytecode = True
os.environ["KYOJIN_METAL_TESTS"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/tests")
for ln in open(os.environ.get("HY4P_FLAGS", os.path.expanduser("/path/to/data"))):
    ln = ln.strip()
    if ln and not ln.startswith("#") and "=" in ln:
        k, v = ln.split("=", 1); os.environ.setdefault(k, v.strip().strip('"'))
import numpy as np, mlx.core as mx
from kyojin_mlx import fixed_kv as FK, foreign_skeleton_kpool as KP, flags as FL, idxfast as IX
from test_hy4_attn_slice import _kp_model
mx.set_default_device(mx.gpu)
FL.set_many({n: os.environ[f.env] for n, f in FL.FLAGS.items() if f.attrs and f.runtime and f.env in os.environ and os.environ[f.env] != f.default})
LQ = int(sys.argv[1]) if len(sys.argv) > 1 else 1
SS = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "1024,8192,32768,65536,131072,262144,524288,1048576").split(",")]
K = int(os.environ.get("HY4P_K", "24"))
rng = np.random.default_rng(1)
d = 4096
m = _kp_model(rng, d, 32, 512, 192, 64, 256, 16, 128)
m.swa, m.window = False, 0
kp = m.kp
KPP, DX, TOPK = kp["kp"], kp["dx"], 511
a = m.kv + m.r
sel = KP.KpoolSelect(m.kpool.ape, m.kpool.pool_norm, m.kpool.k_norm.eps, kp["kp"], kp["top_k"], kp["select_tail"], kp["n_rot"], kp["base"], True)


def T(fn, k=K, w=2):
    """ms per call: k variants of fn(i) evaluated in one mx.eval (i perturbs an input so no call can be shared)."""
    for _ in range(w):
        mx.eval([fn(i) for i in range(k)])
    t = time.perf_counter(); mx.eval([fn(i) for i in range(k)]); return (time.perf_counter() - t) / k * 1e3


def rows(S, J, planted):
    pk = mx.array(rng.standard_normal((1, J, DX)).astype(np.float32))
    iq = rng.standard_normal((1, 16, LQ, DX)).astype(np.float32)
    if planted:                                   # every head points at the mean of 8 target pools: a clear top set
        tgt = rng.choice(J, size=min(8, J), replace=False)
        base = np.array(pk[0])[tgt].mean(axis=0)
        iq = (base[None, None, None, :] + 0.5 * iq).astype(np.float32)
    return pk, mx.array(iq)


out_all = []
for S in SS:
    J = S // KPP
    n_top = min(TOPK, J)
    o = {"S": S, "L": LQ, "J": J}
    try:
        pk, iq = rows(S, J, False)
        w = mx.array(rng.standard_normal((1, LQ, 16)).astype(np.float32))
        wt = w.astype(mx.float32).transpose(0, 2, 1)[..., None]
        lens = mx.full((LQ,), J, dtype=mx.int32)
        buf = mx.array(0.5 * rng.standard_normal((1, 1, S, a + 2 * DX)).astype(np.float32)); off = mx.array([S - 2 * LQ - 8], dtype=mx.int32)
        mx.eval(pk, iq, w, buf)
        eps = lambda i: 1.0 + 1e-6 * i
        def score32(i, pk=pk): return (mx.maximum(iq * eps(i) @ pk[:, None].swapaxes(-1, -2), 0.0) * wt).sum(axis=1)
        pk16 = pk.astype(mx.float16); iq16 = iq.astype(mx.float16); mx.eval(pk16, iq16)
        def score16(i): return (mx.maximum(iq16 * eps(i) @ pk16[:, None].swapaxes(-1, -2), 0.0).astype(mx.float32) * wt).sum(axis=1)
        wq, sc_, bi_ = mx.quantize(pk[0], group_size=64, bits=8); mx.eval(wq, sc_, bi_)
        def score8(i):
            x = (iq[0] * eps(i)).reshape(16 * LQ, DX)
            kq = mx.quantized_matmul(x, wq, sc_, bi_, transpose=True, group_size=64, bits=8).reshape(16, LQ, J)
            return (mx.maximum(kq, 0.0) * wt[0]).sum(axis=0)[None]
        o["score_f32_ms"] = T(score32); o["score_f16_ms"] = T(score16); o["score_i8_ms"] = T(score8)
        s32 = score32(0); mx.eval(s32)
        o["topk_ms"] = T(lambda i: IX.topk_mt(s32.reshape(-1, J) + 1e-7 * i, lens, n_top))
        t = IX.topk_mt(s32.reshape(-1, J), lens, n_top).reshape(1, LQ, n_top); mx.eval(t)
        ok_pool = t >= 0; top = mx.where(ok_pool, t, 0)
        qpos = off[:, None] + KP.ar(LQ)[None, :]
        gstart = (qpos[..., None] // KPP) * KPP
        dct = dict(top=top, ok_pool=ok_pool, pad=mx.zeros((1,), dtype=mx.int32), qpos=qpos, gstart=gstart, J=J, S=S, kp=KPP)
        o["slots_ms"] = T(lambda i: KP.gather_slots({**dct, "qpos": qpos + (i * 0)})[0])
        slots, ok = KP.gather_slots(dct); mx.eval(slots, ok)
        lat_, kpe_ = buf[..., :m.kv], buf[..., m.kv:a]
        qn = mx.array(rng.standard_normal((1, 32, LQ, 192)).astype(np.float32)); qpe = mx.array(rng.standard_normal((1, 32, LQ, 64)).astype(np.float32))
        o["attend_gather_ms"] = T(lambda i: m.attend_gather(qn * eps(i), qpe, lat_, kpe_, slots, ok))
        pkc = mx.zeros((1, S // KPP + 8, DX), dtype=mx.float32)
        o["pk_update_ms"] = T(lambda i: sel.update_pk(pkc, buf, off + 0 * i, LQ, a, DX))
        # pk_update_ms above keeps pkc referenced between calls, so slice_update must copy the whole [1, J, DX] array (J x 512 B: 0.275 ms at 1M = a 128 MB copy, an artefact of the
        # harness). The served cache replaces cache[2] by the result, so the old buffer has no other owner: chain the updates (rebind, eval each) to see the real cost, and time a
        # tiny chained op for the per-eval floor.
        def chain(fn, k=24):
            p_ = mx.zeros((1, S // KPP + 8, DX), dtype=mx.float32); mx.eval(p_); t_ = time.perf_counter()
            for i in range(k):
                p_ = fn(p_); mx.eval(p_)
            return (time.perf_counter() - t_) / k * 1e3
        chain(lambda p_: sel.update_pk(p_, buf, off, LQ, a, DX), 4)
        o["pk_update_chain_ms"] = chain(lambda p_: sel.update_pk(p_, buf, off, LQ, a, DX))
        o["eval_floor_ms"] = chain(lambda p_: mx.slice_update(p_, p_[:, :1], mx.array([0], dtype=mx.int32), (1,)))
        # quality of the cheaper scorers (top-511 pool overlap with fp32), planted and noise
        for tag, planted in (("noise", False), ("planted", True)):
            pk_, iq_ = rows(S, J, planted)
            ref = (mx.maximum(iq_ @ pk_[:, None].swapaxes(-1, -2), 0.0) * wt).sum(axis=1)
            r_t = np.array(IX.topk_mt(ref.reshape(-1, J), lens, n_top)[0])
            p16 = (mx.maximum(iq_.astype(mx.float16) @ pk_.astype(mx.float16)[:, None].swapaxes(-1, -2), 0.0).astype(mx.float32) * wt).sum(axis=1)
            q_, s_, b_ = mx.quantize(pk_[0], group_size=64, bits=8)
            kq8 = mx.quantized_matmul(iq_[0].reshape(16 * LQ, DX), q_, s_, b_, transpose=True, group_size=64, bits=8).reshape(16, LQ, J)
            p8 = (mx.maximum(kq8, 0.0) * wt[0]).sum(axis=0)[None]
            for nm, sc in (("f16", p16), ("i8", p8)):
                c_t = np.array(IX.topk_mt(sc.reshape(-1, J), lens, n_top)[0])
                o[f"overlap_{nm}_{tag}"] = round(len(set(r_t[r_t >= 0].tolist()) & set(c_t[c_t >= 0].tolist())) / max(1, int((r_t >= 0).sum())), 4)
        # the whole layer as one compiled program, gather + pk cache (as served), at depth S
        os.environ["KYOJIN_HY4_FIXED_GATHER"] = os.environ["KYOJIN_HY4_FIXED_GATHER_DEC"] = "1"
        x = mx.array(rng.standard_normal((1, LQ, d)).astype(np.float32))
        cc = FK.FixedKVCache(a + 2 * DX, KPP, 0, 2048, pk_dx=DX); cc.ensure(S, x.dtype)
        cc.cache[0] = mx.zeros(tuple(cc.cache[0].shape), dtype=mx.float32); cc.cache[2] = mx.array(rng.standard_normal(tuple(cc.cache[2].shape)).astype(np.float32))
        cc.cache[1] = mx.array([S - 2 * LQ - 8], dtype=mx.int32); cc._n = S - 2 * LQ - 8
        def fn(x_, b_):
            sh = FK.FixedKVCache.__new__(FK.FixedKVCache); sh.__dict__.update(cc.__dict__); sh.cache = list(b_)
            return (m(x_, mask=None, cache=sh), *sh.cache)
        cf = mx.compile(fn)
        o["layer_ms"] = T(lambda i: cf(x * eps(i), cc.cache)[0], k=12)
    except Exception as e:
        o["error"] = repr(e)[:300]
    out_all.append(o)
    print("RESULTS hy4_depth_profile " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items()}), flush=True)
    del o
    mx.clear_cache()
if os.environ.get("HY4P_OUT"):
    json.dump(out_all, open(os.environ["HY4P_OUT"], "w"), indent=1)
