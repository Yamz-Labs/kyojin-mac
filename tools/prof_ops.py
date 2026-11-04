"""Per-op burst profile of one DS4.1 reuse layer (ds2 L1, routed experts stubbed) at T rows. Every op is evaluated alone after an idle gap (burst clock),
min of N; inputs are materialised first. Prints ms, bytes moved (in+out), the 153 GB/s floor, TFLOPs for GEMMs. usage: prof_ops.py PACK T [reps] [layer]"""
import sys, time, importlib, numpy as np, mlx.core as mx
from kyojin_mlx import patch as P, attn_prefill as AP
pack, T = sys.argv[1], int(sys.argv[2]); reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
LI = int(sys.argv[4]) if len(sys.argv) > 4 else 1
IDLE = 0.8; BW = 153e9
model, spec, cfg = P.build_model(pack); P.patch_model(model, pack, spec, cfg)
H = cfg["text_config"]["hidden_size"]
P._moe_rows = lambda moe, x2, ids2, w2, xsrc=None: mx.zeros((x2.shape[0], H), dtype=mx.float32)
lm = model.language_model
LM = importlib.import_module("mlx_vlm.models.deepseek_v41.language")
L = lm.layers[LI]; A = L.attn
rows = []
def nb(x):
    if isinstance(x, (list, tuple)): return sum(nb(v) for v in x)
    return x.nbytes if isinstance(x, mx.array) else 0
CHAIN = False
def op(name, fn, ins=(), flops=0, n=None):
    if not CHAIN: mx.eval(fn())
    best = 1e9
    for _ in range(1 if CHAIN else (n or reps)):
        if not CHAIN: time.sleep(IDLE)
        t0 = time.perf_counter(); o = fn(); mx.eval(o); best = min(best, time.perf_counter() - t0)
    b = nb(list(ins)) + nb(o)
    rows.append((name, best * 1e3, b / 1e6, b / BW * 1e3, flops / best / 1e12 if flops else 0))
    return o
def profile(h, pre_mix, cache):
    mx.eval(h, pre_mix)
    start = cache.offset; B, S = 1, h.shape[1]; Hh, D = A.n_heads, A.head_dim
    fq = LM.fake_quant_fp8_ue8m0
    attn_pre, attn_post, attn_comb = op("hc_mixes(attn)", lambda: L.hc_mixes(h, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base), [h], 2 * S * 4 * H * 24)
    mx.eval(attn_pre, attn_post, attn_comb)
    x = op("hc_pre(attn)", lambda: L.hc_pre(h, pre_mix), [h, pre_mix])
    x = op("attn_norm", lambda: L.attn_norm(x), [x])
    xq = op("fq(x) [attn in]", lambda: fq(x), [x])
    qa = op("wq_a", lambda: A.wq_a(xq), [xq], 2 * S * H * 1280)
    qr = op("q_norm", lambda: A.q_norm(qa), [qa])
    qrq = op("fq(qr)", lambda: fq(qr), [qr])
    qb = op("wq_b", lambda: A.wq_b(qrq), [qrq], 2 * S * 1280 * Hh * D)
    qv = qb.reshape(B, S, Hh, D).transpose(0, 2, 1, 3)
    q = op("q rope (strided read, [H,T,D] out)", lambda: A.rope(qv, start), [qb])
    # window part pieces
    kvl = op("wkv", lambda: A.wkv(xq), [xq], 2 * S * H * D)
    kvn = op("kv_norm", lambda: A.kv_norm(kvl), [kvl])
    kvr = op("kv rope", lambda: A.rope(kvn[:, None], start).reshape(B, S, D), [kvn])
    kvq = op("fq(kv)", lambda: fq(kvr), [kvr])
    window_kv, wmask = A._window_part(x, start, cache); mx.eval(window_kv)
    pool, idxs = A._compress_part(x, qr, start, cache); mx.eval(pool, idxs)
    out = op("sparse attention core", lambda: A._sparse_attention(q, window_kv, pool, idxs, wmask), [q, window_kv, pool, idxs])
    o2 = op("out rope inverse", lambda: A.rope(out, start, inverse=True), [out])
    og = op("o reshape+transpose (copy)", lambda: mx.contiguous(o2.reshape(B, A.o_groups, -1, S, D).transpose(0, 1, 3, 2, 4).flatten(-2)), [o2])
    oa = op("wo_a", lambda: A.wo_a(og), [og], 2 * S * Hh * D * 1024)
    ob = op("wo_a out transpose (copy)", lambda: mx.contiguous(oa.transpose(0, 2, 1, 3).flatten(-2)), [oa])
    obq = op("fq(o)", lambda: fq(ob), [ob])
    ao = op("wo_b", lambda: A.wo_b(obq), [obq], 2 * S * 8192 * H)
    x1 = op("hc_expand(attn)", lambda: LM.hc_expand(ao, h, attn_post, attn_comb), [ao, h])
    # ffn half
    ffn_pre, ffn_post, ffn_comb = op("hc_mixes(ffn)", lambda: L.hc_mixes(x1, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base), [x1], 2 * S * 4 * H * 24)
    mx.eval(ffn_pre, ffn_post, ffn_comb)
    y = op("hc_pre(ffn)", lambda: L.hc_pre(x1, attn_pre), [x1, attn_pre])
    y = op("ffn_norm", lambda: L.ffn_norm(y), [y])
    G = L.ffn.gate
    sc = op("router logits+score", lambda: mx.sqrt(LM.nn.softplus(y.astype(mx.float32) @ G.weight.T)), [y], 2 * S * H * G.weight.shape[0])
    inds = op("router topk (argpartition)", lambda: mx.argpartition(-(sc + G.bias), kth=G.top_k - 1, axis=-1)[..., :G.top_k].astype(mx.int32), [sc])
    wts = op("router weights", lambda: (lambda w: w / (w.sum(-1, keepdims=True) + 1e-20) * G.routed_scaling_factor)(mx.take_along_axis(sc, inds, axis=-1)), [sc, inds])
    yq = op("fq(y) [routed in]", lambda: fq(y), [y])
    moe = mx.zeros((1, S, H), mx.float32); mx.eval(moe)
    sh = L.ffn.shared_experts
    yq2 = yq
    g_ = op("shared gate_proj", lambda: sh.gate_proj(yq2), [yq2], 2 * S * H * 3072 if False else 0)
    u_ = op("shared up_proj", lambda: sh.up_proj(yq2), [yq2])
    act = op("shared swiglu", lambda: LM._swiglu(g_, u_, sh.swiglu_limit).astype(y.dtype), [g_, u_])
    actq = op("shared fq(act)", lambda: fq(act), [act])
    sd = op("shared down_proj", lambda: sh.down_proj(actq), [actq])
    comb = op("ffn combine (cast+add)", lambda: (moe.astype(mx.float32).sum(-2) if False else moe + sd.astype(mx.float32)).astype(y.dtype), [moe, sd])
    x2 = op("hc_expand(ffn)", lambda: LM.hc_expand(comb, x1, ffn_post, ffn_comb), [comb, x1])
    return x2
# one forward to fill the cache (L0 sources), hooked at layer LI
cls = L.__class__; orig = cls.__call__; done = []
def hooked(self, h, pre_mix, image_mask, cache, _o=orig):
    if self is L and not done:
        done.append(1)
        mx.eval(h, pre_mix)
        for _ in range(2):
            time.sleep(3); t0 = time.perf_counter(); mx.eval(_o(self, h, pre_mix, image_mask, cache)); whole = time.perf_counter() - t0
        global CHAIN
        profile(h, pre_mix, cache); burst = list(rows); rows.clear(); CHAIN = True
        time.sleep(3); profile(h, pre_mix, cache); chain = list(rows); rows[:] = burst
        print("sustained chain sum (ops back to back, each eval'd):", round(sum(r[1] for r in chain), 1), "ms")
        print("sustained per-op: " + ", ".join(f"{r[0]}={r[1]:.1f}" for r in chain))
        tot = sum(r[1] for r in rows)
        print(f"whole layer (stubbed experts, 3 s idle, last of 2): {whole*1e3:.1f} ms | sum of isolated ops {tot:.1f} ms | unattributed {whole*1e3-tot:.1f} ms")
        print(f"{'op':34s} {'ms':>8s} {'MB':>8s} {'floor ms':>9s} {'TF':>6s}")
        for n, ms, mb, fl, tf in rows: print(f"{n:34s} {ms:8.2f} {mb:8.1f} {fl:9.2f} {tf:6.2f}")
        print("peak GiB", round(mx.get_peak_memory() / 2**30, 2)); sys.exit(0)
    return _o(self, h, pre_mix, image_mask, cache)
L.__class__ = type(cls.__name__, (cls,), {"__call__": hooked})
ids = mx.array([np.random.default_rng(3).integers(1, 4000, T).tolist()])
cache = lm.make_cache(); mx.eval(lm(ids, cache=cache).logits)
