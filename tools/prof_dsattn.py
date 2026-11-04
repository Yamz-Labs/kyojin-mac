"""Stage times of one DS4.1 attention layer at prefill (each stage evaluated alone).
  python tools/prof_dsattn.py PACK T [layer=0] [reps=3]"""
import sys, time, numpy as np, mlx.core as mx
from kyojin_mlx import patch as P
import importlib
L = importlib.import_module("mlx_vlm.models.deepseek_v41.language")
V4 = importlib.import_module("mlx_vlm.models.deepseek_v4.language")
pack, T = sys.argv[1], int(sys.argv[2]); li = int(sys.argv[3]) if len(sys.argv) > 3 else 0; reps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
model, spec, cfg = P.build_model(pack); P.patch_model(model, pack, spec, cfg)
lm = model.language_model
attn = lm.model.layers[li].attn if hasattr(lm, "model") else lm.layers[li].attn
fq = L.fake_quant_fp8_ue8m0
rng = np.random.default_rng(0)
x = mx.array((rng.standard_normal((1, T, 5120)) * 1.0).astype(np.float16)); mx.eval(x)
rec = {}
def st(name, fn):
    t0 = time.perf_counter(); r = fn(); mx.eval(r); rec[name] = rec.get(name, 0) + time.perf_counter() - t0; return r
def run(cache, x):
    self = attn; sp = 0; b, s = 1, T
    qr = st("q_lin_a", lambda: self.q_norm(self.wq_a(fq(x))))
    q = st("q_lin_b", lambda: self.wq_b(fq(qr)).reshape(b, s, self.n_heads, self.head_dim).transpose(0, 2, 1, 3))
    q = st("q_rope", lambda: self.rope(q, sp))
    wk, wm = st("window_kv", lambda: self._window_part(x, sp, cache))
    out = None
    if self.compress_ratio:
        # compressor
        ratio = self.compress_ratio; latent = None
        if self.is_kv_source:
            latent = st("compressor", lambda: self.compressor(x, sp, cache))
            def pk():
                n = latent.shape[1]; pos = (sp // ratio + mx.arange(n)) * ratio
                pv = L._apply_rope_at_positions(latent, pos, self.config.qk_rope_head_dim, self._latent_theta, self._latent_yarn)
                pv = L.fake_quant_fp4_e4m3(pv); cache.append_compressed(self.layer_idx, pv); return pv
            st("pool_build", pk)
            cache.compress_kv = cache.compress[self.layer_idx]
        cl = (sp + s) // ratio
        pool = cache.compress_kv[:b, :cl]
        if self.is_index_source:
            ix = self.indexer; end = sp + s
            if ix.owns_k:
                if latent is not None: st("idx_keys", lambda: ix._publish_keys(latent, sp, cache))
                cache.index_k = cache.keys[ix.layer_idx]
            cos, sin = L._index_cos_sin(end, ix.rope_head_dim, ix.rope_theta, ix.yarn)
            positions = mx.arange(sp, end)
            qf = st("idx_q_fq8", lambda: fq(qr))
            ql = st("idx_q_lin", lambda: ix.wq_b(qf).reshape(b, s, ix.n_heads, ix.index_head_dim))
            qrt = st("idx_q_rot", lambda: L._apply_index_rotary(ql, cos[positions][None, :, None, :], sin[positions][None, :, None, :], ix.rope_head_dim))
            qi = st("idx_q_fq4", lambda: L.fake_quant_fp4_ue8m0(qrt))
            ik = cache.index_k[:b, : end // ratio]
            w = st("idx_w", lambda: ix.weights_proj(x) * (ix.scale * ix.n_heads ** -0.5))
            clens = (mx.arange(sp + 1, end + 1) // ratio)[:, None]
            sc = st("idx_scores", lambda: L._index_scores(qi, ik, w, clens, cache.candidates if ix.uses_candidates else None))
            if ix.is_candidate_source:
                cache.candidates = st("idx_cand", lambda: L.select_candidate_blocks(sc, clens, ix.candidate_topk_blocks, ix.candidate_block_size))
            tk = min(ix.index_topk, end // ratio)
            idxs = st("idx_topk", lambda: mx.where((m := mx.sort(mx.argsort(-sc, axis=-1)[..., :tk], axis=-1).astype(mx.int32)) < clens, m, -1).astype(mx.int32))
            cache.topk_idxs = idxs
        else:
            idxs = cache.topk_idxs
        if pool.shape[1]:
            out = st("sparse_attn", lambda: self._sparse_attention(q, wk, pool, idxs, wm))
    if out is None:
        kv = wk[:, None].astype(q.dtype)
        out = st("sdpa_window", lambda: mx.fast.scaled_dot_product_attention(q, kv, kv, scale=self.scale, mask=wm, sinks=self.attn_sink.astype(q.dtype)))
    out = st("o_rope", lambda: self.rope(out, sp, inverse=True))
    def oo():
        o = out.reshape(b, self.o_groups, -1, s, self.head_dim).transpose(0, 1, 3, 2, 4).flatten(-2)
        o = self.wo_a(o); o = o.transpose(0, 2, 1, 3).flatten(-2); return self.wo_b(fq(o))
    return st("o_lin", oo)
allr = []
for i in range(reps + 1):
    cache = lm.make_cache()[0]
    for j in range(li):              # earlier layers (pool / top-k sources) run untimed
        mx.eval(cache.map(lm.layers[j].attn._attention, x))
    rec.clear()
    cache.map(lambda c, xx: run(c, xx), x)
    if i: allr.append(dict(rec))
best = {k: min(r[k] for r in allr) for k in allr[0]}          # per-stage minimum over the repetitions (the Air is shared: totals are noisy)
print(f"{pack.split('/')[-1]} layer {li} T={T} sum of stage minima {1e3 * sum(best.values()):.1f} ms")
for k, v in best.items(): print(f"  {k:12s} {1e3 * v:8.1f} ms")
tt = []
for i in range(reps + 1):
    cache = lm.make_cache()[0]
    for j in range(li):
        mx.eval(cache.map(lm.layers[j].attn._attention, x))
    t0 = time.perf_counter(); mx.eval(cache.map(attn._attention, x)); tt.append(time.perf_counter() - t0)
print(f"  attention layer as a whole (patched path, no stage syncs): {1e3 * min(tt[1:]):.1f} ms")
