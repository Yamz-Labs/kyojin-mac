"""Gate of the prefill sparse attention on a real model slice: inside every layer call, stock `_sparse_attention` vs `ds_pf_attn.sparse_pf` on the SAME inputs.
  python tools/dspf_layer_check.py PACK T [start_chunks=1]
Prints per layer the relative rms / max abs difference of the attention core output and the share of rows whose per-head argmax over D differs. Chunked prefill
(start > 0) is exercised with start_chunks > 1 (the chunk size is T // start_chunks)."""
import sys, importlib, numpy as np, mlx.core as mx
from kyojin_mlx import patch as P, ds_pf_attn as A
pack, T = sys.argv[1], int(sys.argv[2]); nch = int(sys.argv[3]) if len(sys.argv) > 3 else 1
model, spec, cfg = P.build_model(pack); P.patch_model(model, pack, spec, cfg)
lm = model.language_model
V4 = importlib.import_module("mlx_vlm.models.deepseek_v4.language")
cls = lm.layers[0].attn.__class__
cur = cls._sparse_attention
res = []
def cmp(self, q, window, pool, indices, wm):
    ref = V4._sparse_pooled_attention(q.astype(mx.float32), window[:, None], pool, indices, wm, (indices != -1)[:, None], self.scale, self.attn_sink.astype(mx.float32)).astype(q.dtype)
    out = A.sparse_pf(q, window, pool, indices, wm, self.scale, self.attn_sink, self.window_size)
    r, o = np.array(ref.astype(mx.float32)), np.array(out.astype(mx.float32))
    res.append((self.layer_idx, q.shape[2], float(np.sqrt(((r - o) ** 2).mean() / (r ** 2).mean())), float(np.abs(r - o).max()), float((r.argmax(-1) != o.argmax(-1)).mean())))
    return ref
cls._sparse_attention = cmp
mod = importlib.import_module(cls.__module__)
ix_new = mod.Indexer.__call__
ix_res = []
def ix_cmp(self, x, qr, latent, start_pos, offset, cache):
    new = ix_new(self, x, qr, latent, start_pos, offset, cache)
    ref = ix_orig(self, x, qr, None, start_pos, offset, cache)                    # keys already published by the first call
    a, b = np.array(new)[0], np.array(ref)[0]
    same = np.mean([len(set(r1[r1 >= 0]) & set(r2[r2 >= 0])) / max(1, len(set(r2[r2 >= 0]))) for r1, r2 in zip(a, b)])
    ix_res.append((self.layer_idx, a.shape[0], float(same), float((a == b).all(axis=-1).mean())))
    return new
ix_orig = None
# the unpatched Indexer.__call__ is the one wrapped by ds_pf_idx: its original is the closure cell
ix_orig = ix_new.__closure__[[c.cell_contents.__name__ if callable(c.cell_contents) and hasattr(c.cell_contents, '__name__') else '' for c in ix_new.__closure__].index('__call__')].cell_contents
mod.Indexer.__call__ = ix_cmp
ids = mx.array([np.random.default_rng(0).integers(100, 20000, T).tolist()])
cache = lm.make_cache(); n = T // nch
for c in range(nch):
    mx.eval(lm(ids[:, c * n:(c + 1) * n], cache=cache).logits)
for r in res: print(f"layer {r[0]} rows {r[1]} rel-rms {r[2]:.3g} maxabs {r[3]:.3g} argmax(D) differs {100 * r[4]:.2f} %")
for r in ix_res: print(f"indexer layer {r[0]} rows {r[1]} top-k set overlap {100 * r[2]:.3f} %  rows identical {100 * r[3]:.1f} %")
