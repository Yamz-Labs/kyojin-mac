"""Capture the inputs of attn._sparse_attention (layer 0 and 1) at prefill T -> npz. usage: cap_attn.py PACK T OUT_PREFIX"""
import sys, numpy as np, mlx.core as mx
from kyojin_mlx import patch as P
pack, T, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
model, spec, cfg = P.build_model(pack); P.patch_model(model, pack, spec, cfg)
lm = model.language_model
store = {}
for i, l in enumerate(lm.layers):
    at = l.attn; cls = at.__class__; orig = cls._sparse_attention
    def f(self, q, window, pool, idxs, mask, _i=i, _o=orig):
        mx.eval(q, window, pool, idxs, mask)
        store[_i] = dict(q=np.array(q.astype(mx.float32)), q_dtype=str(q.dtype), window=np.array(window.astype(mx.float32)), wdt=str(window.dtype),
                         pool=np.array(pool.astype(mx.float32)), pdt=str(pool.dtype), idxs=np.array(idxs), mask=np.array(mask), sink=np.array(self.attn_sink), scale=self.scale)
        return _o(self, q, window, pool, idxs, mask)
    at.__class__ = type(cls.__name__, (cls,), {"_sparse_attention": f})
ids = mx.array([np.random.default_rng(3).integers(1, 4000, T).tolist()])
mx.eval(lm(ids, cache=lm.make_cache()).logits)
for i, d in store.items():
    np.savez(f"{out}_L{i}.npz", **d); print(i, {k: (v.shape, v.dtype) if hasattr(v, "shape") else v for k, v in d.items()})
