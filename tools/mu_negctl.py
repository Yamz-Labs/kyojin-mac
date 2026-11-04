"""Negative controls for mu_check: switch one exactness measure off and expect bitwise mismatches (the check can fail).
usage: mu_negctl.py stock_matmul|unpinned_moe|stock_sdpa|padded_attend|padded_scores|stock_pool PACK [mu_check args...]
padded_attend / padded_scores / stock_pool replace one kernel of the batched DS4.1 core (kyojin_mlx.dsv4_core) by the same math on stock MLX ops over
the padded, stacked rows (summation order follows the shapes, so a row changes with the number and the lengths of the other rows)."""
import sys, runpy
import mlx.core as mx
from kyojin_mlx import rows as R, patch as P

which = sys.argv.pop(1)
pack = sys.argv[1]
if which == "stock_matmul":
    R.rowmm = lambda x, w: x @ w.T
    import kyojin_mlx.rowinv as RI
    RI.rowmm = R.rowmm
elif which == "unpinned_moe":
    orig = R.enable
    def enable(model, spec, union=False):
        rep = orig(model, spec, union)
        for _, m in model.language_model.named_modules():
            mo = getattr(m, "moe", None)
            if isinstance(mo, P.M.MoeGather):
                mo.pin_rows = None; mo._p.clear()
        return rep
    R.enable = enable
elif which == "stock_sdpa":
    R.SITES["mla_attend_rows"] = lambda self, orig, *a, **k: orig(self, *a, **k)
elif which == "padded_attend":
    from kyojin_mlx import dsv4_core as C
    def attend(q, win, pool, idx, pos, sink, scale, max_topk, sess=None):
        N, H, D = q.shape
        W, K = win.shape[1], idx.shape[1]
        ppos = (pos - W + 1)[:, None] + mx.arange(W)[None]
        kvw = mx.take_along_axis(win, (mx.maximum(ppos, 0) % W)[..., None], axis=1)
        kvp = mx.take_along_axis(pool, mx.maximum(idx, 0)[..., None], axis=1)
        kv = mx.concatenate([kvw, kvp], axis=1).astype(mx.float32)
        ok = mx.concatenate([ppos >= 0, idx >= 0], axis=1)
        s = mx.einsum("nhd,nmd->nhm", q.astype(mx.float32) * scale, kv)
        s = mx.where(ok[:, None, :], s, -mx.inf)
        m = mx.maximum(s.max(-1, keepdims=True), sink[None, :, None])
        e = mx.exp(s - m)
        l = e.sum(-1, keepdims=True) + mx.exp(sink[None, :, None] - m)
        return (mx.einsum("nhm,nmd->nhd", e, kv) / l).astype(q.dtype)
    C.attend = attend
elif which == "padded_scores":
    from kyojin_mlx import dsv4_core as C
    from mlx_vlm.models.deepseek_v41.language import _index_scores
    def index_scores(q, keys, wts, lens, t_max, cand=None, sess=None):
        sc = _index_scores(q[:, None], keys[:, :t_max], wts[:, None], lens[:, None, None] if False else lens[:, None], None if cand is None else cand[:, None, :t_max])
        return sc[:, 0]
    C.index_scores = index_scores
elif which == "stock_pool":
    from kyojin_mlx import dsv4_core as C
    C.pool_groups = lambda kvs, scs, rows: (kvs[rows] * mx.softmax(scs[rows], axis=1)).sum(axis=1)
sys.argv[0] = "mu_check.py"
runpy.run_path("tools/mu_check.py", run_name="__main__")
