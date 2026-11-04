"""interleaved variants of exl3 prefill in one process vs mlx-lm. usage: pf3_var.py EXL3 MLX T C REPS var1,var2,... (vars: stock=seams off, full, s2stock, s1stock; env-like k=v via KYOJIN_ vars not supported)"""
import sys, os, time, mlx.core as mx
from transformers import AutoTokenizer
ep, mp, T, C, R = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]); vars_ = sys.argv[6].split(",")
mx.set_cache_limit(256 * 2 ** 20)
ids = AutoTokenizer.from_pretrained(ep).encode("The history of the Roman Empire begins with the founding of Rome in 753 BC. " * 4000)[:T]
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx import wcache as WC, layer_pf as LP, kernels as KN
be, _ = load_backend(ep)
import os
if os.environ.get("NODECODE"):
    _dc = {}; _dw = WC.decode_wr
    def _fake(lin, _dw=_dw):
        k = tuple(_dw.__code__.co_varnames[:0]) or (lin.out_features, lin.in_features)
        if lin.out_features > 100000: return _dw(lin)
        if k not in _dc: _dc[k] = _dw(lin); mx.eval(_dc[k])
        return _dc[k]
    WC.decode_wr = _fake
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
model, _ = load(mp); lm = model.language_model
s1, s2 = LP.seam1, LP.seam2
def s2stock(inner, svh, res):
    rows, N = inner.shape
    fp = mx.array([1, N, rows], dtype=mx.uint32); grid, tg = KN.S.had_grid(N // 128, rows)
    y = KN.fin_kernel(mx.float16, mx.float16)(inputs=[inner, svh, fp], template=[("OT", mx.float16)], grid=grid, threadgroup=tg, output_shapes=[(rows, N)], output_dtypes=[mx.float16])[0]
    return res + y
def s1stock(inner, svh, x, nw, eps, sg, su, pg, pu):
    rows, K = inner.shape
    fp = mx.array([1, K, rows], dtype=mx.uint32); grid, tg = KN.S.had_grid(K // 128, rows)
    y = KN.fin_kernel(mx.float16, mx.float16)(inputs=[inner, svh, fp], template=[("OT", mx.float16)], grid=grid, threadgroup=tg, output_shapes=[(rows, K)], output_dtypes=[mx.float16])[0]
    h = x + y
    n = mx.fast.rms_norm(h, nw, eps)
    return h, KN.rot(n, sg, mx.float16, perm=pg), KN.rot(n, su, mx.float16, perm=pu)
def s1fa(inner, svh, x, nw, eps, sg, su, pg, pu):
    h = s2(inner, svh, x)
    n = mx.fast.rms_norm(h, nw, eps)
    return h, KN.rot(n, sg, mx.float16, perm=pg), KN.rot(n, su, mx.float16, perm=pu)
_rot0, _rms0 = KN.rot, mx.fast.rms_norm
def setv(v):
    KN.rot = (lambda x, suh, dt=mx.float16, perm=False, **k: x.astype(dt)) if v in ("norot", "nonorm_norot") else _rot0
    mx.fast.rms_norm = (lambda x, w, eps: x) if v in ("nonorm", "nonorm_norot") else _rms0
    LP.seam1, LP.seam2, LP.MAX_ROWS, LP.SPLIT_MAX_ROWS = s1, s2, 1 << 20, 0
    if v in ("wc", "wcoff"): LP.MAX_ROWS = None
    if v == "wcoff": os.environ["KYOJIN_PF_WCACHE_GB"] = "0"
    else: os.environ.pop("KYOJIN_PF_WCACHE_GB", None)
    if v == "split": LP.MAX_ROWS, LP.SPLIT_MAX_ROWS = 0, 1 << 20
    if v == "stock": LP.MAX_ROWS = 0
    if v == "s2stock": LP.seam2 = s2stock
    if v == "s1stock": LP.seam1 = s1stock
    if v == "s1fa": LP.seam1 = s1fa
def run_e(v):
    setv(v); c = be.make_cache(); mx.synchronize(); t0 = time.perf_counter()
    with WC.prefill_scope(be.model, C, -(-T // C)):
        for s in range(0, T, C): mx.eval(be.prefill_chunk(c, ids[s:s + C]))
    return T / (time.perf_counter() - t0)
def run_m():
    c = make_prompt_cache(model); mx.synchronize(); t0 = time.perf_counter()
    for s in range(0, T, C):
        h = lm.model(mx.array(ids[s:s + C])[None], c)[:, -1:]
        lg = lm.model.embed_tokens.as_linear(h) if lm.args.tie_word_embeddings else lm.lm_head(h)
        mx.eval(lg, *[st for ci in c for st in ci.state if st is not None])
    return T / (time.perf_counter() - t0)
for v in vars_: run_e(v)
run_m()
res = {v: [] for v in vars_}; m = []
for r in range(R):
    order = vars_ if r % 2 == 0 else vars_[::-1]
    for v in order: res[v].append(run_e(v))
    m.append(run_m())
m.sort(); mm = m[len(m)//2]
print(f"T={T} C={C} mlx-lm med {mm:.0f}")
for v in vars_:
    x = sorted(res[v]); print(f"  {v:8s} max {x[-1]:.0f} med {x[len(x)//2]:.0f} ratio med {x[len(x)//2]/mm:.3f}")
