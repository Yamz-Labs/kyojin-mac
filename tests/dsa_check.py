"""dsa_decode kernels vs the stock indexer ops: score values, top-k sets / order, candidate mask.  python tests/dsa_check.py"""
import os, sys
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.expanduser("~/kyojin-mlx/vlm_pkg"))
from kyojin_mlx import dsa_decode as D
from mlx_vlm.models.deepseek_v41 import language as L
from mlx_vlm.models.deepseek_v41.fakequant import fake_quant_fp4_ue8m0 as fq4

rng = np.random.default_rng(1)
bad = 0
for dt in (mx.float16, mx.bfloat16):
    for n in (1, 7, 8, 100, 513, 2049, 5000, 16385, 65536):
        cap = n + 37
        q = fq4(mx.array(rng.standard_normal((1, 1, 32, 128)).astype(np.float32)).astype(dt))
        ik = fq4(mx.array(rng.standard_normal((1, cap, 128)).astype(np.float32)).astype(dt))
        w = (mx.array(rng.standard_normal((1, 1, 32)).astype(np.float32)) * 0.05).astype(dt)
        cand = None
        if n > 3000: cand = L.select_candidate_blocks(mx.array(rng.standard_normal((1, n)).astype(np.float32)).astype(dt), n, 2048, 8)
        tile = mx.einsum("bsnd,btd->bsnt", q, ik[:, :n]); ref = (mx.maximum(tile, 0) * w[..., None]).sum(axis=2)
        if cand is not None: ref = mx.where(cand, ref, -mx.inf)
        nn = mx.array([n], dtype=mx.int32)
        sc = D.scores(q.reshape(32, 128), w.reshape(32), ik.reshape(cap, 128), nn, n, cand.reshape(-1) if cand is not None else None)[:n]
        topk = min(512, n)
        r_idx = mx.sort(mx.argsort(-ref, axis=-1)[..., :topk], axis=-1).astype(mx.int32).reshape(-1)
        m_idx = D.select(sc, nn, n, 512)
        r_idx2 = mx.sort(mx.argsort(-sc.reshape(1, -1), axis=-1)[..., :topk], axis=-1).astype(mx.int32).reshape(-1)   # same scores, stock top-k
        mx.eval(ref, sc, r_idx, m_idx, r_idx2)
        a, b = np.array(ref.astype(mx.float32)).reshape(-1), np.array(sc.astype(mx.float32)).reshape(-1)
        fin = np.isfinite(a)
        sdiff = np.abs(a[fin] - b[fin]).max() if fin.any() else 0.0
        eq_masked = ((~fin) == (~np.isfinite(b))).all()
        ok_sel = np.array_equal(np.array(r_idx2), np.array(m_idx))
        ov = len(set(np.array(r_idx).tolist()) & set(np.array(m_idx).tolist())) / topk
        line = f"{str(dt)[-8:]:9s} n={n:6d} score maxdiff {sdiff:.3g} (scale {np.abs(a[fin]).max() if fin.any() else 0:.3g}) mask_eq {eq_masked}  select==stock(same scores) {ok_sel}  overlap vs stock chain {ov:.4f}"
        if not ok_sel or not eq_masked: bad += 1; line += "  <-- FAIL"
        if n > 3000 and cand is None: pass
        print(line)
    # candidate mask
    for n in (50, 3000, 20000, 65536):
        s = mx.array(rng.standard_normal((1, n)).astype(np.float32)).astype(dt)
        if n > 100: s = mx.where(mx.array(rng.random((1, n)) < 0.2), -mx.inf, s)
        r = L.select_candidate_blocks(s, n, 2048, 8); m = D.candidate_mask(s.reshape(-1), mx.array([n], dtype=mx.int32), n, 2048)
        mx.eval(r, m); ok = np.array_equal(np.array(r).reshape(-1), np.array(m))
        print(f"{str(dt)[-8:]:9s} cand mask n={n:6d} equal {ok}")
        if not ok: bad += 1
print("FAIL" if bad else "ALL OK")
