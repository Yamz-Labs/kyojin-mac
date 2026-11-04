import sys, importlib, numpy as np, mlx.core as mx
from kyojin_mlx import ds_pf_idx as I
L = importlib.import_module("mlx_vlm.models.deepseek_v41.language")
T, NH = 300, 32
rng = np.random.default_rng(0)
q = mx.array((rng.standard_normal((1, T, NH, 128)) * 2).astype(np.float16))
cos, sin = L._index_cos_sin(T, 64, 160000.0, (16, 32, 1, 65536))
pos = mx.arange(0, T)
ref = L.fake_quant_fp4_ue8m0(L._apply_index_rotary(q, cos[pos][None, :, None, :], sin[pos][None, :, None, :], 64))
out = I.fq4_rot(q, cos[pos], sin[pos], 64)
mx.eval(ref, out)
print("bit-identical:", bool((ref == out).all()), "mismatches", int((ref != out).sum()))
# scores
for (T, ratio, start) in [(300, 2, 0), (512, 1, 0), (200, 2, 100)]:
    Tc = (start + T) // ratio
    q = mx.array((rng.standard_normal((1, T, 32, 128))).astype(np.float16)); ik = mx.array(rng.standard_normal((1, Tc, 128)).astype(np.float16))
    w = mx.array((rng.standard_normal((1, T, 32)) * 0.05).astype(np.float16))
    clens = (mx.arange(start + 1, start + T + 1) // ratio)[:, None]
    ref = L._index_scores(q, ik, w, clens, None)
    out = I.scores_mm(q, ik, w, start, ratio, max_elems=1 << 20)
    mx.eval(ref, out)
    r, o = np.array(ref.astype(mx.float32))[0] if ref.ndim == 3 else np.array(ref.astype(mx.float32)), np.array(out)
    r = r.reshape(o.shape)
    fin = np.isfinite(r)
    print(T, ratio, start, "finite pattern equal", bool((fin == np.isfinite(o)).all()), "rel-rms", np.sqrt(((r[fin] - o[fin]) ** 2).mean() / (r[fin] ** 2).mean()))
    # top-k overlap
    k = min(512, Tc)
    ti = np.argsort(-np.where(fin, r, -1e30), axis=-1)[:, :k]; tn = np.argsort(-np.where(fin, o, -1e30), axis=-1)[:, :k]
    ov = np.mean([len(set(a) & set(b)) / k for a, b in zip(ti, tn)])
    print("   top-k overlap (finite rows part)", ov)
