"""DS4.1 prefill indexer stages alone (stock ops): python tools/bench_ds_idx.py T [ratio=2] [reps=3]"""
import sys, time, importlib, numpy as np, mlx.core as mx
L = importlib.import_module("mlx_vlm.models.deepseek_v41.language")
T = int(sys.argv[1]); ratio = int(sys.argv[2]) if len(sys.argv) > 2 else 2; reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
Tc = T // ratio; nh, dh = 32, 128
rng = np.random.default_rng(0)
q = mx.array(rng.standard_normal((1, T, nh, dh)).astype(np.float16)); ik = mx.array(rng.standard_normal((1, Tc, dh)).astype(np.float16))
w = mx.array(rng.standard_normal((1, T, nh)).astype(np.float16) * 0.05); clens = (mx.arange(1, T + 1) // ratio)[:, None]; mx.eval(q, ik, w, clens)
rec = {}
def st(n, fn):
    t0 = time.perf_counter(); r = fn(); mx.eval(r); rec[n] = min(rec.get(n, 9e9), time.perf_counter() - t0); return r
for _ in range(reps + 1):
    sc = st("scores", lambda: L._index_scores(q, ik, w, clens, None))
    st("cand_blocks", lambda: L.select_candidate_blocks(sc, clens, 2048, 8))
    am = st("argsort", lambda: mx.argsort(-sc, axis=-1)[..., :512])
    st("sort", lambda: mx.sort(am, axis=-1))
    st("argpartition", lambda: mx.argpartition(-sc, 512, axis=-1)[..., :512])
print(f"indexer T={T} ratio={ratio} Tc={Tc}: " + "  ".join(f"{k} {1e3 * v:.1f}" for k, v in rec.items()) + " ms")
from kyojin_mlx import ds_pf_idx as I
cosr, sinr = L._index_cos_sin(T, 64, 160000.0, (16, 32, 1, 65536)); pos = mx.arange(0, T)
ts = {"scores_mm": [], "fq4rot": [], "stock_fq4rot": []}
for _ in range(reps + 1):
    for n, fn in (("scores_mm", lambda: I.scores_mm(q, ik, w, 0, ratio)), ("fq4rot", lambda: I.fq4_rot(q, cosr[pos], sinr[pos], 64)),
                  ("stock_fq4rot", lambda: L.fake_quant_fp4_ue8m0(L._apply_index_rotary(q, cosr[pos][None, :, None, :], sinr[pos][None, :, None, :], 64)))):
        t0 = time.perf_counter(); mx.eval(fn()); ts[n].append(time.perf_counter() - t0)
print("  new: " + "  ".join(f"{n} {1e3 * min(v[1:]):.1f}" for n, v in ts.items()) + " ms")
